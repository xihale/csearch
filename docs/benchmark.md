# csearch 测试报告

日期：2026-10-29（单次运行）。硬件 RTX 4060 Laptop 8GB，模型 EmbeddingGemma-2（ONNX q4f16，768 维）。

被测仓库：`reimbursement-back-end`（Spring Boot 多模块 + web 前端 + Go ocrd 服务，中文注释密集）。
全仓索引规模 **959 文件 / 5713 chunks**；整体效率测试使用其 `apps/api` 子树副本（751 文件 / 4463 chunks），不触碰原仓库。

## TL;DR

| 维度 | 结果 |
|---|---|
| 正确率（agent 答题，8 题） | 带/不带 codesearch **打平：8/8 vs 8/8** |
| 单题上下文 | **-57%**（~7.5K vs ~17.5K tokens） |
| 单题耗时 | **-23%**（89.7s vs 116s 均值） |
| 8 题整批完成 | **-29%**（139s vs 195s） |
| 检索噪声 | **-97%**（595 vs 23,353 字符/题） |
| 工具自身 | 热查询 p50 **0.27s**；改动自动刷新 **<1s**；冷进程 7–12s；内存 ~1.9 GiB |
| 本地小模型（Qwen3-4B） | 宽判正确率 **3/8 → 5/8**，严格 **1/8 → 4/8**，prompt tokens **-49%**——模型越弱，codesearch 收益越大 |

价值公式：语义检索不提升找到答案的能力，省的是**读无关代码的 token 和注意力**。收益随问题数量线性放大——一个 30 题的熟悉仓库 session，上下文差距约 300K tokens。

## 测试 1：agent 对照实验（带 / 不带 codesearch）

**方法**：8 个 PM 风格探索题（审核状态流转、政策计算、配额占用、签章、OCR 接入、token 机制、操作日志、批量审批进度）。每题一个 scout 子 agent，同模型（GLM-5.3-Flash；免费档 deepseek 路由限流后两臂统一改道，保持公平）、同报告格式（`ANSWER: 文件:行号 | 事实` + 步骤清单）。

- **Arm A**：只准 Grep/Read/Glob
- **Arm B**：必须先 `search_code`（root+问题原文），之后只准 Read；禁止 Grep/Glob

Ground truth 由实验设计者事先钉死。**偏差披露**：出题人知道答案，grep 选词可能间接受污染；测试 2 的机械直测做交叉验证。

### 结果

| Q | Arm A 耗时 | Arm B 耗时 | A 步数 | B 步数 | 判定 |
|---|---|---|---|---|---|
| 1 状态流转 | 119s | 134s | 4 | 5 | 双对（B 首轮 miss，换措辞命中） |
| 2 政策计算 | 164s | 80s | 4 | 5 | 双对 |
| 3 配额占用 | 195s | 86s | 4 | 5 | 双对 |
| 4 电子签章 | 80s | 50s | 3 | 4 | 双对 |
| 5 OCR 接入 | 110s | 120s | 4 | 4 | 双对 |
| 6 token 机制 | 75s | 48s | 3 | 4 | 双对 |
| 7 操作日志 | 84s | 139s | 3 | 4 | 双对 |
| 8 批量审批 | 101s | 61s | 3 | 4 | 双对 |
| **均值** | **116s** | **89.7s** | 3.5 | 4.4 | 8/8 vs 8/8 |
| **中位** | 105.5s | 83s | | | |
| **整批** | 195s | 139s | | | |
| **上下文均值** | ~17.5K | ~7.5K（7 个采样） | | | |

### 定性发现

1. **一跳邻居足够答题**：语义检索严格命中标准文件的比例不高（见测试 2），但未命中时给的多是标准文件的直接调用方/被调方（问 LogAspect 给 OperationLogServiceImpl），agent 顺一跳读完照样答对——这是正确率没掉的机制。
2. **纯自然语言 query 有盲区**：Q1 首轮全命中前端页面，补上代码术语（status enum PENDING APPROVAL）后才命中 `OrderStateMachine`。中文注释密度部分掩盖了此问题。
3. **违规披露**：B3/B5/B8 三题（3/8）中途违规摸了 grep——被强制语义优先的 agent 仍会条件反射用 grep。原始数据未剔除。这直接支持 instructions 的写法：保留 grep 的合法用途（精确标识符跟进），而非一刀切禁止。
4. **B 臂答案普遍更深**：Q8 挖到前端 1.5s 轮询间隔与 `useBatchJob.ts`（A 臂只到后端接口）；Q6/Q2 找到 changelog/README 里的架构决策记录。
5. 8 个 agent 并发共享单个 stdio server（检索排队串行化）未拖垮 B 臂整批耗时（见测试 3 队列数据）。

## 测试 2：工具层直测（检索质量）

8 题原文直接作 query，k=5，判定标准文件是否进 top-5：

| 指标 | 值 |
|---|---|
| 严格 hit@5 | 4/8（50%） |
| MRR@5 | 0.275 |
| 未命中 4 题的性质 | 全部为一跳邻居（调用方/实现类） |
| 单题首轮噪声 | **595 字符** vs grep 轮询 **23,353 字符**（2.5 次探测均值，-97%） |

结论：语义检索的排序质量是"够用"级——不保证标准文件第一，但一跳邻居 + 定向 Read 的组合在 agent 手里等效；真正的赢面在噪声压缩。

## 测试 3：整体效率

`apps/api` 副本（751 文件 / 4463 chunks），进程内直测：

| 项 | 值 | 备注 |
|---|---|---|
| Embedder 初始化（ORT 会话） | 6.4s | 进程级一次性 |
| 冷索引 | **102.9s**（43 chunks/s） | 低于 README 的 79 chunks/s：数据集不同（含 md/yaml/jsonl，短文本占比高） |
| 热查询（n=12） | mean 0.42s / **p50 0.27s** / p95 0.30s / max 2.09s | max 为建库后首查，含懒分配 |
| 增量刷新（1 新文件 + 1 改文件） | **0.97s** | walk 检测 + 重嵌入变更 chunk，search_code 内自动触发 |
| 8 查询串行总耗时 | 1.99s | stdio server 并发即串行 → 8 agent 并发最坏排队 ≈ 2s，可忽略 |
| 进程峰值 RSS | **1911 MiB** | 含 ORT CUDA arena，与 README 显存峰值 1.8GB 一致 |
| 冷进程单查询（子进程） | 12.35s / 6.77s | 差值为磁盘缓存；6.8s ≈ ORT 初始化 + 查询 |

解读：
- **冷启 7–12s 是唯一的真实成本**，stdio server 常驻即摊销为一次；agent 首查 10.8s 属进程级一次性，之后每次 0.3s 量级。
- mtime 驱动的自动刷新让"改完就查"只需 **<1s**，这是 agent 敢把 search_code 当默认入口的前提（不担心陈旧索引）。
- 多 agent 并发的排队成本在 8 并发下 ≈ 2s，不是瓶颈；更大规模并发才需要考虑每会话 server 或批量化。

## 测试 4：本地部署小模型（Qwen3-4B）带 / 不带 codesearch

**部署**：本机源码编译 llama.cpp（b11976，CUDA，nvcc 13.3 / gcc-15 host，~8.6 min）→ `llama-server -m Qwen3-4B-Instruct-2507-Q4_K_M.gguf -c 24576 --cache-type-k/v q8_0 -ngl 99 --jinja`，OpenAI tools 接口。显存 5.4 GB（与 codesearch 的 ~1.9 GB 共存于 8 GB 卡）。

**Harness**：最小 agent 循环（自写 ~150 行 Python，非完整 agent 框架）：同 SYSTEM 提示（含仓库布局一句话 + "证据确认才停"）、温度 0.3、max_tokens 700、15 次工具调用预算、工具输出 6000 字符截断、超限后 elide 最旧工具结果。工具：`grep`（rg，每文件 6 条/总 40 行）、`read`（120 行）、`glob`；**Arm B 额外** `search_code`（进程内直调，embedder 预热后不计时）。同 8 题、同仓库。

### 结果（单次运行，逐题耗时 / prompt tokens / 评分）

| Q | A 耗时 | B 耗时 | A pt | B pt | A 评分 | B 评分 |
|---|---|---|---|---|---|---|
| 1 状态流转 | 7.2s | 6.9s | 21022 | 7294 | 严格✓ | ✓（AdminOrderServiceImpl，调用处即控制点） |
| 2 政策计算 | 6.7s | 5.2s | 20260 | 5266 | 宽✓（README 转述，未到实现） | 严格✓（PolicyEngineServiceImpl） |
| 3 配额占用 | 6.7s | 5.6s | 20823 | 4822 | ✗（文件错，未答审核中区分） | ✗（停在前端展示层） |
| 4 电子签章 | 6.3s | 10.5s | 22388 | 9169 | 宽✓（controller 机制对） | 严格✓（EsignServiceImpl+实体） |
| 5 OCR 接入 | 5.7s | 9.2s | 15110 | 14692 | ✗（跑偏到 Makefile） | ✓（OcrPipelineService，机制对） |
| 6 token 机制 | 5.8s | 8.7s | 15820 | 11260 | ✗（误判"基于 JWT"，核心错） | 严格✓（非 JWT + Redis 会话） |
| 7 操作日志 | 6.5s | 7.9s | 16442 | 7974 | ✗（脱敏结论错） | ✗（脱敏结论错——见发现 3） |
| 8 批量审批 | 6.3s | 8.9s | 13675 | 9064 | ✗（跑偏到 WatchService） | ✗（"无明确证据"，机制猜对但无文件） |
| **合计** | 6.4s | 7.9s | **16,943** | **8,693** | 宽 3/8 · 严格 1/8 | 宽 **5/8** · 严格 **4/8** |

### 发现

1. **小模型上收益放大**：GLM-5.3-Flash 两臂 8/8 打平；Qwen3-4B 宽判 3/8 → 5/8、严格 1/8 → 4/8。机制：4B 的探索策略弱（grep 选词差、过早收工、易被语义相邻的噪声文件带偏——A 臂跑偏到 Makefile/RUNBOOK/VTO/WatchService），`search_code` 首击直接把它投到正确文件群，补偿的正是它最缺的"检索策略"。
2. **Token 同样省一半**：prompt 16.9K → 8.7K（-49%）。A 臂的 prompt 大头是 grep 噪声 dump；对按 token 计费的本地推理，这直接换算成吞吐。墙钟 +23%（检索调用 + 更多轮 read），换正确率 +67%，划算。
3. **codesearch 不改变能力上限**：Q7 脱敏两臂都答错——两臂都到了 `OperationLog` 实体，但脱敏逻辑在 `LogAspect`（需要从实体跳到切面的推理），4B 没做出这一跳。检索工具给证据，不给推理。
4. 4B 的输出稳定性有限（出现 "ANSYER" 拼写错、答案截断），评分已按宽容口径处理。

### 偏差声明

n=8 单次；人工评分（对照事先钉死的 ground truth，同测试 1 口径：宽判=事实正确不论文件，严格=命中标准实现文件）；harness 为最小实现，不代表完整 agent 框架的轨迹质量；两臂提示词/预算/截断完全一致。

## 边界与偏差

- n=8、单次运行、单仓库（中文注释密集，对 grep 臂同样有利）；数据看方向，不看小数点。
- 出题人污染风险已披露（测试 1）；工具层直测（测试 2）作为无 agent 代理交叉验证。
- 两臂 scout 均可用 Read 验证命中，阅读理解能力相同；差异全部来自检索方式。
- 免费/低价模型（GLM-5.3-Flash）；更强模型上下文更贵，codesearch 的 token 节省会被放大。

## 建议

1. 探索性/语义问题 `search_code` 先行，grep 留给精确标识符与正则（已固化进 server instructions）。
2. server 常驻（omp 已按会话挂载），冷启只付一次。
3. 纯标识符类问题给 query 补代码术语，或直接 grep——语义检索对此类是弱项。
4. 收益随问题数线性放大；评估 ROI 时以 token 成本为主、墙钟为辅。

## 性能调优复测

环境：RTX 4060 Laptop 8GB，24 逻辑 CPU；Python 3.12，NumPy 2.5.4（OpenBLAS 默认 24 线程）、
ONNX Runtime 1.31.1、Transformers 5.20.2。对照为本次改动前的代码快照。
本节独立于上面的 agent 答题实验，衡量索引与查询实现本身。

### 结果与测量口径

| 场景 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| zcode 仓库扫描，4429 个匹配文件 | 5.76 s | 0.07 s | 82× |
| 合成目录扫描，4800 源码文件及大量排除目录 | 983.6 ms | 29.1 ms | 34× |
| 9000 块索引，无内容改动的强制增量构建 | 130.1 ms | 77.7 ms | 1.7× |
| 60000×768 索引，构造 Searcher | 235.3 ms | 69.2 ms | 3.4× |
| 同一大索引，MCP 热查询核心，不含推理 | 244.5 ms | 8.52 ms | 29× |
| 同一大索引，完整 CLI stats 进程 | 723.6 ms | 162.6 ms | 4.5× |
| 同一大索引，完整 CLI status 进程 | 1034.3 ms | 146.9 ms | 7.0× |
| 同一大索引，完整 CLI update 进程，已最新 | 714.5 ms | 157.6 ms | 4.5× |
| 512 个真实代码块，q4f16 CUDA 批量编码 | 70.2 chunks/s | 91.2 chunks/s | 吞吐 +30% |
| 切块，60000 个短行 | 331.8 ms | 8.11 ms | 41× |
| 切块，10000 个两行函数及空行 | 46.5 ms | 4.01 ms | 12× |

- 真实扫描对照使用同一 zcode checkout，两侧路径列表完全一致。该项为各一次运行，收益取决于排除目录大小。
- 合成扫描为 40 个模块，各含 120 个源码文件、400 个 node_modules 文件、250 个 .git 文件；各 3 次取中位数。
- 增量构建为 3000 文件、9000 块、64 维，使用确定性轻量编码器隔离文件系统及索引开销；无变化构建各 5 次。
- 60000×768 索引约 176 MiB 向量，重复已有真实向量构造；Searcher 和 MCP 核心预热一次后各 9 次取中位数。
  MCP 核心返回固定查询向量，计入源码新鲜度扫描、排名及片段读取，**不包含模型推理、HTTP 或客户端排队**。
  CLI 为独立 Python 进程各 3 次取中位数，包含启动与导入。
- 推理使用 zcode/packages 排序文件的前 512 块，batch=16、max_length=1024、768 维。
  同一模型会话交错运行原方法与最终方法，各 3 轮；表中为总块数/总时间。
  原方法耗时 8.366/6.868/6.653 s，最终方法 6.558/5.101/5.190 s，含首轮预热成本。
- 切块各 5 次取中位数；另对 202 组输入逐项比较切块文本与行号，结果一致。
  短行收益大，不能将该倍数解释为完整索引的加速倍数。

### 落地改动

1. 文件发现改为 scandir，并在进入目录前剪枝排除目录；保留排序、隐藏文件、后缀大小写及目录符号链接语义。
2. `status/stats` 和已最新的 `update` 只读元信息，不解析块清单或加载向量。
   MCP 最多缓存 4 个仓库的 Searcher；每次仍扫描源码变化，并按索引文件的 inode、大小、mtime_ns、ctime_ns 使缓存失效。
3. 向量只读 mmap；无变化构建只刷新 meta，保留向量、清单和路径列表。
   产物分别通过临时文件加 replace 发布，已打开的旧 mmap 仍可读取。
   不可解码文件导致旧块消失时必须写入新索引，不能命中无变化分支。
4. ONNX 编码按截断后的 token 长度降序分组，结果 scatter 回原始顺序；padding 额外 token 比例从 26.5% 降至 2.6%。
   长度预扫描按 batch 分词，只保留长度；单批查询跳过预扫描。输入已是 int64 时避免 astype 复制。
   最长批次先运行，避免 CUDA arena 在请求逐渐变长时因碎片无法复用而分配失败。
5. MRL 维度切换复用同一模型会话；MCP 查询沿用对应索引的维度，避免无必要的重新加载与全库重编码。
6. 切块维护缓冲区字符计数，消除每行重复求和；top-k 直接 partition 得分数组，避免复制完整负得分数组。

### 正确性、验证与限制

最终实现与原方法比较：512 个文档向量的最小余弦相似度为 **0.99999684**，最大分量差 **0.000941**；
12 个查询向量逐位一致，12/12 查询的首位结果一致，12/12 的 top-10 集合完全一致。
批次形状变化会产生 fp16 数值差异，此样本不能保证所有仓库的近似平局排名都不变。

验证命令：`python -W error::ResourceWarning -m unittest discover -s tests -v`，14 项通过。
覆盖目录剪枝与符号链接、增删改和向量行对齐、touch 无变化、UTF8→二进制、空索引、维度失效、
旧布局恢复、外部重建/替换向量后的缓存失效，以及切块边界。
另实际运行 CUDA CLI 的建库、status/stats/update/search、增量新增与删除、无变化构建；
启动独立 HTTP server，完成 MCP 初始化、查询、切换维度、自动增删刷新及三个并发查询。
167 块重建编码耗时 2.3 s，其间 `index_stats` 在 12.55 ms 内返回上一版完整元信息，未等待编码完成。

取舍：单独测量 mmap 上的热矩阵乘法曾比常驻 RAM 多约 2.6 ms；消除重复加载后 MCP 总耗时仍显著下降。
einsum 和 CPU 线程数 1/2/4/8 均比默认设置慢，因此保留矩阵乘法与 CPU 默认线程配置。
模型冷加载仍约 5–10 s，未获改善；清单首次解析仍为 O(块数)，源码新鲜度检查仍为 O(文件数)。
索引文件逐个原子替换，不提供多个写入进程之间的事务隔离。
这些是本机短样本测量，受文件系统缓存、温度、调频与其他进程负载影响。

## RRF 消融测试（2026-11-01）

问题：RRF 混合检索里 BM25 腿对结果有多大影响？逐路删除测一遍。

系统：本仓 csearch 索引（rb 仓库 1,005 文件 / 6,069 chunks，EmbeddingGemma-2 q4f16，先 `csearch update` 刷新）。
在 csearch 向量上复刻 code-context-mcp 的混合管线：FTS5 BM25 腿 + RRF（k=60，over-fetch pool=top_k×3，按 (path,start,end) 去重），
跑测试 2 的同一组 8 题与钉死的 ground truth，k=5。

脚本：`scripts/ablation_rrf.py`（`PYTHONPATH=. ~/.venvs/code-context/bin/python scripts/ablation_rrf.py`，~15s）。

| 臂 | hit@5 | MRR@5 | 说明 |
|---|---|---|---|
| vec（纯向量） | 2/8 | 0.167 | Esign@1、Ocrd@3 |
| kw-fts（pip 原装 BM25，unicode61） | 0/8 | 0 | 8/8 题返回**空** |
| kw-cjk（BM25 + CJK bigram + OR） | 2/8 | 0.167 | LogAspect@1、BatchJobVTO@3 |
| full-fts（RRF vec+fts） | 2/8 | 0.167 | 与 vec **逐位相同**（BM25 腿贡献 0 个块） |
| full-cjk（RRF vec+cjk） | 3/8 | 0.113 | Q4 从 rank1 被挤出 top5；换得 Q7@5、Q8@2 |

发现：

1. **pip 原装 BM25 腿对中文 query 完全失效**：unicode61 把整句连续中文并成单 token，FTS5 隐式 AND 匹配不到任何代码块，
   8/8 题空结果 → RRF 退化为纯向量。所谓"BM25 影响"在该配置下恒等于零——是腿死了，不是 BM25 无害。
   英文/标识符 query 下该腿正常（探针 "session token redis resolve revoke" 命中 SessionTokenService.java）。
2. **修好中文分词后 BM25 单腿与向量打平但命中不同的题**（互补性真实存在）；但 RRF 融合无净收益：
   hit 2→3，MRR 反降（0.167→0.113）。词面刷分把向量的 rank-1（EsignServiceImpl）挤出 top5，
   换来两个 rank 2/5 的新命中。8 题样本下结论：**加 BM25 腿不改善该类纯中文 PM 问句的检索质量**。
3. 与测试 2（hit@5 4/8）的对照：本次 vec 臂 2/8。已核验系统一致性——Q7 top-5 与 10-29 会话记录为同一批块、
   Q3 top1 同为 AUDIT.md。差异来自 41 个文件改动后的重建与当日口径（文件级/宽松匹配），不影响消融内部对照的有效性。

原始 per-query top-5：`/tmp/ablation_result.json`（--json-out 可重生成）。

## 纯 grep vs cs vs 混合 三臂对比（2026-11-01）

接上节消融：同一语料（rb，1,005 文件 / 6,069 chunks）、同一组 8 题与 ground truth，比较三条检索路径的检索质量与延迟。
脚本：`scripts/bench_grep_vs_cs.py`（`PYTHONPATH=. ~/.venvs/code-context/bin/python scripts/bench_grep_vs_cs.py`，~26s）。

臂定义：

- **grep**：题面机械抽词（中文 2-gram 滑窗 + 英文标识符及其 CamelCase 子词），逐词 `rg -F -i -c` 统计文件级命中，IDF 加权求和取 top-5 文件；`!.git` 排除。只用题面，不接触 ground truth。延迟含每次 rg 进程 spawn（每题 20–30 个词项）。
- **cs**：csearch 纯向量，热模型逐题 embed + 矩阵乘，chunk top-15 去重成文件。
- **hybrid**：RRF(vec pool, BM25(CJK bigram))，即消融中唯一有效的 BM25 配置；延迟只计 BM25+RRF 增量（向量部分已由 cs 臂计时）。

| 臂 | file-hit@5 | file-MRR@5 | ms/query mean | p50 |
|---|---|---|---|---|
| grep | 2/8 | 0.083 | 288.3 | 299.0 |
| cs | 2/8 | 0.167 | 83.6（p50 29.4，均值含首题 GPU 预热离群） | 29.4 |
| hybrid | 3/8 | 0.119 | 15.0（BM25+RRF 增量，整链 ≈ cs + 15ms） | 14.4 |

发现：

1. **质量三者打平（2–3/8），但命中集合互补**：grep 命中 Q5@3、Q7@3；cs 命中 Q4@1、Q5@3；hybrid 命中 Q5@5、Q7@4、Q8@2。Q1/Q2/Q3/Q6 三臂全灭——语义检索、词面检索、融合都解决不了这类需要跨层推理的题。
2. **hybrid ≠ grep ∪ cs**：RRF 融合没有把两臂的命中并起来——cs 的 rank-1（Q4 EsignServiceImpl）被 BM25 挤出 top5，grep 的 Q7@3 融合后退到 @4。与消融结论一致：BM25 腿在中文问句上是稀释而非增强。
3. **延迟**：热查询 cs ≈ 17–30ms、hybrid 整链 ≈ 45ms、grep ≈ 290ms/题（20–30 次 rg spawn 聚合）；CLI 端到端单次 711ms（含 python spawn + daemon HTTP）。冷启动：模型加载 14.7s、FTS5 构建 4.5s；grep 无冷启动成本。
4. **语料差异披露**：rg 可见 1,138 文件 vs 索引 1,005（CODE_EXTS 白名单不含 .json/.xml/.css/Makefile）。grep 臂因此吃到数据/契约文件的高词频噪声（Q4/Q7 的 top-2 被 replay_seed.sql、api.json、INDEX.md 占据）；这也可视为 csearch 白名单的卫生优势。grep 臂的设计（bigram+IDF 全量聚合）比真人 agent 的 grep 用法更系统化——真人还会付多轮推理 token，真实差距比 290ms vs 30ms 更大。

## 外部基准复跑：Agent Retrieval Bench trace2code（2026-11-06）

动机：此前所有检索质量数字都来自自出题（8 个 PM 问句），存在出题人污染。本次改用第三方公开基准
[Agent Retrieval Bench](https://arxiv.org/abs/2607.24882)（ARB，arXiv 2607.24882）的 `trace2code` v2 release：
101 个真实 workflow 样本（复现失败命令 + 失败摘录 → 根因源文件），98 个冻结 base-commit 快照，356,074 chunks / 24,883 文件，
语料与 query 由 ARB 发布，ground truth 与我方完全无关。

**协议复刻**：query = `json.dumps(sample["query"], sort_keys=True)`（ARB `query_text_for_eval`）；gold = `root_cause_files`；
文件排序 = chunk 排序去重（首现即 best）；指标 = Recall@5/10/20（gold 集覆盖率）+ MRR（首个 gold 文件倒数排名）。
lexical 与 bm25 两臂逐字使用 ARB `baseline.py` 的实现（vendored：`scripts/vendor_arb_baseline.py`）。
评分脚本：`scripts/bench_arb_trace2code.py`；语料：`bench_arb_t2c/`（release tar.zst 来自
huggingface.co/datasets/eyuansu71/agent_retrieval_bench，releases/v2_trace2code，39MB）。

**协议有效性验证**：本地复跑的 lexical 行 R@5/10/20/MRR = 0.3432 / 0.4818 / 0.6964 / 0.2075，与 ARB 论文 trace2code 表（n=101）
**逐位一致**——复跑器与发布协议严格对齐，其余各臂数字在同一标尺上可比。

结果（同 101 样本；后三行直接引用 ARB 论文 Table 6 同任务数字）：

| 方法 | R@5 | R@10 | R@20 | MRR | 来源 |
|---|---|---|---|---|---|
| lexical（路径/符号加成 TF-IDF） | 0.3432 | 0.4818 | 0.6964 | **0.2075** | 本地复跑 = 论文逐位一致 |
| BM25（k1=1.5 b=0.75） | 0.2228 | 0.3218 | 0.4934 | 0.1638 | 本地复跑 |
| **vec：EmbeddingGemma-2（csearch 管线）** | 0.1304 | 0.3135 | 0.5512 | 0.1034 | 本地复跑 |
| RRF(vec + BM25, k=60) | 0.1535 | 0.2987 | 0.5347 | 0.1389 | 本地复跑 |
| aider 式 RepoMap（无向量，路径/符号/调用图） | 0.4736 | 0.7145 | 0.8465 | 0.2745 | ARB 论文 |
| Qwen3-Embedding-8B（代码特训） | 0.2442 | 0.5281 | 0.7970 | 0.1654 | ARB 论文 |
| Qwen3-Embedding-4B | 0.0743 | 0.1815 | 0.5050 | 0.0827 | ARB 论文 |

pallets/click 子集（26 样本）：lexical 0.1859/0.2244/0.5064/0.1096，bm25 0.1923/0.3013/0.4551/0.1302，
vec 0.0385/0.0962/0.5128/**0.0683**，rrf 0.0769/0.2436/0.4936/0.0836（R@5/R@10/R@20/MRR）。
vec 在深排位（R@20 0.5128 ≈ lexical 0.5064）但在浅排位大幅落后。

发现：

1. **trace 型查询上词面/结构法完胜向量**：失败摘录里充满可 grep 的字面 span（测试名、断言文本、路径），lexical 靠
   路径/符号加成拿到 MRR 0.2075，最好代码 embedding（Qwen3-8B）也只有 0.1654。EmbeddingGemma-2（0.6B 通用模型，
   非代码特训）MRR 0.1034，介于 Qwen3-4B（0.0827）与 Qwen3-8B（0.1654）之间——量级符合模型规模与领域匹配度预期。
2. **RRF 融合第二次被证伪**：MRR 0.1389 < BM25 单腿 0.1638 < lexical 0.2075。与 11-01 在 rb 语料上的消融同构：
   融合既保不住强腿的 rank-1，也救不回弱腿的深排位。两个独立语料同向 → RBM25 腿默认该关，按 query 类型路由而非融合。
3. **外部证据并不否定语义检索，而是划定了边界**：ARB 的 `code2test`（PR 意图 → 相关测试，语义直接型）上 embedding
   碾压 lexical：Qwen3-4B MRR 0.3225 vs lexical 0.0663（R@20 0.7230 vs 0.2469）。即：NL/意图型 query → 向量赢；
   trace/标识符/日期等字面型 query → grep/lexical 赢。这与 PwC 论文（arXiv 2605.15184）inline grep 全胜、以及
   Claude Code（Boris Cherny 公开说明：早期用 RAG+本地向量库，后换 agentic grep）的工程选择互洽。
4. **所有方法的绝对值都低**（最好 MRR 0.2745），且 ARB 统计：真实 agent 轨迹在 27–35% 样本上从未读过任何 gold 文件
   ——检索是普遍失败面，单一检索器救不了全部；多工具 + agent 决策规则（何时用哪个）比换检索器更重要。
5. 全量嵌入耗时 1444.9s（356K chunks，246 chunks/s，q4f16 CUDA），lexical+BM25 评分全程 <10 分钟；
   向量臂单样本打分 2–142ms（热矩阵乘）。

偏差：n=101 单次运行；EmbeddingGemma-2 为通用嵌入模型 + csearch 非对称 prompt（ARB embedding 行无前缀且为代码特训模型，
本表对比的是"本机现有系统"而非同模型）；样本分布 gin 56/101、click 26/101；`--json-out` 在本次运行末尾因打印 bug 未落盘
（已修复 agg()，指标以上表为准）。

### Query 前缀消融（2026-11-06，补充）

csearch 查询侧带任务前缀（`QUERY_PROMPT` = "task: code retrieval | query: "），文档侧为
`title: <path> | text: <chunk>`（模型卡规定的非对称格式）。固定同一批文档向量，仅改查询编码
（`scripts/bench_query_prefix.py`，pallets/click 26 样本 / 71,096 chunks）：

| 查询编码 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|
| 带前缀（线上行为） | 0.0385 | 0.0962 | **0.5128** | **0.0683** |
| 裸文本（ARB 协议） | 0.0385 | 0.0385 | 0.3846 | 0.0542 |

26 样本中 24 个 MRR 方向一致偏向前缀：R@20 +12.8pp、MRR 相对 +26%。EmbeddingGemma 按 task-prompt
训练，前缀把查询分布对齐到训练时的检索任务上；成本≈零。前缀不改变上文结论——click 子集带前缀的
vec（MRR 0.0683）仍低于 lexical（0.1096）。

### 词面路由 + 双通道融合上线（2026-11-06，cutover）

消融证据驱动，全部落在 `csearch/mcp_server.py`（vector 臂文档/查询编码不动——bg_1 消融确认 title|text
文档编码完胜裸文本 MRR 0.0683 vs 0.0292，融合两者反而稀释 0.0425）：

1. **词面通道**：`LexIndex` 倒排 TF-IDF（identifier/CJK 单字 token，lazy build ~1-2s/repo，LRU 缓存 4 个）。
   选择 TF-IDF 而非 FTS5 BM25：11-01 消融显示 BM25 腿是稀释源，TF-IDF 无文档长度归一化偏差，且倒排查询
   只触 matched postings。
2. **路由器**：`_looks_literal()`（点分扩展名 / CamelCase / 数字串 / 路径片段）——字面型查询才融合
   RRF(vec, lex, k=60)；纯自然语言查询保持纯向量。理由：ARB trace2code 上无条件 RRF 把 MRR 从
   0.1638(BM25) 拖到 0.1389；只有词面信号强的查询才让两腿互补。
3. **`Searcher.rank(scores)`** 新增：接受预合成得分（MCP 融合路径）；CLI `search` 保持
   `searcher.search()` 纯向量不变。

验证：12 个单测全过（test doubles 补 `progress=None` 形参）；daemon 实测——字面型热查询 61ms（含融合），
NL 查询 6-40ms 纯向量；router 判定 6/6 用例正确（"OrderStateMachine 在哪"→字面、"审核状态流转"→NL）。
wire payload：instructions 510 + 三工具描述 353 = **863 chars**（≈-64% vs 修复前 2423）。

偏差披露：路由收益没有外部 ground truth 数字——rb 8 题上 route 臂与 vec 打平（2/8, MRR 0.167，因为 8 题全是
NL 型，路由臂全部落到向量通道），收益预期在字面型查询（trace/标识符），即 ARB trace2code 显示词面法大幅领先
的那类。
