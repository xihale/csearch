# csearch — EmbeddingGemma 2 代码检索（Experimental）

> **⚠️ 实验性项目（Experimental）**：本项目是一个围绕 **Google EmbeddingGemma 2**（`google/embeddinggemma-2` 及其官方量化导出 `onnx-community/embeddinggemma-2-ONNX`）构建的实验性代码检索与 Agent MCP 验证工程，核心目的是率先支持并压测 EmbeddingGemma 2 在本地低显存环境下的量化推理、MRL 降维、非对称任务前缀与词面/向量双通道路由表现。接口与索引格式可能随实验迭代调整。

本地代码与多模态混合搜索：自然语言/代码片段查询 → 定位到代码 `文件:行号`、图片架构图或 PDF 规格书页码。支持可选的实验性多模态索引 **代码 + 图片（`.png/.jpg/.webp` 架构图、UI 截图）+ PDF（`.pdf` RFCs、specs、论文，页级 `file.pdf:pN`）** 到同一个共享 768 维 EmbeddingGemma 2 向量空间 + 词面 TF-IDF 索引中（**默认关闭**，按需通过 `--multimodal` 或 `CSEARCH_MULTIMODAL=1` 显式开启），并在检索阶段保持 **0 MB 额外常驻显存**（93MB `vision_encoder_q4f16.onnx` 仅在索引新图片时按需懒加载并立即释放）。全程离线，量化推理。

实测数据（agent 对照 / 检索质量 / 外部基准 ARB trace2code / 消融实验）见 [BENCHMARK.md](BENCHMARK.md)。

## 与其他产品的区别

市面上面向 Coding Agent 的代码检索方案大体分为三类，csearch 的定位与取舍如下：

| 对比维度 | 纯词面搜索（`ripgrep` / `zoekt` / Google `csearch`） | 云端索引 MCP（Cursor / `claude-context` + Milvus/Zilliz） | 传统本地 RAG MCP（`code-context-mcp` / Chroma / LanceDB / SQLite-vec） | **csearch（本项目）** |
|---|---|---|---|---|
| **核心模型支持** | 无语义模型 | 闭源云端 Embedding（OpenAI / Voyage） | 早期小模型（MiniLM / BGE / Jina v2，多基于 Torch/Optimum） | **原生支持 Google EmbeddingGemma 2**（适配多模态 ONNX 图、8K 窗口、MRL 截断与非对称任务前缀） |
| **多模态与文档支持** | 仅纯文本/代码 | 需云端 OCR/外部解析 | 仅文本代码文件（忽略图片与 PDF） | **代码 + 图片 + PDF 可选统一索引（实验性，默认关闭，`--multimodal` 开启）**：将代码、图片（`.png/.jpg/.webp` 架构图、UI 截图）与 PDF（`.pdf` RFCs、specs、论文，页级 `file.pdf:pN`）索引至同一个共享 768 维向量空间与词面倒排；**检索期 0 MB 额外常驻显存**（93MB `vision_encoder_q4f16.onnx` 仅在索引新图片时懒加载并立即释放） |
| **混合检索策略** | 仅精确字面/正则，搜不动“状态流转在哪控制”等概念问句 | 黑盒融合 | **无条件 RRF(向量 + FTS5 BM25)**：实测 FTS5 `unicode61` 对中文整句完全失效（0 命中），且在纯自然语言问句上 BM25 会把向量 rank-1 挤出前五（见消融实验） | **按查询类型启发式路由**：纯自然语言走纯向量（防稀释）；含标识符/路径/驼峰/数字时才融合倒排 TF-IDF + RRF |
| **多 Agent 显存开销** | 0 | 0（云端付费） | stdio 模式下 **N 个会话 = N 份模型进程**，8GB 显卡开 3 个会话即 OOM | **单常驻 HTTP Daemon + CLI/MCP 共用**，关闭 ORT 2倍扩池策略，常驻显存 ~990 MB 供所有会话共享 |
| **存储与外部依赖** | 无或单文件 trigram | 需 Docker / 外部向量数据库 / API Key | 需 SQLite 扩展、LanceDB、Chroma 或重型 PyTorch 环境（>2GB） | **零向量库依赖**：`.csearch/` 下仅 `vectors.npy`（只读 `mmap`）+ `manifest.jsonl`，`numpy` 矩阵点积热查 8–30ms |
| **索引新鲜度** | 实时扫盘（或需手动重建） | 后台轮询上传云端，有延迟与隐私顾虑 | 常需手动触发 re-index，或每次全量读盘 | **查询时 `scandir` 毫秒级 mtime 校验 + SHA-256 增量复用**，改完代码直接搜自动 <1s 刷新，原子替换不阻塞并发读 |
| **MCP 提示词税** | 无 | 视实现而定 | 动辄 2,400+ 字符 schema/docstring 塞满每个 Agent 会话上下文 | **极致压缩（~1,791 字符）**：决策规则只写一次，明确告诉 Agent“探索问题先 `search_code`、精确标识符用 `grep`” |

一句话总结：**csearch 不试图替代 `rg`**（精确符号搜索永远是 `rg` 最快），而是作为 **EmbeddingGemma 2 的极简本地实验载体**，用最小的依赖（原生 ONNX Runtime + NumPy `mmap`）和经过消融验证的路由策略，在 8GB 消费级显卡上为多个并发 Agent 省下 ~50% 的无关代码阅读 Token。

## 模型与量化

基座 `google/embeddinggemma-2`（740M 多模态，文本骨干 271M，768 维，8K 上下文）。
推理用官方导出的 **onnx-community/embeddinggemma-2-ONNX** 量化档：

| 档位 | 权重 | 大小 | 用途 |
|---|---|---|---|
| `q4f16` | int4 + fp16 激活 | 157 MB | GPU 默认（速度优先）|
| `int8` | int8 动态 | 314 MB | 质量优先 / 无 GPU 时回退 |

对比 torch bf16（文本加载 271M）。实测（RTX 4060 Laptop 8GB，2157 个真实代码 chunk）：

| 方案 | 吞吐 | 权重体积 | 检索质量（vs bf16 参考）|
|---|---|---|---|
| torch bf16 CUDA | 56 chunks/s | 540 MB | 基准 |
| **ONNX q4f16 CUDA（默认）** | **79 chunks/s** | 157 MB | spearman 0.973，top1 一致 67%（差异均为近似平局）|
| ONNX int8 CUDA | 38 chunks/s | 314 MB | spearman 0.999，top1 一致 100% |
| ONNX int8 CPU | 3 chunks/s | 314 MB | 同上，仅限无法用 GPU 时 |

结论：q4f16 速度最快；在意排序稳定性用 `--variant int8`（同走 CUDA EP，质量几乎无损）。

> 附注：单次查询 ~12s 总耗时里约 10s 是 ORT 会话初始化；索引/批量检索场景可忽略。

## 安装

```bash
pip install -e .                            # 或 pip install -r requirements.txt
# 可选安装实验性多模态支持（图片与 PDF 索引）：
# pip install -e ".[multimodal]"
# 可选安装 torch 参考后端（bf16）：
# pip install -e ".[torch]"
python scripts/prepare_model.py             # 物化文本量化权重到 models/（约 450 MB，之后离线可用）
# 若需多模态图片索引，追加 --vision 下载 vision_encoder 权重（+93 MB q4f16）：
# python scripts/prepare_model.py --vision
```

注意：必须用 `scripts/prepare_model.py` 下载为**真实文件**——HF 缓存的符号链接布局会让
ONNX external-data 路径校验失败。脚本同时绕过企业代理的 cert 问题（走系统 CA、强制官方 endpoint）。

`csearch` 命令入口：`~/.local/bin/csearch`（shim：`PYTHONPATH` 指向本仓库 + venv python），
遮蔽 AUR `codesearch-git` 的 `/usr/bin/csearch`（Google trigram grep，需要时用全路径调用）。

## 使用

```bash
csearch "查询词"                                  # = search . "查询词"（首参不是子命令时默认按 search 处理）
csearch ~/some/repo "查询词"                      # 首参是存在的目录 → 作为 repo，其余是查询
cs i|u|s|st [repo] …                             # 子命令短别名：index/update/search/status（cs 是 csearch 的符号链接）
python -m csearch index  [repo]                  # 建索引 / 强制增量重建（默认 repo=.，文本/代码）
python -m csearch index  [repo] --multimodal     # 实验性：同时索引图片与 PDF（默认关闭）
python -m csearch update [repo]                  # 增量刷新；索引已最新时秒回（不加载模型）
python -m csearch search [repo] "查询词" -k 5    # 语义搜索，输出 score + 文件:行号 + 首行
python -m csearch search . "查询词" --show       # 附带打印命中 chunk 的源码（带行号）
python -m csearch status [repo]                  # 索引新鲜度（不加载模型，秒回）
python -m csearch stats  [repo]                  # 单行元信息
```

`index`/`update`/`search` 默认走共享 daemon（`csearch-mcp.service`，见下文 MCP 一节）：
CLI 只发 HTTP 请求，本进程不加载模型——手动命令从 ~6s 降到 <1s，显存只有 daemon 一份。
daemon 不在跑时自动 `systemctl --user start` 拉起（无 unit 则 spawn 裸进程），仍失败就
退回本进程编码并提示。`--local` 强制本进程编码；`CSEARCH_DAEMON_URL` 换 daemon 地址
（默认 `http://127.0.0.1:8765`）。`status`/`stats` 与已最新的 `update` 永远只读本地文件，
不经过 daemon。

索引落在 `<repo>/.csearch/`（vectors.npy + manifest.jsonl + files.json + meta.json；files.json 是
上次扫描的路径集，供 staleness 判断——manifest 不含解码失败/零 chunk 的文件）。
`index`/`update` 会自动把 `.csearch/` 写进 `<repo>/.gitignore`（幂等，已有则跳过）。

模型目录默认取**本包旁的** `models/embeddinggemma-2-ONNX`（`DEFAULT_MODEL_DIR`），
所以在任意仓库里直接 `csearch index` 都可用，不需要先 `cd` 到本仓库；要换权重用 `--model <dir>`。

常用参数：`--multimodal`（实验性：同时索引图片 `.png/.jpg/.webp` 与 PDF `.pdf`，默认关闭；或设置环境变量 `CSEARCH_MULTIMODAL=1`）、`--backend torch`（bf16 参考路径）、`--dim 256`（MRL 截断降维，重归一化）、
`--batch`、`--variant q4f16|int8`；agent 的 MCP server 常驻显存导致 OOM 时加 `--cpu`
（int8 走 CPU EP，报错信息里也会提示）。首次使用 `--multimodal` 建立索引后，元数据会记录该设置，后续 `update` 与 MCP 自动刷新将自动保留多模态索引。`update` 的过期判定走 mtime 快扫，真正重编码
按内容 hash；仅 touch 未改内容时不重新编码，也不重写 vectors/manifest/files.json。
`status`、`stats` 与已最新的 `update` 仅读取元信息，不加载模型或向量矩阵。

### Shell 补全

一条命令安装（自动装 `~/.local/bin/csearch` wrapper + 对应 shell 的补全文件）：

```bash
python -m csearch completions install          # 默认取 $SHELL/登录 shell
python -m csearch completions install bash zsh fish
python -m csearch completions uninstall        # 同样支持指定 shell
```

落点：

| shell | 补全文件 | 额外一行 |
|---|---|---|
| bash | `~/.local/share/bash-completion/completions/csearch` | `~/.bashrc` 里 source 该文件（无 bash-completion 包也能用）|
| zsh | `~/.local/share/zsh/site-functions/_csearch` | `~/.zshenv` 里加 fpath（用户级目录不在默认 fpath）|
| fish | `~/.config/fish/completions/csearch.fish` | 无需（自动加载）|

每个文件带 `managed by csearch completions install` 标记：重装幂等（`unchanged`），
卸载只删带标记的文件与 rc 行——用户自己写的同名文件不会被覆盖，install 会拒绝并提示。

不动 rc 的手动方式：`eval "$(python -m csearch completions bash)"`（zsh/fish 同理，
需先自行定义 `csearch`）。补全为静态脚本，TAB 无 Python 启动开销。

> 注意 PATH 上另有同名 `csearch`（`csearch` 包，正则搜索工具）；`~/.local/bin` 在它前面，
> 装 wrapper 后会遮蔽它。如需并存请自行改名或调整 PATH 顺序。

## 集成要点（踩坑记录）

1. **任务前缀**：query 用 `task: code retrieval | query: ...`（`CodeRetrieval` prompt）；
   文档用 `title: <相对路径> | text: <代码>`。不对称格式，检索质量差异明显。
2. **MRL 截断**：768 → 256/128 时必须**截断后重新 L2 归一化**。
3. **文本-only 加载**（torch 路径）：`config_kwargs={"vision_config": None, "audio_config": None}`
   省 470M 参数；dtype 用 bf16（fp16 会静默 NaN）。
4. **ONNX 图 IO**：`input_ids/attention_mask` + 三个必填多模态输入（文本路径喂 `(0,512)`
   零张量）；输出 `sentence_embedding` 已 mean-pool，包装器只做 L2 归一化。
5. **切块**：行对齐 ~1400 字符 + 200 尾部重叠；超长单行硬切。显存峰值因此从 OOM 降到 1.8 GB。
6. **环境坑**：本机 `HF_ENDPOINT` 指向的 hf-mirror 对部分仓库有**过期快照**（config.json
   旧版导致 transformers 报 model_type 不识别）；`optimum-onnx` 会把 transformers 5.x 降级到
   4.x（embedding_gemma2 架构不被 4.x 识别）——所以本项目绕开 optimum，直接用原生 onnxruntime。
7. **共享显存**：CUDA EP 默认 `arena_extend_strategy=kNextPowerOfTwo`，常驻 server 会为一个
   ~160MB 的模型预留 4GB+，其他会话随即 OOM，所以会话里固定成 `kSameAsRequested`（按需精确增长）。
   即便如此，8GB 卡上同时挂着多个 agent 会话时批量编码仍可能不够：`--batch 2`（实测 snirect-org
   808 chunks 从 OOM 变为 27s 完成）或 `--cpu` 兜底。

## MCP（给 agent 用）

`csearch/mcp_server.py` 暴露三个工具：`search_code`（自动建索引；文件有改动时自动增量刷新，
agent 无需关心索引新鲜度）、`index_codebase`（仅预热超大仓库或强制重建时需要）、`index_stats`。
server instructions 内置决策规则：探索性/语义问题先 `search_code` 后 grep，grep 只留给精确标识符和正则。

查询沿用现有索引维度；跨仓库切换维度只调整 MRL 输出截断，复用模型会话。
daemon 最多缓存 4 个仓库的清单与只读 mmap 向量；每次查询仍扫描源码新鲜度，并检查
meta/manifest/vectors 的 inode、大小及纳秒时间戳，外部重建也会立即使缓存失效。
扫描在进入目录前剪枝 `node_modules`、`.git` 等排除目录。ONNX 多批编码按截断后的
token 长度从长到短分组，减少 padding；返回行顺序保持不变。索引文件分别原子替换，
已打开的 mmap 可继续读取旧文件。性能对比与数值差异见 [BENCHMARK.md](BENCHMARK.md)。

### 两种传输：stdio 单人，HTTP 共享

每个进程都会加载自己的 ONNX arena（实测 1–4 GB 显存），所以 **N 个 agent 会话 = N 份显存**。
2026-10-30 实测：6 个并行 omp 会话各起一个 stdio server，其中两个已加载模型，吃掉
2.2 GB + 4.2 GB，第三个客户端直接 `Failed to allocate 57999360`。

所以默认部署是**一个常驻 daemon + N 个客户端连同一 URL**（streamable-HTTP）：

```bash
python -m csearch.mcp_server --transport http --host 127.0.0.1 --port 8765 [--warm]
curl -s http://127.0.0.1:8765/health   # {"status":"ok","embedder":"ort/model_q4f16/CUDA","busy":false}
```

人用 CLI 走同一 daemon 的原始 JSON 路由：`POST /api/search`（与 `search_code` 同管线，
返回原始 hits）与 `POST /api/index`（ndjson：每个 encode batch 一条
`{"progress":{done,total}}`，结尾 `{"result":…}`）——CLI 的进度条因此能跨进程显示。

- `--warm` 在监听前加载模型：拿不到显存就启动失败，由 systemd 报错，而不是第一个查询时才炸。
- `CSEARCH_MCP_HOST/PORT` 等价于命令行参数；`CSEARCH_ORT_CPU=1` 强制 CPU EP（GPU 被别的会话占满时兜底）。
- 并发：FastMCP 的同步工具跑在事件循环上，会冻住所有客户端；因此三个工具都是 `async` + 线程卸载，
  编码与索引写入由 `_heavy` 锁串行化，`index_stats` 仅读取原子发布的元信息，不等待编码锁。
  实测 3 个并发检索 0.40–0.46 s，daemon 显存 990 MB。

systemd 用户单元（模板见 `contrib/csearch-mcp.service`）：
可复制到 `~/.config/systemd/user/csearch-mcp.service` 并通过 `systemctl --user enable --now csearch-mcp.service` 启用开机自启与失败重启。
`ExecStart` 默认执行 `%h/.local/bin/csearch-mcp --transport http --warm`。

### 客户端配置（新会话生效）

`~/.omp/agent/mcp.json`（已对上游 `mcp-schema.json` 校验通过）：

```json
"mcpServers": {
  "codesearch": {"type": "http", "url": "http://127.0.0.1:8765/mcp", "timeout": 600000}
}
```

Codex CLI（`~/.codex/config.toml`）：

```toml
[mcp_servers.codesearch]
url = "http://127.0.0.1:8765/mcp"
startup_timeout_sec = 30
tool_timeout_sec = 600
```

daemon 没起时客户端报连接错误而不是自动拉起——这是共享实例的代价，靠 systemd 兜住。

### 接线预算

`instructions` 与每个 tool 的 description/schema 全部进入客户端 system prompt，每个会话都付费。
规则：决策规则只在 `instructions` 里写一遍，tool description 只写功能和参数（含参数含义，
因为 FastMCP 不把 docstring 的 `Args` 解析进 JSON Schema，那段是纯 prose 开销）。
`structured_output=False` 去掉 `-> str` 自动生成的 `outputSchema`；description 用显式字符串
而非 docstring，因为 SDK 的 dedent 行为在 mcp 1.27 / 1.30 间不一致。

当前 **1791 字符**（tool schemas 1281 字符，mcp.instructions 510 字符），改动前 2423 字符。新增工具或参数后重跑一次：

```bash
python3 -c "import asyncio,json;from csearch.mcp_server import mcp;\
print(len(json.dumps([t.model_dump(by_alias=True,exclude_none=True) for t in asyncio.run(mcp.list_tools())])),len(mcp.instructions))"
```

## 结构

```
csearch/
  chunker.py      行对齐切块
  embedder.py     量化 ONNX / torch bf16 双 backend 包装（含可选 vision_encoder 懒加载）
  indexer.py      增量索引 + 混合检索（numpy 点积 + 倒排 TF-IDF + RRF）
  mcp_server.py   FastMCP Server（stdio & streamable-HTTP daemon）
  cli.py          index / update / search / status / stats / completions
  completions.py  静态 bash/zsh/fish 补全脚本
scripts/
  prepare_model.py  权重物化（默认文本量化模型，--vision 追加视觉编码器）
contrib/
  csearch-mcp.service  systemd 用户级常驻服务单元模板
```

## 开源协议（License）

本项目采用 **[GNU General Public License v3.0 or later (GPL-3.0-or-later)](LICENSE)** 开源。

