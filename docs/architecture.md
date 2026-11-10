# 架构与对比

`csearch` 围绕 `google/embeddinggemma-2`（及其 ONNX 量化导出 `onnx-community/embeddinggemma-2-ONNX`）构建，探索在 8GB 消费级显卡上为多个 Coding Agent 提供低显存、零外部向量库依赖的混合代码检索。

## 与同类方案的区别

| 维度 | `ripgrep` / `zoekt` | 云端索引 MCP（Cursor / Zilliz） | 本地 RAG MCP（Chroma / LanceDB / SQLite-vec） | `csearch` |
|---|---|---|---|---|
| **模型** | 无语义模型 | 闭源云端 Embedding | 早期小模型（MiniLM / BGE / Jina） | **EmbeddingGemma 2**（ONNX 量化、8K 窗口、MRL 截断、非对称前缀） |
| **混合策略** | 仅精确字面与正则 | 黑盒融合 | 无条件 RRF(向量 + FTS5 BM25)，中文整句易失效，自然语言问句易被 BM25 稀释 | **按查询类型路由**：自然语言走纯向量；含标识符/路径/驼峰/数字时融合倒排 TF-IDF + RRF |
| **多会话显存** | 0 | 0（云端计费） | stdio 模式下 N 个会话加载 N 份模型 | **单 HTTP Daemon 共享**，常驻显存 ~990 MB |
| **存储依赖** | 无或单文件索引 | 外部向量库 / API Key | SQLite 扩展 / LanceDB / Chroma / PyTorch | `.csearch/` 下仅 `vectors.npy`（只读 `mmap`）+ `manifest.jsonl`，热查 8–30ms |
| **新鲜度** | 实时扫盘 | 后台上传，有延迟与隐私顾虑 | 常需手动触发重索引 | 查询时 `scandir` 校验 mtime + SHA-256 增量复用，改动自动 <1s 刷新 |
| **多模态（实验）** | 仅纯文本 | 需云端解析 | 仅文本代码 | 可选 `--multimodal` 索引图片（`.png/.jpg/.webp`）与 PDF（页级 `file.pdf:pN`），检索期不占额外常驻显存 |

`csearch` 不替代 `rg`（精确符号查找永远是 `rg` 最快），而是给探索性问题提供一跳定位，减少 Agent 盲搜无关代码消耗的上下文。

## 模型与量化档位

基座为 `google/embeddinggemma-2`（多模态 740M，文本骨干 271M，768 维，8K 上下文）。默认使用 `onnx-community/embeddinggemma-2-ONNX`：

| 档位 | 权重格式 | 体积 | 吞吐（RTX 4060 8GB） | 相对 bf16 排序质量 | 适用场景 |
|---|---|---|---|---|---|
| `q4f16`（默认） | int4 + fp16 激活 | 157 MB | 79 chunks/s | Spearman 0.973 | GPU 日常使用，速度优先 |
| `int8` | int8 动态量化 | 314 MB | 38 chunks/s (CUDA) / 3 chunks/s (CPU) | Spearman 0.999 | 排序稳定性优先，或 `--cpu` 回退 |
| `torch` | bf16 参考后端 | 540 MB | 56 chunks/s | 基准 | 精度对照实验 |

## 工程细节与踩坑记录

1. **非对称任务前缀**：查询使用 `task: code retrieval | query: ...`，文档使用 `title: <相对路径> | text: <代码>`。去掉前缀会导致 MRR 下降约 20%。
2. **MRL 截断降维**：支持 `--dim 256` 或 `128`，截断后重新做 L2 归一化；跨仓库切换维度直接复用同一模型会话。
3. **ONNX 多模态图输入**：主图包含 `input_ids`、`attention_mask` 与 `image_features`。纯文本编码时向 `image_features` 传入 `(0, 512)` 零张量；启用 `--multimodal` 索引新图片时，按需加载 `vision_encoder_q4f16.onnx`（93 MB）提取特征，跑完立即释放会话。
4. **行对齐切块与长度分组**：按 ~1400 字符切块并保留 200 字符行尾重叠；批量编码前按 token 长度降序分组（padding 浪费从 26.5% 降至 2.6%），长批次优先分配以避免 CUDA 显存碎片。
5. **抑制 ORT 显存膨胀**：CUDA EP 默认 `kNextPowerOfTwo` 会为 160 MB 模型预留 4GB+ 显存。`csearch` 将其固定为 `kSameAsRequested`，8GB 显卡可与本地 LLM 或多个 Agent 并存。显存紧张时可传 `--batch 2` 或 `--cpu`。
6. **绕开 HF 符号链接与镜像旧快照**：`scripts/prepare_model.py` 将权重物化为真实文件（修复 ONNX external-data 拒绝读取 symlink 的问题），并直连官方 endpoint 避免镜像站旧版 `config.json` 报错。

## 代码结构

```
csearch/
  chunker.py      行对齐切块
  embedder.py     ONNX / Torch 双后端封装（含可选视觉编码器懒加载）
  indexer.py      增量索引与混合检索（numpy mmap 点积 + 倒排 TF-IDF + RRF）
  mcp_server.py   FastMCP Server（stdio 与 streamable-HTTP daemon）
  cli.py          命令行入口
  completions.py  bash / zsh / fish 静态补全生成
scripts/
  prepare_model.py  模型权重下载与物化
```
