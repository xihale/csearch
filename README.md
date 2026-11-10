# csearch

> **Experimental**：围绕 **Google EmbeddingGemma 2**（`onnx-community/embeddinggemma-2-ONNX`）构建的本地代码检索实验项目，提供 CLI 与 MCP Server。

输入自然语言或代码片段，定位到 `<文件>:<行号>`。全程离线运行，基于 ONNX q4f16/int8 量化推理与 NumPy `mmap`，无需外部向量数据库。

## 核心特性

- **查询感知混合路由**：纯自然语言查询走向量检索（防 BM25 词面稀释）；包含标识符、路径或驼峰命名时，自动融合倒排 TF-IDF 与 RRF。
- **单守护进程共享显存**：通过 HTTP Daemon（`csearch-mcp`）供多个 Agent 会话与 CLI 共享同一模型实例，常驻显存 ~990 MB，热查询 8–30ms。
- **毫秒级增量同步**：查询时通过 `scandir` 校验文件 `mtime`，仅对内容哈希变化的块重编码，改完代码直接搜自动在 <1s 内刷新。
- **可选多模态索引（实验性，默认关闭）**：传入 `--multimodal` 可将架构图（`.png/.jpg/.webp`）与 PDF 文档（`.pdf`）编码至同一向量空间，视觉编码器按需加载、用完即释，检索期 0 额外显存。

## 快速开始

```bash
# 1. 安装
pip install -e .
python scripts/prepare_model.py            # 下载文本量化模型至 models/ (~450 MB)

# 可选：启用图片与 PDF 索引支持
# pip install -e ".[multimodal]" && python scripts/prepare_model.py --vision

# 2. 检索与建库
csearch "状态流转在哪控制"                   # 在当前仓库搜索（自动建库/增量刷新）
csearch search ~/repo "token 校验" --show   # 搜索并打印命中的代码片段
csearch index ~/repo --multimodal          # 显式开启多模态索引
```

作为 MCP Server 接入 Agent（推荐常驻 HTTP 模式）：

```bash
csearch-mcp --transport http --host 127.0.0.1 --port 8765 --warm
```

## 文档索引

- **[使用与部署指南 (docs/usage.md)](docs/usage.md)**：CLI 参数、MCP 客户端配置、systemd 常驻服务、Shell 补全。
- **[架构与同类对比 (docs/architecture.md)](docs/architecture.md)**：与 `rg` / 云端 MCP / 本地 RAG 的区别、量化档位表现、工程踩坑记录。
- **[评测与消融报告 (docs/benchmark.md)](docs/benchmark.md)**：Agent 对照实验、ARB `trace2code` 外部基准复跑、RRF 与前缀消融数据。

## License

[GPL-3.0-or-later](LICENSE)
