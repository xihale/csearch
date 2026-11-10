# 使用与部署指南

## CLI 详细用法

```bash
csearch "查询词"                                  # 默认在当前目录搜索
csearch ~/some/repo "查询词"                      # 指定仓库目录搜索
csearch search . "查询词" -k 5 --show             # 附带打印命中代码片段（带行号/页码）
csearch index  [repo]                             # 建立或增量更新索引
csearch index  [repo] --multimodal                # 实验性：同时索引图片与 PDF（默认关闭）
csearch update [repo]                             # 增量刷新；无改动时直接读取元信息秒回
csearch status [repo]                             # 检查索引是否过期（不加载模型）
csearch stats  [repo]                             # 单行输出索引统计信息
```

子命令支持短别名：`i`（`index`）、`u`（`update`）、`s`（`search`）、`st`（`status`）。

### 常用参数与环境变量

- `--multimodal` / `CSEARCH_MULTIMODAL=1`：开启图片（`.png/.jpg/.jpeg/.webp`）与 PDF（`.pdf`）索引。首次开启后会记录在 `<repo>/.csearch/meta.json` 中，后续 `update` 自动沿用。
- `--variant q4f16|int8`：切换量化档位（默认 `q4f16`）。
- `--dim 768|512|256|128`：MRL 向量维度截断。
- `--batch N`：编码批大小（显存不足时可调小，如 `--batch 2`）。
- `--cpu` / `CSEARCH_ORT_CPU=1`：强制使用 CPU 推理。
- `--local`：跳过常驻 HTTP daemon，在当前进程直接加载模型。
- `--model <dir>`：指定模型目录（默认读取包目录下的 `models/embeddinggemma-2-ONNX`）。

索引数据保存在 `<repo>/.csearch/`，建库时会自动将该目录追加到 `<repo>/.gitignore`。

## MCP Server 部署（供 Agent 使用）

`csearch-mcp` 暴露三个工具：`search_code`（查询时自动检测文件变更并增量刷新）、`index_codebase`、`index_stats`。

### 常驻 HTTP Daemon（推荐）

多个 Agent 会话若各自启动 stdio 进程，会导致显存成倍占用。推荐启动单个 HTTP daemon 供所有 Agent 和 CLI 共享：

```bash
csearch-mcp --transport http --host 127.0.0.1 --port 8765 --warm
curl -s http://127.0.0.1:8765/health
```

可使用 `contrib/csearch-mcp.service` 配置 systemd 用户服务开机自启：

```bash
cp contrib/csearch-mcp.service ~/.config/systemd/user/
systemctl --user enable --now csearch-mcp.service
```

### 客户端配置

**MCP JSON（Claude Code / Cursor / OMP 等）**：

```json
{
  "mcpServers": {
    "codesearch": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp",
      "timeout": 600000
    }
  }
}
```

**Codex CLI（`~/.codex/config.toml`）**：

```toml
[mcp_servers.codesearch]
url = "http://127.0.0.1:8765/mcp"
startup_timeout_sec = 30
tool_timeout_sec = 600
```

## Shell 自动补全

支持安装 bash、zsh、fish 的静态补全脚本（无 Python 启动延迟）：

```bash
python -m csearch completions install          # 自动检测当前登录 shell
python -m csearch completions install bash zsh fish
python -m csearch completions uninstall        # 卸载补全
```

若不想修改 shell 配置文件，也可手动求值：`eval "$(python -m csearch completions bash)"`。
