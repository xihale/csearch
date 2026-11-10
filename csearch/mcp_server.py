"""MCP server exposing csearch to coding agents.

Two transports, one code path:

  stdio           one client per process (default)
  streamable-http many clients share ONE process and one Embedder

Sharing matters on a single GPU: every process loads its own ONNX arena
(~2-4 GB), so N stdio clients = N copies and the GPU runs out. Run one
``--transport http`` daemon (systemd user unit) and point every client at
``http://127.0.0.1:8765/mcp``.

Tools:
  search_code(root, query, k=5, refresh=False)
      Semantic code search; auto-indexes when no index exists.
  index_codebase(root, dim=768)
      Build or incrementally update the index for a repository.
  index_stats(root)
      Index metadata (chunks, backend, freshness).

Run:  <venv>/python -m csearch.mcp_server [--transport http --host H --port P]
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import anyio
import numpy as np
from mcp.server.fastmcp import FastMCP

from .embedder import DEFAULT_MODEL_DIR, SUPPORTED_DIMS, Embedder
from .indexer import (
    MANIFEST_NAME, META_NAME, VECTORS_NAME, Searcher, build_index,
    first_line, index_dir_for, index_is_stale, load_meta,
)

_first_line = first_line

# Wire cost note: `instructions` and every tool description land in the client's
# system prompt, so the decision rule lives here exactly once and the tools stay
# functional. Do not restate the rule in a tool docstring.
mcp = FastMCP(
    "codesearch",
    instructions=(
        "Hybrid code search over local repos, fully offline. DECISION: exploratory "
        "or natural-language questions ('where is X handled', 'how does Y work', "
        "unfamiliar repo) -> search_code before grep; reserve grep for exact "
        "identifiers, literals and regex. search_code routes internally: literal-ish "
        "queries (identifiers/paths/numbers) go to the lexical channel, natural "
        "language to the semantic channel, and both are rank-fused otherwise. It "
        "indexes and stale-refreshes on its own, so pass the repo path and query only."
    ),
    host=os.environ.get("CSEARCH_MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("CSEARCH_MCP_PORT", "8765")),
)

# One GPU, one encode at a time. FastMCP runs sync tools on the event loop, so
# every tool is async and hands its blocking body to a worker thread; this lock
# keeps concurrent clients from overlapping ONNX encodes (2x peak = GPU OOM) and
# keeps index writes serialized. Searches are ~50 ms, so contention is invisible.
_heavy = threading.Lock()
_embedder: Embedder | None = None
_embedder_lock = threading.Lock()
_searchers: OrderedDict[Path, tuple[tuple, Searcher]] = OrderedDict()


def _make_embedder(dim: int) -> Embedder:
    return Embedder(
        DEFAULT_MODEL_DIR,
        backend="ort",
        dim=dim,
        cpu=os.environ.get("CSEARCH_ORT_CPU") == "1",
    )


def _get_embedder(dim: int = 768) -> Embedder:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = _make_embedder(dim)
        return _embedder


def _resolve(root: str) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"not a directory: {root}")
    return path


def _format_hits(hits: list[dict], root: Path) -> str:
    if not hits:
        return "no results"
    lines = []
    for h in hits:
        first = first_line(root, h["path"], h["start"])
        lines.append(f"{h['score']:.3f}  {h['path']}:{h['start'] + 1}-{h['end']}  {first}")
    return "\n".join(lines)


# ---- blocking bodies (worker thread, _heavy held) -------------------------


def _get_searcher(repo: Path) -> Searcher:
    index_dir = index_dir_for(repo)
    signature = tuple(
        (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        for s in (
            (index_dir / name).stat()
            for name in (META_NAME, MANIFEST_NAME, VECTORS_NAME)
        )
    )
    cached = _searchers.get(repo)
    if cached is None or cached[0] != signature:
        _searchers[repo] = signature, Searcher(repo)
    _searchers.move_to_end(repo)
    if len(_searchers) > 4:
        _searchers.popitem(last=False)
    return _searchers[repo][1]


def _hits_sync(repo: Path, query: str, k: int, refresh: bool) -> tuple[str, list[dict]]:
    """Auto-index/stale-refresh, then ranked hits. Caller holds _heavy."""
    meta_path = index_dir_for(repo) / META_NAME
    dim = load_meta(repo)["dim"] if meta_path.exists() else 768
    embedder = _get_embedder(dim)
    prefix = ""
    if not meta_path.exists():
        note = build_index(repo, embedder, dim=dim)
        prefix = f"[auto-indexed: {note['chunks']} chunks, {note['encode_seconds']}s]\n"
    elif refresh or index_is_stale(repo):
        note = build_index(repo, embedder, dim=dim)
        prefix = (
            f"[index refreshed: {note['changed_files']} files changed, "
            f"{note['encode_seconds']}s]\n"
        )
    searcher = _get_searcher(repo)
    hits = searcher.search(embedder, query, k=k, dim=dim)
    return prefix, hits


def _search_sync(repo: Path, query: str, k: int, refresh: bool) -> str:
    prefix, hits = _hits_sync(repo, query, k, refresh)
    return prefix + _format_hits(hits, repo)


def _index_sync(repo: Path, dim: int, multimodal: bool = False) -> str:
    s = build_index(repo, _get_embedder(dim), dim=dim, multimodal=multimodal)
    return (f"indexed {s['files']} files -> {s['chunks']} chunks "
            f"(reused {s['reused_chunks']}, encoded {s['reembedded_chunks']}) "
            f"in {s['encode_seconds']}s")


def _stats_sync(repo: Path) -> str:
    m = load_meta(repo)
    return (f"chunks={m['chunks']} files={m['files']} dim={m['dim']} "
            f"backend={m['backend']} updated={m['updated']}")


def _offload(fn, *args):
    def call():
        with _heavy:
            return fn(*args)

    return anyio.to_thread.run_sync(call)


# ---- tools ----------------------------------------------------------------
# Descriptions are passed explicitly instead of relying on docstrings: the SDK
# dedents docstrings differently across versions (mcp 1.27 vs 1.30), so inline
# text is the only way to keep the wire payload byte-stable and minimal. Param
# meaning ships in the JSON schema; do not restate the decision rule here.


@mcp.tool(
    structured_output=False,
    description=(
        "Ranked file:line hits for one natural-language or code query; auto-indexes. "
        "root=repo path, query=what to find, k=hits (default 5), refresh=force re-index."
    ),
)
async def search_code(root: str, query: str, k: int = 5, refresh: bool = False) -> str:
    """Semantic search over one repo; builds/refreshes the index as needed."""
    repo = _resolve(root)
    return await _offload(_search_sync, repo, query, max(1, min(k, 20)), refresh)


@mcp.tool(
    structured_output=False,
    description=(
        "Force a full index pass — rarely needed; search_code self-indexes. "
        "root=repo path, dim=embedding width (default 768), multimodal=images+PDFs (default false)."
    ),
)
async def index_codebase(root: str, dim: int = 768, multimodal: bool = False) -> str:
    """Rebuild the index (optionally at another embedding dimension)."""
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"dim must be one of {SUPPORTED_DIMS}")
    repo = _resolve(root)
    return await _offload(_index_sync, repo, dim, multimodal)


@mcp.tool(
    structured_output=False,
    description="Index metadata: chunks, files, dim, backend, last update.",
)
async def index_stats(root: str) -> str:
    """Read <root>/.csearch/meta.json without touching the embedder."""
    repo = _resolve(root)
    return await anyio.to_thread.run_sync(_stats_sync, repo)


def _check_http_origin(request):
    from starlette.responses import JSONResponse

    raw_host = request.headers.get("host") or ""
    if raw_host.startswith("[") and "]" in raw_host:
        host = raw_host[: raw_host.index("]") + 1].lower()
    else:
        host = raw_host.partition(":")[0].lower()
    if host not in ("", "127.0.0.1", "localhost", "::1", "[::1]"):
        return JSONResponse({"error": "forbidden host"}, status_code=403)
    origin = request.headers.get("origin")
    if origin:
        return JSONResponse({"error": "cross-origin requests forbidden"}, status_code=403)
    return None


@mcp.custom_route("/api/search", methods=["POST"])
async def api_search(request):
    """Raw JSON search for the human CLI (same pipeline as search_code)."""
    from starlette.responses import JSONResponse

    denied = _check_http_origin(request)
    if denied is not None:
        return denied

    body = await request.json()
    try:
        repo = _resolve(body.get("root", ""))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    k = max(1, min(int(body.get("k", 5)), 200))

    def call() -> tuple[str, list[dict]]:
        with _heavy:
            return _hits_sync(repo, str(body.get("query", "")), k,
                              bool(body.get("refresh", False)))

    try:
        note, hits = await anyio.to_thread.run_sync(call)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return JSONResponse({"note": note.strip(), "hits": hits})


@mcp.custom_route("/api/index", methods=["POST"])
async def api_index(request):
    """Incremental index build with ndjson progress for the human CLI.

    Emits {"progress": {done, total}} per encode batch, then {"result": stats}
    (or {"error": msg}); the stream ends after exactly one terminal event.
    """
    from starlette.responses import JSONResponse, StreamingResponse

    denied = _check_http_origin(request)
    if denied is not None:
        return denied

    body = await request.json()
    try:
        repo = _resolve(body.get("root", ""))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    dim = int(body.get("dim", 768))
    if dim not in SUPPORTED_DIMS:
        return JSONResponse({"error": f"dim must be one of {SUPPORTED_DIMS}"},
                            status_code=400)
    multimodal = body.get("multimodal")
    if multimodal is not None:
        multimodal = bool(multimodal)

    events: queue.Queue = queue.Queue()

    def worker() -> None:
        def cb(done: int, total: int) -> None:
            events.put({"progress": {"done": done, "total": total}})

        try:
            with _heavy:
                s = build_index(repo, _get_embedder(dim), progress=cb, dim=dim, multimodal=multimodal)
            events.put({"result": s})
        except Exception as e:  # stream the failure; CLI decides the exit code
            events.put({"error": str(e)})
        events.put(None)

    threading.Thread(target=worker, daemon=True, name="csearch-index").start()

    async def gen():
        while True:
            item = await anyio.to_thread.run_sync(events.get)
            if item is None:
                break
            yield (json.dumps(item) + "\n").encode()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    from starlette.responses import JSONResponse

    denied = _check_http_origin(request)
    if denied is not None:
        return denied

    with _embedder_lock:
        loaded = _embedder.info if _embedder is not None else None
    return JSONResponse({"status": "ok", "embedder": loaded, "busy": _heavy.locked()})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="csearch.mcp_server")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default=mcp.settings.host)
    parser.add_argument("--port", type=int, default=mcp.settings.port)
    parser.add_argument(
        "--warm",
        action="store_true",
        help="load the embedding model before serving (a daemon that cannot get "
             "GPU memory fails at startup instead of during the first query)",
    )
    args = parser.parse_args(argv)

    # diagnostics must go to stderr: stdout is the MCP protocol channel on stdio
    print(f"codesearch mcp {args.transport} starting", file=sys.stderr)
    if args.warm:
        print(f"loading embedder: {_get_embedder().info}", file=sys.stderr)
    if args.transport == "http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"listening on http://{args.host}:{args.port}{mcp.settings.streamable_http_path}",
              file=sys.stderr)
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
