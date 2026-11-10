"""CLI: python -m csearch {index,update,search,status,stats,completions}

Human-facing entrypoint (agents use csearch/mcp_server.py). All output is
plain text, aligned, with counts grouped for readability.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import urllib.error
import urllib.request

from tqdm import tqdm

from . import indexer
from .completions import COMPLETIONS, SHELLS, default_shell, install, uninstall
from .embedder import DEFAULT_MODEL_DIR, Embedder, SUPPORTED_DIMS
from .indexer import (
    INDEX_DIRNAME, Searcher, build_index, ensure_gitignored, first_line,
    index_dir_for, index_is_stale, load_meta,
)

_ensure_gitignored = ensure_gitignored


def _use_local(ns) -> bool:
    return bool(
        getattr(ns, "local", False)
        or getattr(ns, "model", None) is not None
        or getattr(ns, "backend", "ort") != "ort"
        or getattr(ns, "variant", None) is not None
        or getattr(ns, "batch", 16) != 16
        or getattr(ns, "cpu", False)
    )


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None,
                   help="model dir prepared by scripts/prepare_model.py "
                        f"(default: {DEFAULT_MODEL_DIR})")
    p.add_argument("--backend", choices=("ort", "torch"), default="ort",
                   help="ort = quantized ONNX (default), torch = SentenceTransformer bf16")
    p.add_argument("--variant", choices=("q4f16", "int8"), default=None,
                   help="override ONNX weights (default: q4f16 on GPU, int8 otherwise)")
    p.add_argument("--dim", type=int, default=768, choices=SUPPORTED_DIMS,
                   help="MRL embedding dimension (truncated + renormalized)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--cpu", action="store_true",
                   help="force CPU inference (use when agents' servers hold the GPU)")
    p.add_argument("--multimodal", action="store_true", default=None,
                   help="experimental: also index images (.png/.jpg/.webp) and PDFs (.pdf)")


def _make_embedder(args) -> Embedder:
    return Embedder(args.model or DEFAULT_MODEL_DIR, backend=args.backend,
                    variant=args.variant, dim=args.dim, batch_size=args.batch,
                    cpu=args.cpu)


def _ago(epoch: float | None) -> str:
    if epoch is None:
        return ""
    s = max(0, int(time.time() - epoch))
    for unit, size in (("s", 60), ("m", 60), ("h", 24)):
        if s < size:
            return f" ({s}{unit} ago)"
        s //= size
    return f" ({s}d ago)"


class _EncodingBar:
    """Per-batch embedding progress on stderr. No-op unless stderr is a TTY."""

    def __init__(self) -> None:
        self.bar = (tqdm(unit="chunk", desc="embedding", leave=False,
                         dynamic_ncols=True)
                    if sys.stderr.isatty() else None)

    def __call__(self, done: int, total: int) -> None:
        if self.bar is None:
            return
        if self.bar.total != total:
            self.bar.total = total
        self.bar.update(done - self.bar.n)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


# ---- shared daemon (csearch-mcp.service: one ONNX session for all clients) --


class _DaemonDown(RuntimeError):
    pass


def _daemon_url() -> str:
    return os.environ.get("CSEARCH_DAEMON_URL", "http://127.0.0.1:8765").rstrip("/")


def _daemon_healthy(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(_daemon_url() + "/health", timeout=timeout) as r:
            return json.loads(r.read()).get("status") == "ok"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _ensure_daemon() -> bool:
    """Healthy -> True. Else start the systemd unit (or spawn detached) and wait.

    A custom CSEARCH_DAEMON_URL skips the unit entirely: that address has its
    own lifecycle (spawned below with a matching CSEARCH_MCP_PORT).
    """
    if _daemon_healthy():
        return True
    custom = os.environ.get("CSEARCH_DAEMON_URL") is not None
    if not custom:
        try:
            subprocess.run(["systemctl", "--user", "start", "csearch-mcp.service"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=15)
        except (OSError, subprocess.SubprocessError):
            pass
        for _ in range(24):  # unit start: only the model load (~6s) is the wait
            if _daemon_healthy(0.5):
                return True
            time.sleep(0.5)
    log_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "csearch"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        port = urlparse(_daemon_url()).port
        if port:
            env["CSEARCH_MCP_PORT"] = str(port)
        with (log_dir / "daemon.log").open("ab") as log:
            subprocess.Popen(
                [sys.executable, "-m", "csearch.mcp_server", "--transport", "http", "--warm"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, env=env)
    except OSError:
        return False
    for _ in range(60):  # cold spawn: ~8s on GPU; headroom for CPU embedders
        if _daemon_healthy(0.5):
            return True
        time.sleep(0.5)
    return False


def _daemon_post(path: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        _daemon_url() + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("error", e.reason)
        except (OSError, ValueError):
            detail = e.reason
        raise RuntimeError(f"daemon {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise _DaemonDown(str(e)) from None


def _api_search(root: Path, query: str, k: int) -> dict:
    with _daemon_post("/api/search",
                      {"root": str(root), "query": query, "k": k},
                      timeout=600) as resp:
        return json.loads(resp.read())


def _api_index(root: Path, dim: int, progress, multimodal: bool | None = None) -> dict:
    stats = None
    payload: dict = {"root": str(root), "dim": dim}
    if multimodal is not None:
        payload["multimodal"] = bool(multimodal)
    with _daemon_post("/api/index", payload, timeout=3600) as resp:
        for raw in resp:
            ev = json.loads(raw)
            if "progress" in ev:
                if progress is not None:
                    p = ev["progress"]
                    progress(p["done"], p["total"])
            elif "result" in ev:
                stats = ev["result"]
            elif "error" in ev:
                raise RuntimeError(f"daemon: {ev['error']}")
    if stats is None:
        raise RuntimeError("daemon closed the stream before a result")
    return stats


def _report(root: Path, s: dict) -> None:
    print(f"  files       {s['files']:,}")
    print(f"  chunks      {s['chunks']:,}")
    print(f"  encoding    re-encoded {s['reembedded_chunks']:,} chunks "
          f"from {s['changed_files']} changed file(s), "
          f"reused {s['reused_chunks']:,} in {s['encode_seconds']}s")
    print(f"  gitignore   {s.get('gitignore') or _ensure_gitignored(root)}")


def _build_and_report(root: Path, ns) -> None:
    t0 = time.time()
    print("loading model ...", file=sys.stderr, flush=True)
    embedder = _make_embedder(ns)
    progress = _EncodingBar()
    try:
        s = build_index(root, embedder, progress=progress, multimodal=getattr(ns, "multimodal", None))
    finally:
        progress.close()
    _report(root, s)
    print(f"done in {time.time() - t0:.1f}s")


def _build_via_daemon(root: Path, ns) -> bool:
    """Index through the shared daemon (auto-started). False -> build locally."""
    if _use_local(ns):
        return False
    if not _ensure_daemon():
        print("daemon unreachable; building in-process", file=sys.stderr)
        return False
    t0 = time.time()
    bar = _EncodingBar()
    try:
        s = _api_index(root, ns.dim, bar, multimodal=getattr(ns, "multimodal", None))
    finally:
        bar.close()
    _report(root, s)
    print(f"done in {time.time() - t0:.1f}s")
    return True


def cmd_index(ns) -> int:
    root = Path(ns.root).expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    print(f"indexing {root}")
    if _build_via_daemon(root, ns):
        return 0
    _build_and_report(root, ns)
    return 0


def cmd_update(ns) -> int:
    """Incremental refresh; exits without loading the model when already fresh."""
    root = Path(ns.root).expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    if not index_is_stale(root):
        m = load_meta(root)
        print(f"up to date: {m['files']:,} files, {m['chunks']:,} chunks, "
              f"updated {m['updated']}{_ago(m.get('updated_epoch'))}")
        return 0
    print(f"changes detected, updating {root}")
    if _build_via_daemon(root, ns):
        return 0
    _build_and_report(root, ns)
    return 0


def _print_hit_source(root: Path, h: dict) -> None:
    if h.get("kind") == "image" or h["path"].lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        print(f"      {h.get('preview') or _first_line(root, h['path'], h['start'])}")
        return
    if h.get("kind") == "pdf" or h["path"].lower().endswith(".pdf"):
        target_page = h["start"] + 1
        page_lines: list[str] = []
        try:
            for pnum, ptext in indexer._extract_pdf_pages(root / h["path"]):
                if pnum == target_page:
                    page_lines = [line.strip() for line in ptext.splitlines() if line.strip()]
                    break
        except Exception:
            pass
        if page_lines:
            for line in page_lines[:20]:
                print(f"      p.{target_page}  {line[:120]}")
        else:
            print(f"      {h.get('preview') or _first_line(root, h['path'], h['start'])}")
        return
    try:
        lines = Path(root, h["path"]).read_text(errors="replace").splitlines()
        chunk = lines[h["start"]:h["end"]]
    except (OSError, IndexError):
        print("      <source unavailable>")
        return
    width = len(str(h["end"]))
    for i, line in enumerate(chunk, start=h["start"] + 1):
        print(f"      {i:>{width}}  {line.rstrip()[:120]}")


def _print_hits(root: Path, ns, hits: list[dict]) -> None:
    if not hits:
        print("empty index — run `python -m csearch index` first")
        return
    print(f"query: {ns.query!r}")
    width = max(len(h["path"]) for h in hits)
    for rank, h in enumerate(hits, 1):
        loc = f"{h['path']}:{h['start'] + 1}-{h['end']}"
        snippet = _first_line(root, h["path"], h["start"])
        print(f"  {rank}. {h['score']:.3f}  {loc:<{width + 10}}  {snippet}")
        if ns.show:
            _print_hit_source(root, h)


def cmd_search(ns) -> int:
    root = Path(ns.root).expanduser().resolve()
    if not _use_local(ns):
        if not _ensure_daemon():
            print("daemon unreachable; loading model in-process", file=sys.stderr)
        else:
            try:
                data = _api_search(root, ns.query, ns.k)
            except _DaemonDown as e:
                print(f"daemon dropped mid-query ({e}); loading model in-process",
                      file=sys.stderr)
            except RuntimeError as e:
                print(str(e), file=sys.stderr)
                return 2
            else:
                if data.get("note"):
                    print(data["note"], file=sys.stderr)
                _print_hits(root, ns, data["hits"])
                return 0
    try:
        searcher = Searcher(root)
    except FileNotFoundError as e:
        print(f"{e}", file=sys.stderr)
        return 2
    print("loading model ...", file=sys.stderr, flush=True)
    embedder = _make_embedder(ns)
    hits = searcher.search(embedder, ns.query, k=ns.k)
    _print_hits(root, ns, hits)
    return 0


def _first_line(root: Path, rel: str, start: int) -> str:
    return first_line(root, rel, start, limit=80)


def cmd_status(ns) -> int:
    """Index existence and freshness — no model load, fast."""
    root = Path(ns.root).expanduser().resolve()
    index_dir = index_dir_for(root)
    if not (index_dir / "meta.json").exists():
        print(f"no index at {index_dir} — run `python -m csearch index {ns.root}`")
        return 1
    try:
        m = load_meta(root)
    except (OSError, KeyError, ValueError) as e:
        print(f"index at {index_dir} is unreadable ({e}) — delete it and re-run `index`")
        return 1
    stale = index_is_stale(root)
    print(f"index: {index_dir}")
    print(f"  status    {'stale (files changed since last index)' if stale else 'fresh'}")
    print(f"  files     {m['files']:,}")
    print(f"  chunks    {m['chunks']:,}")
    print(f"  model     {m['model_dir']} ({m['backend']}, dim {m['dim']})")
    print(f"  updated   {m['updated']}{_ago(m.get('updated_epoch'))}")
    return 0


def cmd_stats(ns) -> int:
    try:
        m = load_meta(Path(ns.root).expanduser().resolve())
    except FileNotFoundError as e:
        print(f"{e}", file=sys.stderr)
        return 2
    print(f"chunks={m['chunks']:,} files={m['files']:,} dim={m['dim']} "
          f"backend={m['backend']} model={m['model_dir']} updated={m['updated']}")
    return 0


def cmd_completions(ns) -> int:
    """`completions <shell>`: print script; `completions install|uninstall [shell…]`: manage."""
    if ns.action in (None, *SHELLS):
        if ns.shells:
            print(f"unexpected arguments: {' '.join(ns.shells)}", file=sys.stderr)
            return 2
        if ns.action is None:
            print("usage: csearch completions {bash|zsh|fish|install|uninstall} [shell ...]",
                  file=sys.stderr)
            return 2
        sys.stdout.write(COMPLETIONS[ns.action])
        return 0
    if ns.action in ("install", "uninstall"):
        shells = tuple(ns.shells) or ()
        if not shells:
            d = default_shell()
            if d is None:
                print("specify a shell: csearch completions install bash|zsh|fish "
                      f"($SHELL={os.environ.get('SHELL', '?')} is none of {', '.join(SHELLS)})",
                      file=sys.stderr)
                return 2
            shells = (d,)
        bad = [s for s in shells if s not in SHELLS]
        if bad:
            print(f"unknown shell(s): {' '.join(bad)} — choose from {', '.join(SHELLS)}",
                  file=sys.stderr)
            return 2
        rc = 0
        op = install if ns.action == "install" else uninstall
        for s in shells:
            print(f"{ns.action} {s}:")
            rc |= op(s)
        return rc
    print(f"unknown action: {ns.action} — expected a shell name or install/uninstall",
          file=sys.stderr)
    return 2


SUBCOMMANDS = frozenset({"index", "update", "search", "status", "stats", "completions"})
ALIASES = {"i": "index", "u": "update", "s": "search", "st": "status"}
_VALUED_FLAGS = {"-k", "--model", "--backend", "--variant", "--dim", "--batch"}


def _positional_args(args: list[str]) -> list[str]:
    pos: list[str] = []
    skip = False
    for i, token in enumerate(args):
        if skip:
            skip = False
            continue
        if token == "--":
            pos.extend(args[i + 1:])
            break
        if token.startswith("-"):
            if token in _VALUED_FLAGS:
                skip = True
            continue
        pos.append(token)
    return pos


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ALIASES:
        argv = [ALIASES[argv[0]], *argv[1:]]
    if argv and argv[0] not in SUBCOMMANDS:
        # bare invocation: `csearch "query"` or `csearch <repo> "query"`
        if not argv[0].startswith("-"):
            pos = _positional_args(argv)
            if len(pos) >= 2 and Path(pos[0]).expanduser().is_dir():
                argv = ["search", argv[0], *argv[1:]]
            else:
                argv = ["search", ".", *argv]
    elif argv and argv[0] == "search":
        pos = _positional_args(argv[1:])
        if len(pos) == 1:
            argv.insert(1, ".")
    ap = argparse.ArgumentParser(
        prog="csearch",
        description="EmbeddingGemma 2 local code search (human CLI; agents: csearch.mcp_server)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build the index, or force an incremental rebuild")
    p.add_argument("root", nargs="?", default=".", help="repo root (default: .)")
    _add_model_args(p)
    p.add_argument("--local", action="store_true",
                   help="embed in this process instead of the shared daemon")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("update", help="incrementally refresh; instant no-op when fresh")
    p.add_argument("root", nargs="?", default=".")
    _add_model_args(p)
    p.add_argument("--local", action="store_true",
                   help="embed in this process instead of the shared daemon")
    p.set_defaults(fn=cmd_update)

    p = sub.add_parser("search", help="semantic query -> file:line hits "
                                      "(also the default when the first arg "
                                      "is not a subcommand)")
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--show", action="store_true", help="print the source of each hit")
    _add_model_args(p)
    p.add_argument("--local", action="store_true",
                   help="embed in this process instead of the shared daemon")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("status", help="index freshness without loading the model")
    p.add_argument("root", nargs="?", default=".")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("stats", help="one-line index metadata")
    p.add_argument("root", nargs="?", default=".")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("completions", help="print or install/uninstall shell completions")
    p.add_argument("action", nargs="?", default=None, metavar="SHELL|ACTION",
                   help="bash|zsh|fish print a script; install/uninstall manage files")
    p.add_argument("shells", nargs="*", metavar="shell",
                   help="shells for install/uninstall (default: from $SHELL)")
    p.set_defaults(fn=cmd_completions)

    ns = ap.parse_args(argv)
    try:
        return ns.fn(ns)
    except Exception as e:
        if "Failed to allocate" in str(e) or "out of memory" in str(e).lower():
            print("error: GPU out of memory — another session is likely holding VRAM; "
                  "retry with --cpu", file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
