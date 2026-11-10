"""Incremental code index: walk, filter, chunk, embed, persist.

Layout under <root>/.csearch/:
  meta.json       index parameters (dim, backend, model dir, ...)
  manifest.jsonl  one record per chunk: path, line range, file content hash
  vectors.npy     float32 embeddings, row i = manifest line i
  files.json      paths seen at last build; staleness ground truth (the manifest
                  omits undecodable/zero-chunk files)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .chunker import chunk_code
from .embedder import Embedder

INDEX_DIRNAME = ".csearch"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTS = {".pdf"}
MIN_IMAGE_BYTES = 512      # skip tiny icons < 512B
MIN_IMAGE_DIM = 64          # skip tiny icons < 64x64 px

_PDF_PAGE_CACHE: dict[tuple[str, int], list[tuple[int, str]]] = {}


def _extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Extract (1-based page_num, page_text) for non-empty pages in a PDF file."""
    try:
        resolved = str(path.resolve())
        mtime = path.stat().st_mtime_ns
        cache_key = (resolved, mtime)
        if cache_key in _PDF_PAGE_CACHE:
            return _PDF_PAGE_CACHE[cache_key]
    except OSError:
        cache_key = None

    pages: list[tuple[int, str]] = []
    # 1. Try pypdf if installed
    try:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
        if pages:
            if cache_key is not None:
                if len(_PDF_PAGE_CACHE) > 128:
                    _PDF_PAGE_CACHE.clear()
                _PDF_PAGE_CACHE[cache_key] = pages
            return pages
    except Exception:
        pass

    # 2. Fallback to pdftotext CLI
    if shutil.which("pdftotext"):
        try:
            proc = subprocess.run(
                ["pdftotext", "-enc", "UTF-8", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            raw_pages = proc.stdout.split("\f")
            for i, ptext in enumerate(raw_pages, start=1):
                if ptext.strip():
                    pages.append((i, ptext))
        except Exception:
            pass

    if cache_key is not None and pages:
        if len(_PDF_PAGE_CACHE) > 128:
            _PDF_PAGE_CACHE.clear()
        _PDF_PAGE_CACHE[cache_key] = pages
    return pages


def ensure_gitignored(root: Path) -> str:
    """Idempotently add '.csearch/' to <root>/.gitignore. Returns status text."""
    entry = INDEX_DIRNAME + "/"
    gi = root / ".gitignore"
    try:
        text = gi.read_text() if gi.exists() else ""
    except OSError as e:
        return f"skipped ({e})"
    if any(line.strip() in (entry, INDEX_DIRNAME) for line in text.splitlines()):
        return "already ignored"
    if text and not text.endswith("\n"):
        text += "\n"
    try:
        gi.write_text(text + entry + "\n")
    except OSError as e:
        return f"skipped ({e})"
    return f"added {entry} to .gitignore"


_ensure_gitignored = ensure_gitignored


def first_line(root: Path, rel: str, start: int, limit: int = 100) -> str:
    ext = os.path.splitext(rel)[1].lower()
    if ext in IMAGE_EXTS:
        try:
            from PIL import Image

            with Image.open(root / rel) as img:
                return f"[image {img.width}x{img.height}]"[:limit]
        except Exception:
            return "[image]"[:limit]
    if ext in PDF_EXTS:
        try:
            pages = _extract_pdf_pages(root / rel)
            target_page = start + 1
            for pnum, ptext in pages:
                if pnum == target_page:
                    for line in ptext.splitlines():
                        s = line.strip()
                        if s:
                            return f"[pdf p.{target_page}] {s}"[:limit]
            return f"[pdf page {target_page}]"[:limit]
        except Exception:
            return f"[pdf page {start + 1}]"[:limit]
    try:
        line = (root / rel).read_text(errors="replace").splitlines()[start]
        return line.strip()[:limit]
    except (OSError, IndexError):
        return ""


_first_line = first_line


# ---- lexical channel --------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]")
_ID_HINT_RE = re.compile(
    r"\w+\.\w{1,4}\b|[A-Z][a-z]+[A-Z]\w*|[a-z]+[A-Z]\w*|\b\d{2,}\b|[\w-]+/[\w.-]+"
)


def looks_literal(query: str) -> bool:
    return bool(_ID_HINT_RE.search(query))


_looks_literal = looks_literal


def _toks(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _read_chunk(root: Path, rec: dict) -> str:
    kind = rec.get("kind")
    ext = os.path.splitext(rec.get("path", ""))[1].lower()
    if kind == "image" or ext in IMAGE_EXTS:
        return f"{rec['path']} {rec.get('preview', '')}"
    if kind == "pdf" or ext in PDF_EXTS:
        page_num = rec.get("start", 0) + 1
        try:
            pages = _extract_pdf_pages(root / rec["path"])
            for p, txt in pages:
                if p == page_num:
                    return txt
        except Exception:
            pass
        return rec.get("preview", "")
    try:
        lines = (root / rec["path"]).read_text(errors="replace").splitlines()
        return "\n".join(lines[rec["start"] : rec["end"]])
    except OSError:
        return ""


class LexIndex:
    """Inverted TF-IDF scorer over manifest chunk texts. Build is one pass over
    the corpus (~1-2s per repo, cached); queries touch only matching postings."""

    def __init__(self, root: Path, records: list[dict]):
        df: dict[str, int] = {}
        postings: dict[str, list[tuple[int, dict[str, int]]]] = {}
        file_cache: dict[str, list[str]] = {}
        pdf_cache: dict[str, dict[int, str]] = {}

        def get_chunk(rec: dict) -> str:
            path_str = rec["path"]
            kind = rec.get("kind")
            ext = os.path.splitext(path_str)[1].lower()
            if kind == "image" or ext in IMAGE_EXTS:
                return f"{path_str} {rec.get('preview', '')}"
            if kind == "pdf" or ext in PDF_EXTS:
                if path_str not in pdf_cache:
                    try:
                        pages = _extract_pdf_pages(root / path_str)
                        pdf_cache[path_str] = {p: txt for p, txt in pages}
                    except Exception:
                        pdf_cache[path_str] = {}
                page_num = rec.get("start", 0) + 1
                page_text = pdf_cache[path_str].get(page_num, "")
                preview = rec.get("preview", "")
                return f"{path_str} {preview} {page_text}"
            if path_str not in file_cache:
                try:
                    file_cache[path_str] = (root / path_str).read_text(errors="replace").splitlines()
                except OSError:
                    file_cache[path_str] = []
            lines = file_cache[path_str]
            return "\n".join(lines[rec["start"] : rec["end"]])

        for i, rec in enumerate(records):
            tf: dict[str, int] = {}
            for t in _toks(get_chunk(rec)):
                tf[t] = tf.get(t, 0) + 1
            for t in tf:
                df[t] = df.get(t, 0) + 1
                postings.setdefault(t, []).append((i, tf))
        self.n = len(records)
        self.postings = {
            t: [(i, tf, np.float32(np.log((self.n + 1) / (df[t] + 1)) + 1.0)) for i, tf in pl]
            for t, pl in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        qtf: dict[str, int] = {}
        for t in _toks(query):
            qtf[t] = qtf.get(t, 0) + 1
        for t, f in qtf.items():
            for i, doc_tf, w in self.postings.get(t, ()):
                dtf = doc_tf.get(t, 0)
                if dtf:
                    out[i] += w * f * dtf
        return out


def rrf(a: np.ndarray, b: np.ndarray, k: int = 60) -> np.ndarray:
    ra, rb = np.empty(len(a)), np.empty(len(b))
    ra[np.argsort(-a)] = np.arange(1, len(a) + 1)
    rb[np.argsort(-b)] = np.arange(1, len(b) + 1)
    return 1 / (k + ra) + 1 / (k + rb)


_rrf = rrf

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".nox",
    "dist", "build", "target", ".next", ".nuxt", ".idea", ".vscode",
    INDEX_DIRNAME,
}

CODE_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts", ".go", ".rs", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".cxx", ".cs", ".rb", ".php", ".swift", ".m", ".mm", ".scala",
    ".lua", ".r", ".dart", ".vue", ".svelte", ".sql", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".toml", ".yaml", ".yml", ".ini", ".cfg",
    ".proto", ".graphql", ".sol", ".vim", ".ex", ".exs", ".erl", ".clj",
    ".cljs", ".hs", ".nim", ".zig", ".jl", ".pl", ".pm", ".tf", ".hcl",
    ".md", ".rst", ".txt",
}

MULTIMODAL_EXTS = CODE_EXTS | IMAGE_EXTS | PDF_EXTS


def multimodal_enabled(multimodal: bool | None = None) -> bool:
    if multimodal is not None:
        return bool(multimodal)
    return os.environ.get("CSEARCH_MULTIMODAL") == "1"

META_NAME = "meta.json"
MANIFEST_NAME = "manifest.jsonl"
VECTORS_NAME = "vectors.npy"
FILES_NAME = "files.json"


def index_dir_for(root: Path) -> Path:
    return root / INDEX_DIRNAME


def load_meta(root: Path, index_dir: Path | None = None) -> dict:
    """Read meta.json without touching manifest/vectors (metadata-only fast path)."""
    root = root.resolve()
    index_dir = index_dir or index_dir_for(root)
    try:
        return json.loads((index_dir / META_NAME).read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"no index at {index_dir}; run `index` first") from None


def index_is_stale(root: Path, index_dir: Path | None = None) -> bool:
    """True when the index is missing, unreadable, or any file changed on disk.

    Detects modified mtimes, added files, and removed files in one walk.
    Old indexes without ``updated_epoch`` count as stale once, self-healing.
    """
    root = root.resolve()
    index_dir = index_dir or index_dir_for(root)
    try:
        meta = load_meta(root, index_dir)
    except (OSError, ValueError):
        return True
    updated = meta.get("updated_epoch")
    if updated is None:
        return True
    mm = bool(meta.get("multimodal", False))
    files_path = index_dir / FILES_NAME
    if files_path.exists():
        try:
            indexed = set(json.loads(files_path.read_text()))
        except (OSError, json.JSONDecodeError):
            return True
    else:
        # old layout: the manifest only holds chunk-bearing files, so any
        # undecodable/zero-chunk file made every walk look stale; the next
        # build writes files.json and self-heals
        try:
            with (index_dir / MANIFEST_NAME).open() as f:
                indexed = {json.loads(line)["path"] for line in f}
        except (OSError, json.JSONDecodeError, KeyError):
            return True
    seen: set[str] = set()
    for path in discover_files(root, multimodal=mm):
        rel = path.relative_to(root).as_posix()
        seen.add(rel)
        try:
            if path.stat().st_mtime > updated:
                return True
        except OSError:
            return True
    return seen != indexed


def discover_files(
    root: Path, exts: set[str] | None = None, multimodal: bool | None = None
) -> list[Path]:
    if exts is None:
        exts = MULTIMODAL_EXTS if multimodal_enabled(multimodal) else CODE_EXTS
    files: list[Path] = []
    stack = [root]
    while stack:
        try:
            # materialize before descending: entries belong to this dir's iterator
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            if is_dir:
                # prune skipped subtrees before walking them; symlinked dirs are
                # listed but never entered (rglob's no-recurse-into-symlinks rule)
                if entry.name not in SKIP_DIRS and not entry.is_symlink():
                    stack.append(Path(entry.path))
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in exts:
                continue
            if ext in IMAGE_EXTS:
                try:
                    if entry.stat().st_size < MIN_IMAGE_BYTES:
                        continue
                except OSError:
                    continue
            files.append(Path(entry.path))
    files.sort()
    return files


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _atomic_write(path: Path, data: str) -> None:
    # readers (MCP metadata path) key on complete files: never expose half a write
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


def _publish_meta(
    root: Path,
    index_dir: Path,
    embedder: Embedder,
    n_files: int,
    n_chunks: int,
    dim: int | None = None,
    multimodal: bool = False,
) -> None:
    target_dim = embedder.dim if dim is None else dim
    _atomic_write(
        index_dir / META_NAME,
        json.dumps(
            {
                "root": str(root),
                "dim": target_dim,
                "backend": embedder.backend,
                "model_dir": str(embedder.model_dir),
                "files": n_files,
                "chunks": n_chunks,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_epoch": time.time(),
                "multimodal": bool(multimodal),
            },
            ensure_ascii=False,
            indent=1,
        ),
    )


def _load_state(index_dir: Path) -> tuple[list[dict], np.ndarray | None]:
    manifest_path = index_dir / MANIFEST_NAME
    vectors_path = index_dir / VECTORS_NAME
    records: list[dict] = []
    vectors = None
    if manifest_path.exists() and vectors_path.exists():
        with manifest_path.open() as f:
            records = [json.loads(line) for line in f]
        # mmap: build only reads the rows it reuses
        vectors = np.load(vectors_path, mmap_mode="r")
    return records, vectors


def build_index(
    root: Path,
    embedder: Embedder,
    index_dir: Path | None = None,
    exts: set[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    dim: int | None = None,
    multimodal: bool | None = None,
) -> dict:
    """Create or incrementally update the index. Returns stats."""
    root = root.resolve()
    target_dim = embedder.dim if dim is None else dim
    index_dir = index_dir or index_dir_for(root)
    index_dir.mkdir(parents=True, exist_ok=True)

    old_records, old_vectors = _load_state(index_dir)
    # changing dim/backend/model invalidates all vectors: rebuild from scratch
    meta_path = index_dir / META_NAME
    if meta_path.exists():
        try:
            prev_meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            prev_meta = None  # unreadable: old vectors' dim can't be trusted
        if prev_meta is None or (prev_meta.get("dim"), prev_meta.get("backend"), prev_meta.get("model_dir")) != (
            target_dim, embedder.backend, str(embedder.model_dir)
        ):
            old_records, old_vectors = [], None
    else:
        prev_meta = None

    if multimodal is None and prev_meta is not None:
        use_mm = bool(prev_meta.get("multimodal", False)) or multimodal_enabled(None)
    else:
        use_mm = multimodal_enabled(multimodal)

    # transient: row of each old record in the old vectors matrix
    for row, rec in enumerate(old_records):
        rec["_row"] = row
    old_by_path: dict[str, list[dict]] = {}
    for rec in old_records:
        old_by_path.setdefault(rec["path"], []).append(rec)
    old_vectors = (
        old_vectors if old_vectors is not None and len(old_vectors) == len(old_records) else None
    )

    files = discover_files(root, exts=exts, multimodal=use_mm)
    rels = [path.relative_to(root).as_posix() for path in files]
    titles: list[str] = []
    texts: list[str] = []
    fresh_text_indices: list[int] = []

    fresh_image_indices: list[int] = []
    image_titles: list[str] = []
    image_paths: list[Path] = []
    image_previews: list[str] = []

    new_records: list[dict] = []
    reused = 0
    changed_files = 0

    for path, rel in zip(files, rels):
        data = path.read_bytes()
        fhash = _file_hash(data)
        prev = old_by_path.get(rel)
        if prev and prev[0]["hash"] == fhash and old_vectors is not None:
            new_records.extend(prev)
            reused += len(prev)
            continue

        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            width, height = 0, 0
            try:
                from PIL import Image

                with Image.open(path) as img:
                    width, height = img.size
                    if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
                        continue
            except Exception:
                continue
            changed_files += 1
            rec = {
                "path": rel,
                "hash": fhash,
                "start": 0,
                "end": 1,
                "kind": "image",
                "preview": f"[image {width}x{height}]",
            }
            new_records.append(rec)
            fresh_image_indices.append(len(new_records) - 1)
            image_titles.append(rel)
            image_paths.append(path)
            image_previews.append(rec["preview"])
            continue

        if ext in PDF_EXTS:
            pages = _extract_pdf_pages(path)
            if not pages:
                continue
            changed_files += 1
            for page_num, page_text in pages:
                for _c_start, _c_end, chunk in chunk_code(page_text):
                    first_ln = next((ln.strip()[:100] for ln in chunk.splitlines() if ln.strip()), "[pdf]")
                    rec = {
                        "path": rel,
                        "hash": fhash,
                        "start": page_num - 1,
                        "end": page_num,
                        "kind": "pdf",
                        "preview": f"[pdf p.{page_num}] {first_ln}",
                    }
                    new_records.append(rec)
                    fresh_text_indices.append(len(new_records) - 1)
                    titles.append(f"{rel} (page {page_num})")
                    texts.append(chunk)
            continue

        try:
            src = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        changed_files += 1
        for start, end, chunk in chunk_code(src):
            new_records.append({"path": rel, "hash": fhash, "start": start, "end": end})
            fresh_text_indices.append(len(new_records) - 1)
            titles.append(rel)
            texts.append(chunk)

    try:
        prev_files = json.loads((index_dir / FILES_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        prev_files = None  # legacy layout: force the full self-healing write path

    gitignore_status = ensure_gitignored(root)

    if (
        changed_files == 0
        and not texts
        and not fresh_image_indices
        and reused == len(old_records)  # no old rows dropped (e.g. now-undecodable file)
        and prev_files == rels
        and old_vectors is not None
    ):
        # no-op: every file reused — keep vectors/manifest/files.json bytes and
        # mtimes (external caches fingerprint them); refresh only meta so mtime
        # staleness still self-heals
        _publish_meta(root, index_dir, embedder, len(files), len(new_records), dim=target_dim, multimodal=use_mm)
        return {
            "files": len(files),
            "chunks": len(new_records),
            "reembedded_chunks": 0,
            "reused_chunks": reused,
            "changed_files": 0,
            "encode_seconds": 0.0,
            "gitignore": gitignore_status,
        }

    t0 = time.time()
    vectors = np.empty((len(new_records), target_dim), np.float32)
    if texts:
        encoded = embedder.encode_docs(titles, texts, progress=progress, dim=target_dim)
        for idx, vec in zip(fresh_text_indices, encoded):
            vectors[idx] = vec
    if fresh_image_indices:
        encoded_imgs = None
        if hasattr(embedder, "encode_images"):
            try:
                encoded_imgs = embedder.encode_images(
                    image_titles, image_paths, progress=progress, dim=target_dim
                )
            except Exception:
                encoded_imgs = None
        if encoded_imgs is None:
            encoded_imgs = embedder.encode_docs(image_titles, image_previews, dim=target_dim)
        for idx, vec in zip(fresh_image_indices, encoded_imgs):
            vectors[idx] = vec
    encode_s = time.time() - t0

    # splice reused rows (old matrix) and freshly encoded rows into one matrix
    kept_new_pos, kept_old_rows = [], []
    for pos, rec in enumerate(new_records):
        if "_row" in rec:
            kept_new_pos.append(pos)
            kept_old_rows.append(rec.pop("_row"))
    if kept_new_pos:
        vectors[kept_new_pos] = old_vectors[kept_old_rows]
    old_vectors = None

    # replace, not rewrite in place: a live Searcher's mmap of the old inode
    # stays valid and no reader ever sees a truncated/half-written file
    vectors_tmp = index_dir / (VECTORS_NAME + ".tmp")
    with vectors_tmp.open("wb") as f:
        np.save(f, vectors)
    os.replace(vectors_tmp, index_dir / VECTORS_NAME)
    _atomic_write(
        index_dir / MANIFEST_NAME,
        "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in new_records),
    )
    _atomic_write(index_dir / FILES_NAME, json.dumps(rels, ensure_ascii=False))
    _publish_meta(root, index_dir, embedder, len(files), len(new_records), dim=target_dim, multimodal=use_mm)

    return {
        "files": len(files),
        "chunks": len(new_records),
        "reembedded_chunks": len(texts) + len(fresh_image_indices),
        "reused_chunks": reused,
        "changed_files": changed_files,
        "encode_seconds": round(encode_s, 1),
        "gitignore": gitignore_status,
    }


class Searcher:
    def __init__(self, root: Path, index_dir: Path | None = None):
        self.root = root.resolve()
        self.index_dir = index_dir or index_dir_for(self.root)
        self.meta = load_meta(self.root, self.index_dir)
        with (self.index_dir / MANIFEST_NAME).open() as f:
            self.records = [json.loads(line) for line in f]
        # mmap: init is O(1) and rebuilds that replace the file never SIGBUS us
        self.vectors = np.load(self.index_dir / VECTORS_NAME, mmap_mode="r")
        self._lex: LexIndex | None = None

    def _get_lex(self) -> LexIndex:
        if self._lex is None:
            self._lex = LexIndex(self.root, self.records)
        return self._lex

    def search(self, embedder: Embedder, query: str, k: int = 5, dim: int | None = None) -> list[dict]:
        target_dim = dim if dim is not None else self.meta.get("dim")
        q = embedder.encode_queries([query], dim=target_dim)[0]
        vec = self.vectors @ q
        if looks_literal(query):
            lex = self._get_lex().scores(query)
            if lex.any():
                vec = rrf(vec, lex)
        return self.rank(vec, k=k)

    def rank(self, scores: np.ndarray, k: int = 5) -> list[dict]:
        """Top-k hits for precomputed scores (vector, or fused vector+lexical)."""
        k = min(k, len(self.records))
        if k == 0:
            return []
        top = np.argpartition(scores, len(scores) - k)[-k:]
        top = top[np.argsort(-scores[top])]
        return [{"score": float(scores[idx]), **self.records[idx]} for idx in top]
