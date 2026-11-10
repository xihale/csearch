"""Behavioral tests for csearch.indexer: incremental invariants, pruning,
no-op builds, dim invalidation, legacy layouts, mmap-backed Searcher."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from csearch.embedder import Embedder

from csearch.indexer import (
    CODE_EXTS,
    FILES_NAME,
    MANIFEST_NAME,
    META_NAME,
    MULTIMODAL_EXTS,
    VECTORS_NAME,
    Searcher,
    build_index,
    discover_files,
    first_line,
    index_dir_for,
    index_is_stale,
    load_meta,
    multimodal_enabled,
)


class FakeEmbedder(Embedder):
    """Deterministic hash-based stand-in; same interface as Embedder, no model."""

    def __init__(self, dim=8, backend="fake", model_dir="/fake-model"):
        self.dim = dim
        self.backend = backend
        self.model_dir = Path(model_dir)

    def _vec(self, text: str, dim: int | None = None) -> np.ndarray:
        target_dim = dim or self.dim
        digest = hashlib.sha256(f"{target_dim}|{text}".encode()).digest()
        vec = np.frombuffer(digest * -(-target_dim // len(digest)), np.uint8)[: target_dim].astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode_docs(self, titles, texts, progress=None, dim=None):
        target_dim = dim or self.dim
        vecs = [self._vec(f"{t}|{x}", dim=target_dim) for t, x in zip(titles, texts)]
        return np.stack(vecs) if vecs else np.empty((0, target_dim), np.float32)

    def encode_queries(self, texts, dim=None):
        target_dim = dim or self.dim
        return np.stack([self._vec(f"query|{x}", dim=target_dim) for x in texts])


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def src(n_lines: int = 60) -> str:
    return "".join(f"def f{i}():\n    return {i}\n" for i in range(n_lines))


def manifest_rows(index_dir: Path) -> list[dict]:
    with (index_dir / MANIFEST_NAME).open() as f:
        return [json.loads(line) for line in f]


def vector_rows(index_dir: Path) -> np.ndarray:
    return np.load(index_dir / VECTORS_NAME, mmap_mode="r")


class IndexerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.index_dir = index_dir_for(self.root)
        self.emb = FakeEmbedder()

    # -- discover_files ------------------------------------------------------

    def test_discover_prunes_skips_and_keeps_symlink_semantics(self):
        write(self.root, "src/a.py", "x = 1\n")
        write(self.root, "src/sub/b.TS", "x = 1\n")  # suffix match is case-insensitive
        write(self.root, "realdir/g.py", "x = 1\n")
        write(self.root, "data.json", "{}\n")  # not a code extension
        write(self.root, "assets/logo.png", "x")  # not a code extension
        write(self.root, "node_modules/pkg/d.py", "x = 1\n")  # pruned subtree
        write(self.root, ".git/objects/e.py", "x = 1\n")  # pruned subtree
        os.symlink(self.root / "realdir", self.root / "dirlink")
        write(self.root, "realdir/inner.py", "x = 1\n")
        os.symlink(self.root / "src/a.py", self.root / "link.py")  # symlinked file

        found = discover_files(self.root)
        self.assertEqual(
            found,
            [
                self.root / "link.py",
                self.root / "realdir/g.py",
                self.root / "realdir/inner.py",
                self.root / "src/a.py",
                self.root / "src/sub/b.TS",
            ],
        )

    def test_multimodal_enabled_helper(self):
        from unittest.mock import patch

        self.assertTrue(multimodal_enabled(True))
        self.assertFalse(multimodal_enabled(False))
        with patch.dict(os.environ, {"CSEARCH_MULTIMODAL": "1"}):
            self.assertTrue(multimodal_enabled())
            self.assertTrue(multimodal_enabled(None))
            self.assertFalse(multimodal_enabled(False))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(multimodal_enabled())
            self.assertFalse(multimodal_enabled(None))
            self.assertTrue(multimodal_enabled(True))

    def test_discover_files_multimodal(self):
        write(self.root, "a.py", "x = 1\n")
        (self.root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 600)
        (self.root / "doc.pdf").write_bytes(b"%PDF-1.4\n" + b"\x00" * 100)

        self.assertEqual(discover_files(self.root), [self.root / "a.py"])
        self.assertEqual(
            discover_files(self.root, multimodal=False),
            [self.root / "a.py"],
        )
        self.assertEqual(
            discover_files(self.root, multimodal=True),
            [self.root / "a.py", self.root / "doc.pdf", self.root / "logo.png"],
        )

    # -- build / incremental invariants --------------------------------------

    def test_add_modify_delete_keeps_manifest_row_alignment(self):
        write(self.root, "a.py", src())
        write(self.root, "b.py", src(70))
        stats = build_index(self.root, self.emb)
        self.assertEqual(stats["files"], 2)
        self.assertEqual(stats["changed_files"], 2)
        self.assertEqual(stats["reused_chunks"], 0)
        self.assertFalse(index_is_stale(self.root))

        rows1 = manifest_rows(self.index_dir)
        vec1 = vector_rows(self.index_dir)
        self.assertEqual(vec1.shape[0], len(rows1))  # row i of vectors = line i
        self.assertEqual({r["path"] for r in rows1}, {"a.py", "b.py"})
        b_idx1 = [i for i, r in enumerate(rows1) if r["path"] == "b.py"]
        a_idx1 = [i for i, r in enumerate(rows1) if r["path"] == "a.py"]
        b_vecs1 = vec1[b_idx1].copy()
        a_vec1 = vec1[a_idx1[0]].copy()

        write(self.root, "a.py", src(61).replace("return 0", "return 9", 1))
        self.assertTrue(index_is_stale(self.root))
        stats = build_index(self.root, self.emb)
        self.assertEqual(stats["changed_files"], 1)
        self.assertGreater(stats["reused_chunks"], 0)

        rows2 = manifest_rows(self.index_dir)
        vec2 = vector_rows(self.index_dir)
        self.assertEqual(vec2.shape[0], len(rows2))
        # untouched file: manifest records and vector rows reused verbatim
        b_idx2 = [i for i, r in enumerate(rows2) if r["path"] == "b.py"]
        self.assertEqual([rows2[i] for i in b_idx2], [rows1[i] for i in b_idx1])
        self.assertTrue(np.array_equal(vec2[b_idx2], b_vecs1))
        # changed file: rows re-encoded
        a_idx2 = [i for i, r in enumerate(rows2) if r["path"] == "a.py"]
        self.assertFalse(np.array_equal(vec2[a_idx2[0]], a_vec1))
        self.assertFalse(index_is_stale(self.root))

        (self.root / "b.py").unlink()  # delete
        self.assertTrue(index_is_stale(self.root))
        stats = build_index(self.root, self.emb)
        self.assertEqual(stats["files"], 1)
        self.assertFalse(any(r["path"] == "b.py" for r in manifest_rows(self.index_dir)))
        self.assertTrue(np.array_equal(vector_rows(self.index_dir), vec2[: len(vec2) - len(b_idx1)]))
        self.assertFalse(index_is_stale(self.root))

    def test_touch_noop_preserves_vectors_manifest_files_and_refreshes_meta(self):
        write(self.root, "a.py", src())
        write(self.root, "b.py", src(70))
        build_index(self.root, self.emb)
        manifest_bytes = (self.index_dir / MANIFEST_NAME).read_bytes()
        mtimes = {
            name: (self.index_dir / name).lstat().st_mtime_ns
            for name in (VECTORS_NAME, MANIFEST_NAME, FILES_NAME)
        }
        old_epoch = load_meta(self.root)["updated_epoch"]

        os.utime(self.root / "a.py")  # mtime bumped, content identical
        self.assertTrue(index_is_stale(self.root))
        stats = build_index(self.root, self.emb)
        self.assertEqual(stats["changed_files"], 0)
        self.assertEqual(stats["reembedded_chunks"], 0)
        self.assertGreater(stats["reused_chunks"], 0)

        # heavy artifacts untouched (external caches fingerprint mtimes/inodes)
        for name, ns in mtimes.items():
            self.assertEqual((self.index_dir / name).lstat().st_mtime_ns, ns, name)
        self.assertEqual((self.index_dir / MANIFEST_NAME).read_bytes(), manifest_bytes)
        # only meta refreshes, clearing mtime staleness
        self.assertGreater(load_meta(self.root)["updated_epoch"], old_epoch)
        self.assertFalse(index_is_stale(self.root))
        self.assertFalse(list(self.index_dir.glob("*.tmp")))
        # rows preserved and searchable
        self.assertEqual(manifest_rows(self.index_dir), [json.loads(l) for l in manifest_bytes.decode().splitlines()])

    def test_utf8_file_becoming_binary_removes_old_chunks(self):
        path = write(self.root, "a.py", src())
        build_index(self.root, self.emb)
        old_searcher = Searcher(self.root)
        old_vectors = old_searcher.vectors.copy()
        path.write_bytes(b"\xff\xfe\x00")
        stats = build_index(self.root, self.emb)
        self.assertEqual(stats["chunks"], 0)
        self.assertEqual(manifest_rows(self.index_dir), [])
        self.assertEqual(Searcher(self.root).search(self.emb, "q", 5), [])
        self.assertTrue(np.array_equal(old_searcher.vectors, old_vectors))
        self.assertFalse(index_is_stale(self.root))

    def test_empty_root_roundtrip(self):
        stats = build_index(self.root, self.emb)
        self.assertEqual(stats["files"], 0)
        self.assertEqual(stats["chunks"], 0)
        self.assertFalse(index_is_stale(self.root))
        self.assertEqual(Searcher(self.root).search(self.emb, "q", 5), [])

        # second empty build is a no-op too
        vec_ns = (self.index_dir / VECTORS_NAME).lstat().st_mtime_ns
        stats = build_index(self.root, self.emb)
        self.assertEqual(stats["changed_files"], 0)
        self.assertEqual((self.index_dir / VECTORS_NAME).lstat().st_mtime_ns, vec_ns)

        write(self.root, "a.py", src())
        self.assertTrue(index_is_stale(self.root))
        stats = build_index(self.root, self.emb)
        self.assertEqual(stats["files"], 1)
        self.assertFalse(index_is_stale(self.root))

    def test_dim_change_invalidates_all_vectors(self):
        write(self.root, "a.py", src())
        write(self.root, "b.py", src(70))
        build_index(self.root, self.emb)
        emb16 = FakeEmbedder(dim=16)
        stats = build_index(self.root, emb16)
        self.assertEqual(stats["reused_chunks"], 0)
        self.assertEqual(stats["reembedded_chunks"], stats["chunks"])
        self.assertEqual(vector_rows(self.index_dir).shape, (stats["chunks"], 16))
        self.assertEqual(load_meta(self.root)["dim"], 16)
        self.assertFalse(index_is_stale(self.root))
        hits = Searcher(self.root).search(emb16, "q", 5)
        self.assertEqual(len(hits), min(5, stats["chunks"]))

    def test_corrupt_meta_self_heals_without_crash(self):
        write(self.root, "a.py", src())
        build_index(self.root, self.emb)
        (self.index_dir / META_NAME).write_text("{oops")
        self.assertTrue(index_is_stale(self.root))
        stats = build_index(self.root, self.emb)
        self.assertEqual(stats["chunks"], len(manifest_rows(self.index_dir)))
        self.assertFalse(index_is_stale(self.root))

    def test_legacy_layout_without_files_json_self_heals(self):
        write(self.root, "a.py", src())
        write(self.root, "b.py", src(70))
        build_index(self.root, self.emb)
        (self.index_dir / FILES_NAME).unlink()
        self.assertFalse(index_is_stale(self.root))  # unchanged legacy tree is fresh
        os.utime(self.root / "b.py")
        self.assertTrue(index_is_stale(self.root))
        stats = build_index(self.root, self.emb)
        self.assertTrue((self.index_dir / FILES_NAME).exists())
        self.assertEqual(stats["changed_files"], 0)
        self.assertGreater(stats["reused_chunks"], 0)
        self.assertFalse(index_is_stale(self.root))

    # -- search / metadata helper --------------------------------------------

    def test_search_order_and_top_k_bounds(self):
        for name in ("a.py", "b.py", "c.py", "d.py"):
            write(self.root, name, src())
        build_index(self.root, self.emb)
        total = len(manifest_rows(self.index_dir))
        searcher = Searcher(self.root)

        hits = searcher.search(self.emb, "query", k=2)
        self.assertEqual(len(hits), 2)
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        all_hits = searcher.search(self.emb, "query", k=1000)
        self.assertEqual(len(all_hits), total)
        self.assertEqual(hits, all_hits[:2])

    def test_build_index_ensures_gitignored(self):
        write(self.root, "a.py", "x = 1\n")
        stats = build_index(self.root, self.emb)
        self.assertIn("gitignore", stats)
        self.assertEqual(stats["gitignore"], "added .csearch/ to .gitignore")
        gi_text = (self.root / ".gitignore").read_text()
        self.assertIn(".csearch/", gi_text)

        # No-op build returns 'already ignored'
        stats_noop = build_index(self.root, self.emb)
        self.assertEqual(stats_noop["gitignore"], "already ignored")

    def test_multimodal_image_and_pdf_indexing_and_search(self):
        from unittest.mock import patch
        from PIL import Image

        img_path = self.root / "docs" / "architecture_diagram.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.frombytes("RGB", (64, 64), os.urandom(64 * 64 * 3))
        img.save(img_path)
        self.assertGreaterEqual(img_path.stat().st_size, 512)

        pdf_path = self.root / "docs" / "rfc_spec.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 mock content")

        mock_pages = [
            (1, "OAuth2 Token Revocation Protocol\nSection 1: Overview\n"),
            (2, "State Machine Transitions\n"),
        ]

        with patch("csearch.indexer._extract_pdf_pages", return_value=mock_pages):
            # 1. Default (multimodal=False/None): ignores image and PDF files
            stats_default = build_index(self.root, self.emb)
            self.assertEqual(stats_default["files"], 0)
            self.assertEqual(stats_default["chunks"], 0)
            self.assertEqual(manifest_rows(self.index_dir), [])
            self.assertFalse(load_meta(self.root).get("multimodal", False))

            # 2. Opt-in: build_index(..., multimodal=True) indexes image and PDF
            stats = build_index(self.root, self.emb, multimodal=True)
            self.assertEqual(stats["files"], 2)
            self.assertEqual(stats["changed_files"], 2)
            self.assertTrue(load_meta(self.root).get("multimodal", False))

            rows = manifest_rows(self.index_dir)
            img_rows = [r for r in rows if r["path"] == "docs/architecture_diagram.png"]
            pdf_rows = [r for r in rows if r["path"] == "docs/rfc_spec.pdf"]

            self.assertEqual(len(img_rows), 1)
            self.assertEqual(img_rows[0].get("kind"), "image")

            self.assertEqual(len(pdf_rows), 2)
            self.assertTrue(all(r.get("kind") == "pdf" for r in pdf_rows))
            self.assertEqual([r["start"] + 1 for r in pdf_rows], [1, 2])

            searcher = Searcher(self.root)

            hits_pdf = searcher.search(self.emb, "OAuth2 Token", k=5)
            self.assertTrue(hits_pdf)
            top_pdf = hits_pdf[0]
            self.assertEqual(top_pdf["path"], "docs/rfc_spec.pdf")
            self.assertEqual(top_pdf.get("kind"), "pdf")
            self.assertTrue(top_pdf.get("preview", "").startswith("[pdf p.1]"))
            self.assertTrue(first_line(self.root, top_pdf["path"], top_pdf["start"]).startswith("[pdf p.1]"))

            hits_img = searcher.search(self.emb, "architecture_diagram.png", k=5)
            self.assertTrue(hits_img)
            top_img = hits_img[0]
            self.assertEqual(top_img["path"], "docs/architecture_diagram.png")
            self.assertEqual(top_img.get("kind"), "image")
            self.assertTrue(top_img.get("preview", "").startswith("[image"))
            self.assertTrue(first_line(self.root, top_img["path"], top_img["start"]).startswith("[image"))

            # 3. Subsequent build_index(..., multimodal=None) preserves multimodal=True from meta.json
            stats2 = build_index(self.root, self.emb)
            self.assertEqual(stats2["changed_files"], 0)
            self.assertEqual(stats2["reused_chunks"], stats["chunks"])
            self.assertTrue(load_meta(self.root).get("multimodal", False))


if __name__ == "__main__":
    unittest.main()
