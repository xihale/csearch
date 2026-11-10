import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from csearch import mcp_server as server
from csearch.embedder import Embedder
from csearch.indexer import build_index, load_meta


class WordEmbedder(Embedder):
    backend = "ort"
    model_dir = Path("/test-model")

    def __init__(self, dim=128):
        self.dim = dim

    def encode_docs(self, titles, texts, progress=None, dim=None):
        return self.encode_queries(texts, dim=dim)

    def encode_queries(self, texts, dim=None):
        target_dim = dim or self.dim
        vectors = np.zeros((len(texts), target_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, 0 if "alpha" in text else 1] = 1
        return vectors


class ResidentSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.embedder = WordEmbedder()
        for attribute, value in (
            ("_embedder", self.embedder),
            ("_searchers", OrderedDict()),
        ):
            p = patch.object(server, attribute, value)
            p.start()
            self.addCleanup(p.stop)

    def make_repo(self, name, dim=128):
        root = self.root / name
        root.mkdir()
        (root / "a.py").write_text("alpha\n")
        (root / "b.py").write_text("beta\n")
        build_index(root, WordEmbedder(dim))
        return root

    def test_external_rebuild_updates_cached_hits(self):
        root = self.make_repo("repo")
        self.assertIn("a.py:", server._search_sync(root, "alpha", 1, False))
        (root / "a.py").write_text("beta\n")
        (root / "b.py").write_text("alpha\n")
        build_index(root, WordEmbedder())
        self.assertIn("b.py:", server._search_sync(root, "alpha", 1, False))

    def test_vector_replacement_without_metadata_change_updates_hits(self):
        root = self.make_repo("repo")
        self.assertIn("a.py:", server._search_sync(root, "alpha", 1, False))
        index = root / ".csearch"
        replacement = index / "replacement.npy"
        vectors = np.load(index / "vectors.npy")[::-1].copy()
        np.save(replacement, vectors)
        replacement.replace(index / "vectors.npy")
        self.assertIn("b.py:", server._search_sync(root, "alpha", 1, False))

    def test_search_keeps_each_repository_dimension(self):
        small = self.make_repo("small", 128)
        large = self.make_repo("large", 768)
        before = {root: load_meta(root) for root in (small, large)}
        for root in (small, large, small):
            self.assertIn("a.py:", server._search_sync(root, "alpha", 1, False))
            self.assertEqual(load_meta(root), before[root])

    def test_literal_query_refreshes_lexical_index_after_file_changes(self):
        root = self.root / "literal_repo"
        root.mkdir()
        (root / "a.py").write_text("class FooBar:\n    pass\n")
        build_index(root, WordEmbedder())

        res = server._search_sync(root, "FooBar", 5, False)
        self.assertIn("a.py:", res)

        (root / "b.py").write_text("class FooBarHelper:\n    pass\n")
        res2 = server._search_sync(root, "FooBar", 5, True)
        self.assertIn("a.py:", res2)
        self.assertIn("b.py:", res2)

    def test_auto_index_creates_gitignore(self):
        root = self.root / "auto_repo"
        root.mkdir()
        (root / "a.py").write_text("alpha\n")
        self.assertFalse((root / ".gitignore").exists())
        server._search_sync(root, "alpha", 1, False)
        self.assertTrue((root / ".gitignore").exists())
        self.assertIn(".csearch/", (root / ".gitignore").read_text())

    def test_http_origin_and_host_check(self):
        import json

        class DummyRequest:
            def __init__(self, headers):
                self.headers = headers

        for host in ("127.0.0.1:8765", "localhost:8765", "[::1]:8765", "127.0.0.1", "localhost", "::1", "[::1]"):
            req = DummyRequest({"host": host})
            self.assertIsNone(server._check_http_origin(req))

        req = DummyRequest({"host": "evil.com"})
        resp = server._check_http_origin(req)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(json.loads(resp.body.decode()), {"error": "forbidden host"})

        req = DummyRequest({"host": "localhost:8765", "origin": "http://evil.com"})
        resp = server._check_http_origin(req)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(json.loads(resp.body.decode()), {"error": "cross-origin requests forbidden"})


if __name__ == "__main__":
    unittest.main()
