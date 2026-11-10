#!/usr/bin/env python
"""Query-routing + dual-channel retrieval ablation on the rb repo (8 PM questions).

Arms (identical ground truth to ablation_rrf.py / BENCHMARK.md test 2):
  vec-title   deployed arm: chunk encoded as "title: <path> | text: <code>",
              query = QUERY_PROMPT + question (title|text asymmetric format)
  vec-raw     ablation arm: chunk encoded as raw code, query as raw question
  lex         lexical leg: TF-IDF bigram chunk scoring over the manifest texts
  route       heuristic router: contains identifiers/paths/digits -> lex; else vec-title
  rrf-tl      rank-fuse(vec-title, lex) at chunk level, dedupe to files
  rrff-tl     weighted fusion: 2*vec-title + 1*lex score-level (min-max normalised)

File-level hit@5 / MRR@5, matching ablation_rrf.py metric definitions.

  PYTHONPATH=scripts:scripts ~/.venvs/code-context/bin/python scripts/bench_route_rb.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_rrf import MODEL_DIR, QUERIES, REPO, _chunk_text  # noqa: E402

from csearch.embedder import Embedder  # noqa: E402
from csearch.indexer import Searcher, index_dir_for  # noqa: E402

ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*[A-Z_a-z]\w*[._]|\b\w+\.\w{1,4}\b|[A-Z][a-z]+[A-Z]\w*|\d+")


def looks_literal(q: str) -> bool:
    """Heuristic: identifiers, dotted paths, CamelCase, digits -> lexical leg."""
    return bool(ID_RE.search(q))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]", text.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--json-out", type=Path, default=None)
    ns = ap.parse_args()

    repo = ns.repo
    if not repo.is_dir():
        print(f"benchmark repo not found: {repo} (pass --repo <path> or set CSEARCH_BENCH_REPO; for the public benchmark run scripts/bench_arb_trace2code.py)", file=sys.stderr)
        return 2

    searcher = Searcher(repo, index_dir_for(repo))
    recs = searcher.records
    texts = [_chunk_text(repo, r["path"], r["start"], r["end"]) for r in recs]
    emb = Embedder(MODEL_DIR)

    # --- doc vectors (both doc encodings), computed once -----------------------
    t0 = time.perf_counter()
    v_title = searcher.vectors  # deployed: title|text encoding
    v_raw = emb._encode(texts, is_query=False)
    print(f"doc embed: {time.perf_counter() - t0:.0f}s ({len(recs)} chunks)", file=sys.stderr)

    # --- lexical TF-IDF over bigram tokens -------------------------------------
    t0 = time.perf_counter()
    docs = [tokenize(t) for t in texts]
    df: dict[str, int] = {}
    for d in docs:
        for tok in set(d):
            df[tok] = df.get(tok, 0) + 1
    n = len(docs)
    idf = {t: np.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
    norms = np.array([np.sqrt(sum((idf[t] ** 2) * d.count(t) for t in set(d))) or 1.0 for d in docs])
    print(f"lex index: {time.perf_counter() - t0:.0f}s", file=sys.stderr)

    def lex_scores(q: str) -> np.ndarray:
        qt = tokenize(q)
        tf: dict[str, int] = {}
        for t in qt:
            tf[t] = tf.get(t, 0) + 1
        sc = np.zeros(n)
        for t, f in tf.items():
            if t in idf:
                sc += idf[t] * f * (np.array([d.count(t) for d in docs]) / norms)
        return sc

    def rrf_fuse(a: np.ndarray, b: np.ndarray, k: int = 60) -> np.ndarray:
        ra, rb = np.empty(len(a)), np.empty(len(b))
        ra[np.argsort(-a)] = np.arange(1, len(a) + 1)
        rb[np.argsort(-b)] = np.arange(1, len(b) + 1)
        return 1 / (k + ra) + 1 / (k + rb)

    def top_files(scores: np.ndarray, k: int = 5) -> list[str]:
        idx = np.argsort(-scores)[: 15]
        seen: dict[str, None] = {}
        for i in idx:
            seen.setdefault(recs[i]["path"], None)
            if len(seen) == k:
                break
        return list(seen)

    arms = {
        "vec-title": lambda q: top_files(v_title @ emb.encode_queries([q])[0]),
        "vec-raw": lambda q: top_files(v_raw @ emb._encode([q], is_query=False)[0]),
        "lex": lambda q: top_files(lex_scores(q)),
        "route": lambda q: (
            top_files(lex_scores(q)) if looks_literal(q) else top_files(v_title @ emb.encode_queries([q])[0])
        ),
        "rrf-tl": lambda q: top_files(
            rrf_fuse(v_title @ emb.encode_queries([q])[0], lex_scores(q))
        ),
        "rrff-tl": lambda q: top_files(
            2 * _mm(v_title @ emb.encode_queries([q])[0]) + 1 * _mm(lex_scores(q))
        ),
    }

    def _mm(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else x

    results: dict[str, list[bool]] = {a: [] for a in arms}
    mrrs: dict[str, list[float]] = {a: [] for a in arms}
    for query, golds in QUERIES:
        gold = set(golds)
        for name, fn in arms.items():
            files = fn(query)
            hit = any(any(g in f for f in files) for g in gold)
            results[name].append(hit)
            rank = next((i for i, f in enumerate(files, 1) if any(g in f for g in gold)), 0)
            mrrs[name].append(1.0 / rank if rank else 0.0)

    print(f"\n== rb repo, {len(QUERIES)} queries, {n} chunks ==")
    print(f"{'arm':10s} {'hit@5':>6s} {'MRR@5':>7s}")
    for name in arms:
        print(f"{name:10s} {sum(results[name]):>4d}/8 {np.mean(mrrs[name]):>7.3f}")

    if ns.json_out:
        out = {
            "meta": {"repo": str(repo), "chunks": n},
            "arms": {name: {"hit@5": sum(results[name]), "mrr@5": round(float(np.mean(mrrs[name])), 4)}
                     for name in arms},
        }
        ns.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"wrote {ns.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
