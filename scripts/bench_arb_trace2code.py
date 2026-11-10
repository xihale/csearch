#!/usr/bin/env python
"""Agent Retrieval Bench (ARB) trace2code v2 — local rerun with four arms.

Corpus: bench_arb_t2c/corpus/v2_trace2code/<repo_slug>/<base_commit>.chunks.jsonl
(release tar.zst from huggingface.co/datasets/eyuansu71/agent_retrieval_bench).
Samples: bench_arb_t2c/benchmark/v2_trace2code/trace2code.jsonl (101 samples).
Protocol: query = json.dumps(sample["query"], sort_keys=True)  (ARB
query_text_for_eval); gold = gold.root_cause_files; file ranking = first
occurrence in chunk-ranked order (unique_ranked_paths); metrics = Recall@5/10/20
(gold-set fraction) + MRR — identical formulas to ARB baseline.py.

Arms:
  lexical  ARB rank_chunks_with_scores (verbatim vendored)
  bm25     ARB rank_chunks_bm25_with_scores (verbatim vendored)
  vec      csearch EmbeddingGemma-2 (ort q4f16 CUDA): docs = "title: <path> |
           text: <chunk>", query = QUERY_PROMPT + query text, cosine, ties
           broken by (-score, path, chunk_id)
  rrf      reciprocal rank fusion of the vec and bm25 chunk orders (k=60),
           same tie-break, then path dedupe

Usage:
  PYTHONPATH=. ~/.venvs/code-context/bin/python scripts/bench_arb_trace2code.py \
      [--repos pallets/click,...] [--limit N] [--json-out PATH] [--batch 16]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import vendor_arb_baseline as arb  # noqa: E402
from csearch.embedder import DEFAULT_MODEL_DIR, Embedder  # noqa: E402

RRF_K = 60
CORPUS = ROOT / "bench_arb_t2c" / "corpus" / "v2_trace2code"
SAMPLES = ROOT / "bench_arb_t2c" / "benchmark" / "v2_trace2code" / "trace2code.jsonl"
ARMS = ("lexical", "bm25", "vec", "rrf")


def load_samples(repos: list[str] | None, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in SAMPLES.read_text().splitlines() if line.strip()]
    if repos:
        rows = [r for r in rows if r["repo"] in repos]
    if limit:
        rows = rows[:limit]
    return rows


def chunks_path(repo: str, base_commit: str) -> Path:
    return CORPUS / repo.replace("/", "__") / f"{base_commit}.chunks.jsonl"


def ordered(ranked: list[tuple[float, dict]]) -> list[dict]:
    return [chunk for _, chunk in ranked]


def ordered_vec(scores: np.ndarray, chunks: list[dict]) -> list[dict]:
    keyed = sorted(
        ((float(s), str(c["path"]), str(c["chunk_id"]), c) for s, c in zip(scores, chunks)),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return [item[3] for item in keyed]


def rrf_order(vec_order: list[dict], bm25_order: list[dict]) -> list[dict]:
    fused: dict[str, float] = {}
    for rank, chunk in enumerate(vec_order):
        fused[str(chunk["chunk_id"])] = fused.get(str(chunk["chunk_id"]), 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, chunk in enumerate(bm25_order):
        fused[str(chunk["chunk_id"])] = fused.get(str(chunk["chunk_id"]), 0.0) + 1.0 / (RRF_K + rank + 1)
    keyed = sorted(
        (
            (fused[str(c["chunk_id"])], str(c["path"]), str(c["chunk_id"]), c)
            for c in vec_order  # vec_order already contains every chunk of the snapshot
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return [item[3] for item in keyed]


def metrics_for(ranked_chunks: list[dict], gold: set[str]) -> dict:
    paths = arb.unique_ranked_paths(ranked_chunks)
    return {
        "r5": arb.recall_at(gold, paths, 5),
        "r10": arb.recall_at(gold, paths, 10),
        "r20": arb.recall_at(gold, paths, 20),
        "mrr": arb.reciprocal_rank(gold, paths),
        "first_gold": next((i for i, p in enumerate(paths, 1) if p in gold), None),
        "top5": paths[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    repos = [r for r in args.repos.split(",") if r] or None
    samples = load_samples(repos, args.limit or None)
    if not samples:
        print("no samples", file=sys.stderr)
        return 1

    by_snapshot: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        by_snapshot[(s["repo"], s["base_commit"])].append(s)

    embedder = Embedder(DEFAULT_MODEL_DIR, dim=args.dim, batch_size=args.batch)
    print(f"embedder: {embedder.info} dim={embedder.dim}", file=sys.stderr)

    details: list[dict] = []
    embed_s = 0.0
    for n_snap, (key, group) in enumerate(by_snapshot.items(), 1):
        repo, base = key
        path = chunks_path(repo, base)
        if not path.exists():
            print(f"MISSING corpus {path}", file=sys.stderr)
            return 1
        chunks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        t0 = time.perf_counter()
        doc_vecs = embedder.encode_docs(
            [str(c["path"]) for c in chunks],
            [str(c["text"]) for c in chunks],
            progress=lambda d, n: print(f"\r  embed {repo}@{base[:8]} {d}/{n}", end="", file=sys.stderr, flush=True)
            if d % 20000 == 0 or d == n
            else None,
        )
        embed_s += time.perf_counter() - t0
        qtexts = [arb.query_text_for_eval(s) for s in group]
        qvecs = embedder.encode_queries(qtexts)

        for s, qv in zip(group, qvecs):
            gold = set(arb.target_gold_files(s))
            t0 = time.perf_counter()
            lex = ordered(arb.rank_chunks_with_scores(arb.query_text_for_eval(s), chunks))
            bm = ordered(arb.rank_chunks_bm25_with_scores(arb.query_text_for_eval(s), chunks))
            lex_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            vec = ordered_vec(doc_vecs @ qv, chunks)
            vec_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            fus = rrf_order(vec, bm)
            fus_ms = (time.perf_counter() - t0) * 1000

            row = {
                "id": s["id"],
                "repo": repo,
                "base_commit": base,
                "gold": sorted(gold),
                "n_chunks": len(chunks),
                "arms": {
                    "lexical": metrics_for(lex, gold),
                    "bm25": metrics_for(bm, gold),
                    "vec": metrics_for(vec, gold),
                    "rrf": metrics_for(fus, gold),
                },
                "ms": {"lex+bm25": round(lex_ms, 1), "vec": round(vec_ms, 2), "rrf": round(fus_ms, 2)},
            }
            details.append(row)
            hits = "".join("H" if row["arms"][a]["first_gold"] else "." for a in ARMS)
            print(
                f"\r[{n_snap}/{len(by_snapshot)}] {repo.split('/')[1]}@{base[:8]} "
                f"{hits} lex_ms={row['ms']['lex+bm25']:.0f} vec_ms={row['ms']['vec']:.0f}",
                file=sys.stderr,
            )
        del doc_vecs

    # --- aggregate ---------------------------------------------------------
    def agg(rows: list[dict]) -> dict:
        out: dict = {"n_samples": len(rows)}
        for arm in ARMS:
            out[arm] = {
                f"R@{k}": float(np.mean([r["arms"][arm][f"r{k}"] for r in rows])) for k in (5, 10, 20)
            }
            out[arm]["MRR"] = float(np.mean([r["arms"][arm]["mrr"] for r in rows]))
        return out

    summary = {
        "embedder": embedder.info,
        "dim": embedder.dim,
        "n_samples": len(details),
        "embed_total_s": round(embed_s, 1),
        "overall": agg(details),
        "pallets_click": agg([r for r in details if r["repo"] == "pallets/click"]),
        "per_repo": {repo: agg([r for r in details if r["repo"] == repo]) for repo in sorted({r["repo"] for r in details})},
    }

    def show(title: str, arms: dict, n: int, embed_s: float) -> None:
        print(f"\n== {title} (n={n}, embed {embed_s}s) ==")
        hdr = f"{'arm':9s} " + " ".join(f"{m:>8s}" for m in ("R@5", "R@10", "R@20", "MRR"))
        print(hdr)
        for arm in ARMS:
            vals = arms[arm]
            print(
                f"{arm:9s} "
                + " ".join(f"{vals[m]:8.4f}" for m in ("R@5", "R@10", "R@20", "MRR"))
            )

    show("OVERALL trace2code", summary["overall"], summary["n_samples"], summary["embed_total_s"])
    click_rows = [r for r in details if r["repo"] == "pallets/click"]
    if click_rows:
        show("pallets/click subset", summary["pallets_click"], len(click_rows), summary["embed_total_s"])
    for repo, s in summary["per_repo"].items():
        print(f"\n-- {repo} (n={s['n_samples']}) --")
        for arm in ARMS:
            vals = s[arm]
            print(f"  {arm:9s} " + " ".join(f"{vals[m]:.4f}" for m in ("R@5", "R@10", "R@20", "MRR")))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summary": summary, "details": details}, indent=1))
        print(f"\ndetails -> {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
