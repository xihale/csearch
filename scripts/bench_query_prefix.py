#!/usr/bin/env python
"""Query-prefix ablation for the csearch vector arm on ARB trace2code (click).

EmbeddingGemma-2 is trained with task-conditioned prompts: csearch encodes
queries as QUERY_PROMPT + text ("task: code retrieval | query: ") and docs as
"title: <path> | text: <chunk>" (embedder.doc_text). This script measures what
the query-side prefix buys on the pallets/click subset (26 samples, 71,096
chunks): same doc vectors, two query encodings — with prefix (deployed
behaviour) vs raw text (ARB embedding protocol).

  PYTHONPATH=. ~/.venvs/code-context/bin/python scripts/bench_query_prefix.py
"""

from __future__ import annotations

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

CORPUS = ROOT / "bench_arb_t2c" / "corpus" / "v2_trace2code"
SAMPLES = ROOT / "bench_arb_t2c" / "benchmark" / "v2_trace2code" / "trace2code.jsonl"


def metrics_for(scores: np.ndarray, chunks: list[dict], gold: set[str]) -> dict:
    keyed = sorted(
        ((float(s), str(c["path"]), str(c["chunk_id"]), c) for s, c in zip(scores, chunks)),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    paths = arb.unique_ranked_paths([item[3] for item in keyed])
    return {
        "r5": arb.recall_at(gold, paths, 5),
        "r10": arb.recall_at(gold, paths, 10),
        "r20": arb.recall_at(gold, paths, 20),
        "mrr": arb.reciprocal_rank(gold, paths),
    }


def main() -> int:
    samples = [
        json.loads(line)
        for line in SAMPLES.read_text().splitlines()
        if line.strip() and json.loads(line)["repo"] == "pallets/click"
    ]
    by_snapshot: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        by_snapshot[(s["repo"], s["base_commit"])].append(s)

    emb = Embedder(DEFAULT_MODEL_DIR)
    print(f"embedder: {emb.info}", file=sys.stderr)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for key, group in by_snapshot.items():
        repo, base = key
        path = CORPUS / repo.replace("/", "__") / f"{base}.chunks.jsonl"
        chunks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        doc_vecs = emb.encode_docs([str(c["path"]) for c in chunks], [str(c["text"]) for c in chunks])
        for s in group:
            qt = arb.query_text_for_eval(s)
            gold = set(arb.target_gold_files(s))
            q_pref = emb.encode_queries([qt])[0]
            q_raw = emb._encode([qt], is_query=False)[0]
            rows.append(
                {
                    "id": s["id"],
                    "arms": {
                        "prefix": metrics_for(doc_vecs @ q_pref, chunks, gold),
                        "raw": metrics_for(doc_vecs @ q_raw, chunks, gold),
                    },
                }
            )
        print(f"  {repo}@{base[:8]} done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)

    print(f"\n== query-prefix ablation, pallets/click n={len(rows)} ==")
    print(f"{'arm':9s} " + " ".join(f"{m:>8s}" for m in ("R@5", "R@10", "R@20", "MRR")))
    for arm in ("prefix", "raw"):
        vals = {
            "R@5": float(np.mean([r["arms"][arm]["r5"] for r in rows])),
            "R@10": float(np.mean([r["arms"][arm]["r10"] for r in rows])),
            "R@20": float(np.mean([r["arms"][arm]["r20"] for r in rows])),
            "MRR": float(np.mean([r["arms"][arm]["mrr"] for r in rows])),
        }
        print(f"{arm:9s} " + " ".join(f"{vals[m]:8.4f}" for m in ("R@5", "R@10", "R@20", "MRR")))

    flips = [
        (r["id"], r["arms"]["prefix"]["mrr"], r["arms"]["raw"]["mrr"])
        for r in rows
        if abs(r["arms"]["prefix"]["mrr"] - r["arms"]["raw"]["mrr"]) > 1e-9
    ]
    print(f"\nsamples with different MRR: {len(flips)}/{len(rows)}")
    for sid, mp, mr in flips:
        print(f"  {sid}: prefix={mp:.3f} raw={mr:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
