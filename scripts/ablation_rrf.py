#!/usr/bin/env python
"""RRF ablation on the csearch index: vector leg vs BM25 leg vs fused.

Reproduces code-context-mcp's hybrid pipeline (vector + BM25/FTS5 fused via
Reciprocal Rank Fusion, over-fetch pool = top_k*3, RRF k=60, dedupe by
(path, start, end)) on top of the local csearch index, so the ablation shares
the exact EmbeddingGemma vectors of BENCHMARK.md test 2.

Arms (each returns top-k after the same fusion/truncation path):
  vec       cosine vector leg only (BM25 removed)
  kw-fts    BM25 leg only, pip-faithful unicode61 tokenizer + query sanitiser
            (contiguous CJK runs collapse to single tokens)
  kw-cjk    BM25 leg only, CJK char-bigram tokenizer with OR query semantics
  full-fts  RRF(vec, kw-fts)
  full-cjk  RRF(vec, kw-cjk)

Ground truth: the 8 pinned benchmark questions of BENCHMARK.md test 2.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from csearch.embedder import DEFAULT_MODEL_DIR, Embedder
from csearch.indexer import MANIFEST_NAME, Searcher, index_dir_for, load_meta

REPO = Path(os.environ.get("CSEARCH_BENCH_REPO", "reimbursement-back-end"))
MODEL_DIR = DEFAULT_MODEL_DIR

TOP_K = 5
POOL = TOP_K * 3  # pip _OVER_FETCH_MULTIPLIER
RRF_K = 60  # pip _RRF_K

# --- BENCHMARK.md test 2: pinned queries + canonical files -------------------

QUERIES: list[tuple[str, tuple[str, ...]]] = [
    ("报销单从提交到打款要经过哪些审核状态?状态流转在哪里统一控制?",
     ("OrderStateMachine.java", "OrderStates.java")),
    ("可报销金额是怎么按政策算出来的?规则从哪里加载、能否热更新?",
     ("PolicyEngineServiceImpl.java",)),
    ("学生每月报销配额是如何占用的?审核中/已报销金额怎么区分?",
     ("QuotaServiceImpl.java", "QuotaLedgerShadow.java")),
    ("电子签章是怎么上传和管理的?谁可以带有签名?",
     ("EsignServiceImpl.java",)),
    ("发票图片的 OCR 是怎么接入的?Java 侧调用了哪些接口、如何容错?",
     ("OcrdClient.java", "OcrPipelineService.java")),
    ("登录后的 token 是什么形式?如何校验和吊销?是 JWT 吗?",
     ("SessionTokenServiceImpl.java",)),
    ("管理端操作日志是怎么记录的?会记录哪些信息?敏感字段会脱敏吗?",
     ("LogAspect.java",)),
    ("批量审批的进度是怎么反馈给前端的?会主动推送吗?",
     ("BatchJobVTO.java",)),
]

# --- pip query sanitisation (keyword_index_sqlite.py) -------------------------

_FTS_KEEP_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_FTS_BOOLEAN_RE = re.compile(r"\b(AND|OR|NOT|NEAR)\b", re.IGNORECASE)
_STOP_WORDS = frozenset("""
a an the are was were be been being have has had do does did of on at to by
into about between how what where when who which why i we you he she they it
this that these those but so yet also can will would should could may might
""".split())

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def _sanitise(query: str) -> str:
    """pip _sanitise: bare whitespace-joined tokens, no FTS5 syntax."""
    q = _FTS_KEEP_RE.sub(" ", query)
    q = _FTS_BOOLEAN_RE.sub(" ", q)
    q = " ".join(w for w in q.split() if w.lower() not in _STOP_WORDS)
    return q.strip()


def _tokens_with_cjk_bigrams(text: str) -> list[str]:
    """Identifiers split on case/digits + CJK char bigrams (BM25-viable CJK)."""
    out: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff\u3400-\u4dbf]+", text):
        if _CJK_RE.match(raw):
            if len(raw) == 1:
                out.append(raw)
            else:
                out.extend(raw[i:i + 2] for i in range(len(raw) - 1))
        elif any(c.isupper() for c in raw[1:]):
            out.extend(t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", raw))
        else:
            out.append(raw.lower())
    return out


def _fts_or_query(tokens: list[str]) -> str:
    """OR semantics — bigram AND would demand the exact phrase."""
    seen: dict[str, None] = {}
    for t in tokens:
        seen.setdefault(t.replace('"', '""'), None)
    return " OR ".join(f'"{t}"' for t in seen)


# --- index builders -----------------------------------------------------------

def _chunk_text(root: Path, rel: str, start: int, end: int) -> str:
    try:
        data = (root / rel).read_bytes()
        lines = data.decode("utf-8", errors="replace").splitlines()
        return "\n".join(lines[start:end])
    except OSError:
        return ""


class KwIndex:
    """FTS5 BM25 over csearch manifest chunks; unicode61 (pip-faithful) or
    pre-tokenized CJK-bigram mode."""

    def __init__(self, mode: str, records: list[dict], root: Path):
        self.mode = mode
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE VIRTUAL TABLE f USING fts5(path UNINDEXED, line_start UNINDEXED, "
            "line_end UNINDEXED, body, tokenize='unicode61 remove_diacritics 2')"
        )
        rows = []
        for r in records:
            body = _chunk_text(root, r["path"], r["start"], r["end"])
            if mode == "cjk":
                body = " ".join(_tokens_with_cjk_bigrams(body))
            rows.append((r["path"], r["start"], r["end"], body))
        self.conn.executemany("INSERT INTO f VALUES (?,?,?,?)", rows)
        self.conn.commit()

    def search(self, query: str, k: int) -> list[tuple[str, int, int]]:
        if self.mode == "fts":
            q = _sanitise(query)
            if not q:
                return []
        else:
            q = _fts_or_query(_tokens_with_cjk_bigrams(_sanitise(query)))
            if not q:
                return []
        try:
            cur = self.conn.execute(
                "SELECT path, line_start, line_end, bm25(f) AS score FROM f "
                "WHERE f MATCH ? ORDER BY score LIMIT ?", (q, k))
        except sqlite3.OperationalError:
            return []
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]


def _rrf_fuse(a: list, b: list, k_constant: int = RRF_K) -> list:
    """pip _rrf_fuse keyed on (path, start, end)."""
    scores: dict[tuple, float] = {}
    for hits in (a, b):
        for rank, key in enumerate(hits):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_constant + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


# --- arms ----------------------------------------------------------------------

class Arms:
    def __init__(self, searcher: Searcher, embedder: Embedder, root: Path):
        self.records = searcher.records
        self.vectors = np.asarray(searcher.vectors)
        self.embedder = embedder
        self.root = root
        self.kw_fts = KwIndex("fts", self.records, root)
        self.kw_cjk = KwIndex("cjk", self.records, root)

    def vec(self, qvec: np.ndarray, k: int) -> list:
        sims = self.vectors @ qvec
        top = np.argpartition(sims, len(sims) - k)[-k:]
        top = top[np.argsort(-sims[top])]
        return [(self.records[int(i)]["path"], self.records[int(i)]["start"],
                 self.records[int(i)]["end"]) for i in top]

    def run(self, qvec: np.ndarray, query: str, arm: str) -> list:
        if arm == "vec":
            fused = self.vec(qvec, POOL)
        elif arm == "kw-fts":
            fused = self.kw_fts.search(query, POOL)
        elif arm == "kw-cjk":
            fused = self.kw_cjk.search(query, POOL)
        elif arm == "full-fts":
            fused = _rrf_fuse(self.vec(qvec, POOL), self.kw_fts.search(query, POOL))
        elif arm == "full-cjk":
            fused = _rrf_fuse(self.vec(qvec, POOL), self.kw_cjk.search(query, POOL))
        else:
            raise ValueError(arm)
        return fused[:TOP_K]


def evaluate(arms: Arms, qvecs: np.ndarray) -> dict:
    names = ["vec", "kw-fts", "kw-cjk", "full-fts", "full-cjk"]
    hits = {n: 0 for n in names}
    mrr = {n: 0.0 for n in names}
    detail: dict[str, list] = {n: [] for n in names}
    for qi, (query, canon) in enumerate(QUERIES):
        canon_set = set(canon)
        per_arm_rank: dict[str, int | None] = {}
        for n in names:
            top = arms.run(qvecs[qi], query, n)
            rank = next((i + 1 for i, (p, _, _) in enumerate(top)
                         if Path(p).name in canon_set), None)
            per_arm_rank[n] = rank
            if rank:
                hits[n] += 1
                mrr[n] += 1.0 / rank
            detail[n].append((query, top, rank))
    return {"names": names, "hits": hits, "mrr": mrr, "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--json-out", type=Path, default=None)
    ns = ap.parse_args()

    if not ns.repo.is_dir():
        print(f"benchmark repo not found: {ns.repo} (pass --repo <path> or set CSEARCH_BENCH_REPO; for the public benchmark run scripts/bench_arb_trace2code.py)", file=sys.stderr)
        return 2

    t0 = time.time()
    meta = load_meta(ns.repo, index_dir_for(ns.repo))
    searcher = Searcher(ns.repo)
    print(f"index: {meta['files']} files / {meta['chunks']} chunks, dim={meta['dim']}, "
          f"updated={meta['updated']}", file=sys.stderr)
    embedder = Embedder(MODEL_DIR, dim=meta["dim"])
    arms = Arms(searcher, embedder, ns.repo)
    print(f"arms ready in {time.time() - t0:.1f}s (fts + cjk FTS5 built)", file=sys.stderr)

    qtexts = [q for q, _ in QUERIES]
    t0 = time.time()
    qvecs = embedder.encode_queries(qtexts)
    print(f"encoded {len(qtexts)} queries in {time.time() - t0:.2f}s", file=sys.stderr)

    res = evaluate(arms, qvecs)

    print(f"\n{'arm':<9} {'hit@5':>6} {'MRR@5':>7}")
    for n in res["names"]:
        print(f"{n:<9} {res['hits'][n]:>4}/8 {res['mrr'][n]:>7.3f}")

    print("\nper-query best canonical rank (chunk-level, top-5):")
    print(f"{'query':<28}" + "".join(f"{n:>9}" for n in res["names"]))
    for qi, (query, _) in enumerate(QUERIES):
        cells = []
        for n in res["names"]:
            r = res["detail"][n][qi][2]
            cells.append(f"{r!s:>9}" if r else f"{'-':>9}")
        print(f"{query[:26]:<28}" + "".join(cells))

    # FULL vs VEC top-5 overlap — the direct "BM25 影响了吗" readout
    print("\nfull vs vec top-5 difference (chunks added / dropped by adding BM25):")
    for qi, (query, _) in enumerate(QUERIES):
        v = set(res["detail"]["vec"][qi][1])
        for n in ("full-fts", "full-cjk"):
            f = res["detail"][n][qi][1]
            added = [x for x in f if x not in v]
            dropped = [x for x in v if x not in set(f)]
            print(f"  Q{qi + 1} {n}: +{len(added)} -{len(dropped)}  "
                  f"vec_rank={res['detail']['vec'][qi][2]} new_rank={res['detail'][n][qi][2]}")

    if ns.json_out:
        out = {
            "meta": {"repo": str(ns.repo), "chunks": meta["chunks"], "dim": meta["dim"]},
            "arms": {n: {"hit@5": res["hits"][n], "mrr@5": round(res["mrr"][n], 4)}
                     for n in res["names"]},
            "detail": {n: [{"query": q, "rank": r,
                            "top5": [f"{p}:{s + 1}-{e}" for p, s, e in t]}
                           for q, t, r in res["detail"][n]] for n in res["names"]},
        }
        ns.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nwrote {ns.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
