#!/usr/bin/env python
"""纯 grep vs cs(向量) vs 混合(RRF) 三臂对比：文件级 hit@5 / MRR@5 + 每题延迟。

臂定义（同一语料 = csearch 索引覆盖的 rb 仓库）：
  grep   机械抽取题面词项（中文 2-gram / 英文标识符及其 CamelCase 部分 / 数字串），
         逐个 `rg -F -i -c` 统计文件级命中次数，IDF 加权求和取 top-5 文件。
         只用题面文本，不接触 ground truth；延迟含 rg 进程 spawn。
  cs     csearch 纯向量检索：热模型逐题 embed + 矩阵乘，chunk top-15 去重成文件取 top-5。
  hybrid RRF(vec pool, BM25(CJK bigram) pool) —— 消融测试中唯一有效的 BM25 配置，
         融合后去重成文件取 top-5。

Ground truth 与 BENCHMARK.md 测试 2 / scripts/ablation_rrf.py 相同（8 题钉死）。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_rrf import MODEL_DIR, POOL, QUERIES, REPO, KwIndex, _rrf_fuse, _sanitise

from csearch.embedder import Embedder
from csearch.indexer import Searcher, index_dir_for, load_meta

TOP_K = 5

_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]{2,}")
_LATIN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL = re.compile(r"[A-Z]?[a-z]{2,}|[A-Z]{2,}")
_DIGITS = re.compile(r"\d{2,}")


def extract_patterns(query: str) -> list[str]:
    """题面 → 去重词项表。中文按 2-gram 滑窗（子串 grep 的可检索单元），
    英文按标识符整体 + CamelCase 子词（去噪：长度<3 丢弃）。"""
    pats: set[str] = set()
    for run in _CJK_RUN.findall(query):
        pats.update(run[i:i + 2] for i in range(len(run) - 1))
    for tok in _LATIN.findall(query):
        pats.add(tok.lower())
        pats.update(m.lower() for m in _CAMEL.findall(tok))
    pats.update(_DIGITS.findall(query))
    pats.discard("")
    return sorted(pats)


class GrepArm:
    """逐词项 `rg -F -i -c` 收集文件级计数，IDF 加权评分取 top-5。
    延迟含 rg 进程 spawn，不含语料枚举（预扫描一次）。"""

    def __init__(self, repo: Path):
        self.repo = repo
        self._rg = shutil.which("rg")
        if not self._rg:
            raise SystemExit("ripgrep (rg) 不在 PATH 中，grep 臂无法运行")
        proc = subprocess.run(
            [self._rg, "--files", "--hidden", "-g", "!.git", "-g", "!.git/**", "."],
            cwd=self.repo, capture_output=True, text=True, check=False)
        self.corpus = [l for l in proc.stdout.splitlines() if l]

    def _rg_counts(self, pattern: str) -> dict[str, int]:
        proc = subprocess.run(
            [self._rg, "-F", "-i", "-c", "--hidden", "-g", "!.git", "-g", "!.git/**",
             pattern, "."],
            cwd=self.repo, capture_output=True, text=True, check=False)
        out = {}
        for line in proc.stdout.splitlines():
            path, _, cnt = line.rpartition(":")
            if path and cnt.isdigit():
                out[path] = int(cnt)
        return out

    def search(self, query: str) -> tuple[list[str], float, int]:
        """返回 (top-5 文件, 耗时秒, 词项数)。"""
        pats = extract_patterns(query)
        t0 = time.perf_counter()
        n = len(self.corpus) or 1
        scored: dict[str, float] = {}
        raw: dict[str, int] = {}
        for p in pats:
            cnts = self._rg_counts(p)
            df = len(cnts)
            if df == 0:
                continue
            idf = float(np.log(n / df))
            for path, c in cnts.items():
                scored[path] = scored.get(path, 0.0) + idf * c
                raw[path] = raw.get(path, 0) + c
        top = sorted(scored, key=lambda p: (-scored[p], -raw[p], p))[:TOP_K]
        return top, time.perf_counter() - t0, len(pats)


def dedupe_files(chunk_keys: list[tuple[str, int, int]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path, _, _ in chunk_keys:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def eval_file_arm(ranked: list[str], canon: tuple[str, ...]) -> int | None:
    canon_set = set(canon)
    return next((i + 1 for i, p in enumerate(ranked)
                 if Path(p).name in canon_set), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--json-out", type=Path, default=None)
    ns = ap.parse_args()

    repo = ns.repo
    if not repo.is_dir():
        print(f"benchmark repo not found: {repo} (pass --repo <path> or set CSEARCH_BENCH_REPO; for the public benchmark run scripts/bench_arb_trace2code.py)", file=sys.stderr)
        return 2

    # --- 冷启动分量 -------------------------------------------------------
    t0 = time.perf_counter()
    meta = load_meta(repo, index_dir_for(repo))
    searcher = Searcher(repo)
    t_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    embedder = Embedder(MODEL_DIR, dim=meta["dim"])
    t_model = time.perf_counter() - t0

    t0 = time.perf_counter()
    kw = KwIndex("cjk", searcher.records, repo)
    t_fts = time.perf_counter() - t0

    vectors = np.asarray(searcher.vectors)
    n_chunks = len(searcher.records)

    grep = GrepArm(repo)

    print(f"corpus: index {meta['chunks']} chunks / {meta['files']} files, "
          f"rg 可见 {len(grep.corpus)} files")
    print(f"cold: index {t_index:.2f}s, model {t_model:.2f}s, fts5(cjk) {t_fts:.2f}s\n")

    names = ["grep", "cs", "hybrid"]
    hits = {n: 0 for n in names}
    mrr = {n: 0.0 for n in names}
    lat: dict[str, list[float]] = {n: [] for n in names}
    ranks: dict[str, list] = {n: [] for n in names}
    top5_store: dict[str, list] = {n: [] for n in names}

    qvecs_cache = []
    for query, canon in QUERIES:
        canon_set = set(canon)

        # --- cs：逐题热检索（embed + matmul） ---
        t0 = time.perf_counter()
        q = embedder.encode_queries([query])[0]
        sims = vectors @ q
        top = np.argpartition(sims, n_chunks - POOL)[-POOL:]
        top = top[np.argsort(-sims[top])]
        vec_keys = [(searcher.records[int(i)]["path"], searcher.records[int(i)]["start"],
                     searcher.records[int(i)]["end"]) for i in top]
        cs_files = dedupe_files(vec_keys)[:TOP_K]
        dt_cs = time.perf_counter() - t0
        qvecs_cache.append((q, vec_keys))

        # --- hybrid：vec pool + BM25 cjk pool → RRF ---
        t0 = time.perf_counter()
        keys = [k[:3] for k in vec_keys]
        kw_keys = kw.search(query, POOL)
        fused = _rrf_fuse(keys, kw_keys)[:TOP_K]
        hy_files = dedupe_files(fused)[:TOP_K]
        dt_hy = time.perf_counter() - t0

        # --- grep：逐词项 rg + IDF ---
        g_files, dt_g, n_pats = grep.search(query)

        for n, files, dt in (("grep", g_files, dt_g), ("cs", cs_files, dt_cs),
                             ("hybrid", hy_files, dt_hy)):
            lat[n].append(dt * 1000)
            r = eval_file_arm(files, canon)
            ranks[n].append(r)
            top5_store[n].append(files)
            if r:
                hits[n] += 1
                mrr[n] += 1.0 / r

    # --- 汇总 -------------------------------------------------------------
    print(f"{'arm':<7} {'file-hit@5':>10} {'file-MRR@5':>10} "
          f"{'ms/query mean':>14} {'p50':>7}")
    for n in names:
        arr = sorted(lat[n])
        p50 = arr[len(arr) // 2]
        print(f"{n:<7} {hits[n]:>8}/8 {mrr[n]/8:>10.3f} "
              f"{sum(lat[n])/len(lat[n]):>14.1f} {p50:>7.1f}")

    print(f"\nper-query best canonical-file rank (top-5):")
    print(f"{'query':<26}" + "".join(f"{n:>8}" for n in names))
    for qi, (query, _) in enumerate(QUERIES):
        cells = [f"{ranks[n][qi]:>8}" if ranks[n][qi] else f"{'-':>8}" for n in names]
        print(f"{query[:24]:<26}" + "".join(cells))

    print("\nQ4 / Q7 top-5 files per arm（肉眼校验用）:")
    for qi in (3, 6):
        print(f"  Q{qi + 1} {QUERIES[qi][0][:22]}")
        for n in names:
            for p in top5_store[n][qi]:
                print(f"    {n:<7} {p}")

    # --- 端到端 CLI 一次（走常驻 daemon 的真实路径） -----------------------
    q0 = QUERIES[3][0]
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "csearch", "search", str(repo), q0],
        capture_output=True, text=True, check=False)
    dt_cli = time.perf_counter() - t0
    print(f"\n端到端: `csearch search` CLI 单题（含 spawn+daemon HTTP）: {dt_cli * 1000:.0f} ms, "
          f"exit={proc.returncode}, 输出行数={len(proc.stdout.splitlines())}")

    out = {"arms": {n: {"file_hit@5": hits[n], "file_mrr@5": round(mrr[n] / 8, 4),
                        "ms_mean": round(sum(lat[n]) / len(lat[n]), 1),
                        "ms_p50": round(sorted(lat[n])[len(lat[n]) // 2], 1)}
                    for n in names},
           "ranks": ranks, "cold": {"index_s": round(t_index, 2), "model_s": round(t_model, 2),
                                    "fts_s": round(t_fts, 2)}}
    if ns.json_out:
        ns.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"wrote {ns.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
