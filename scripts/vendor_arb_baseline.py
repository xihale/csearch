"""Verbatim vendored functions from Agent Retrieval Bench (ARB) evaluator.

Source: https://github.com/eyuansu62/agent-retrieval-bench
        src/agent_retrieval_bench/baseline.py (main branch, fetched 2026-11-06)
Copied unchanged except: relative imports dropped, unused helpers omitted.
Used so our trace2code rerun applies the exact published protocol.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
BM25_K1 = 1.5
BM25_B = 0.75


# --- gold / query -------------------------------------------------------------


def target_gold_files(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") or {}
    if gold.get("no_gold") is True:
        return []
    explicit_files = _gold_paths(gold.get("files") or [])
    if explicit_files:
        return explicit_files
    if sample.get("task_type") == "code2test":
        return _dedupe(gold.get("related_tests") or [])
    if sample.get("task_type") == "comment2context":
        context_files = _gold_paths(gold.get("must_context_files") or gold.get("context_files") or [])
        if context_files:
            return context_files
    root_files = _dedupe(gold.get("root_cause_files") or [])
    return root_files or _dedupe(gold.get("related_tests") or [])


def query_text_for_eval(sample: dict[str, Any]) -> str:
    return json.dumps(sample.get("query") or {}, ensure_ascii=False, sort_keys=True)


def _gold_paths(values: Iterable[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, dict) and value.get("path"):
            paths.append(str(value["path"]))
    return _dedupe(paths)


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


# --- ranking (exact ARB baselines) ---------------------------------------------


def rank_chunks_with_scores(query_text: str, chunks: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return [(0.0, chunk) for chunk in chunks]
    doc_tokens: list[set[str]] = []
    document_frequency = Counter()
    for chunk in chunks:
        tokens = set(tokenize(chunk_text(chunk)))
        doc_tokens.append(tokens)
        document_frequency.update(tokens)
    total_docs = max(1, len(chunks))
    query_counts = Counter(query_tokens)
    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    lowered_query = query_text.lower()
    for chunk, tokens in zip(chunks, doc_tokens):
        score = 0.0
        for token, count in query_counts.items():
            if token in tokens:
                score += (1.0 + math.log(count)) * math.log((total_docs + 1) / (1 + document_frequency[token]) + 1.0)
        path = str(chunk.get("path", ""))
        lowered_path = path.lower()
        basename = Path(path).name.lower()
        if lowered_path and lowered_path in lowered_query:
            score += 25.0
        if basename and basename in lowered_query:
            score += 8.0
        if chunk.get("symbol") and str(chunk["symbol"]).lower() in lowered_query:
            score += 5.0
        normalized = score / max(1.0, math.sqrt(len(tokens)))
        scored.append((normalized, path, str(chunk.get("chunk_id", "")), chunk))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(item[0], item[3]) for item in scored]


def rank_chunks_bm25_with_scores(
    query_text: str,
    chunks: list[dict[str, Any]],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[tuple[float, dict[str, Any]]]:
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return [(0.0, chunk) for chunk in chunks]
    query_counts = Counter(query_tokens)
    document_tokens = [tokenize(chunk_text(chunk)) for chunk in chunks]
    document_lengths = [len(tokens) for tokens in document_tokens]
    average_document_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
    document_frequency = Counter()
    for tokens in document_tokens:
        document_frequency.update(set(tokens))

    total_docs = max(1, len(chunks))
    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    for chunk, tokens, doc_length in zip(chunks, document_tokens, document_lengths):
        term_counts = Counter(tokens)
        score = 0.0
        for token, query_count in query_counts.items():
            tf = term_counts.get(token, 0)
            if not tf:
                continue
            df = document_frequency[token]
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            length_norm = 1.0 - b
            if average_document_length > 0:
                length_norm += b * (doc_length / average_document_length)
            denominator = tf + k1 * length_norm
            score += query_count * idf * ((tf * (k1 + 1.0)) / denominator)
        path = str(chunk.get("path", ""))
        scored.append((score, path, str(chunk.get("chunk_id", "")), chunk))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(item[0], item[3]) for item in scored]


# --- metrics -------------------------------------------------------------------


def unique_ranked_paths(ranked_chunks: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for chunk in ranked_chunks:
        path = str(chunk.get("path", ""))
        if path and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def recall_at(gold: set[str], ranked_paths: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(ranked_paths[:k])) / len(gold)


def reciprocal_rank(gold: set[str], ranked_paths: list[str]) -> float:
    for index, path in enumerate(ranked_paths, start=1):
        if path in gold:
            return 1.0 / index
    return 0.0


def chunk_text(chunk: dict[str, Any]) -> str:
    return " ".join(
        str(part)
        for part in [chunk.get("path"), chunk.get("symbol"), chunk.get("kind"), chunk.get("text")]
        if part
    )


def tokenize(text: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [token.lower() for token in TOKEN_RE.findall(spaced)]
