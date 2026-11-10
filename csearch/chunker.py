"""Line-aware code chunking.

Chunk size is bounded by characters (the model contexts in tokens are roughly
proportional), cut at line boundaries, and retain a tail of the previous window
so definitions spanning a boundary stay retrievable from both sides.
"""

from __future__ import annotations

DEFAULT_MAX_CHARS = 1400  # ~400-500 tokens; well inside the 8K window
DEFAULT_OVERLAP_CHARS = 200


def chunk_code(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[tuple[int, int, str]]:
    """Split source text into chunks.

    Returns (start_line, end_line, text) with 0-based inclusive start and
    exclusive end line numbers. Single lines longer than ``max_chars``
    (minified/generated code) are split into their own pieces.
    """
    lines = text.splitlines(keepends=True)
    out: list[tuple[int, int, str]] = []
    buf: list[str] = []
    buf_start = 0
    buf_chars = 0

    def push(end_line: int) -> None:
        if buf:
            out.append((buf_start, end_line, "".join(buf)))

    for i, ln in enumerate(lines):
        # pathological single line: emit it as self-contained pieces
        while len(ln) > max_chars:
            push(i)
            buf = []
            buf_chars = 0
            buf_start = i
            out.append((i, i + 1, ln[:max_chars]))
            ln = ln[max_chars:]

        if buf and buf_chars + len(ln) > max_chars:
            push(i)
            # keep a tail of the window we just closed for overlap
            tail: list[str] = []
            size = 0
            j = len(buf) - 1
            while j >= 0 and size + len(buf[j]) <= overlap_chars:
                tail.insert(0, buf[j])
                size += len(buf[j])
                j -= 1
            buf = tail
            buf_chars = size
            buf_start = i - len(tail)

        if not buf:
            buf_start = i
        buf.append(ln)
        buf_chars += len(ln)

    push(len(lines))
    return out

