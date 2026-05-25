"""BM25-ish retrieval via Postgres FTS.

Two-path query strategy:

1. **Precise (websearch_to_tsquery).** Tried first. `websearch_to_tsquery`
   AND-joins free-text terms, which is more discriminating when every term
   actually appears in some chunk. For short keyword queries or
   well-targeted natural-language questions this is the better signal.

2. **OR fallback (to_tsquery).** Used only when the precise path returns
   zero rows. Long natural-language dev_gold questions routinely AND
   themselves into nothing — e.g. ~15 lemmatised terms joined by AND
   match no chunk even when the topic is heavily covered. We
   regex-tokenise the query in Python, drop stopwords and <3-char tokens,
   and feed the survivors to `to_tsquery` as `term1 | term2 | ...`. The
   token regex `[a-z0-9]+` is narrower than `to_tsquery`'s required
   syntax, so escaping is unnecessary.

The `chunks.tsv` column is GENERATED ALWAYS and indexed by `chunks_tsv_idx`
(GIN), so both paths stay fast at v1 corpus scale.
"""

from __future__ import annotations

import re

import psycopg

from retrieve.types import ChannelResult

_PRECISE_SQL = """
SELECT chunk_id, video_id, start_sec, end_sec, text,
       ts_rank(tsv, websearch_to_tsquery('english', %s)) AS score
FROM chunks
WHERE tsv @@ websearch_to_tsquery('english', %s)
ORDER BY score DESC
LIMIT %s
"""

_FALLBACK_SQL = """
SELECT chunk_id, video_id, start_sec, end_sec, text,
       ts_rank(tsv, to_tsquery('english', %s)) AS score
FROM chunks
WHERE tsv @@ to_tsquery('english', %s)
ORDER BY score DESC
LIMIT %s
"""

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "how",
        "what",
        "why",
        "when",
        "where",
        "who",
        "which",
        "does",
        "do",
        "did",
        "done",
        "doing",
        "this",
        "that",
        "these",
        "those",
        "such",
        "and",
        "but",
        "or",
        "nor",
        "so",
        "as",
        "also",
        "than",
        "very",
        "much",
        "many",
        "more",
        "most",
        "some",
        "any",
        "all",
        "each",
        "every",
        "between",
        "us",
        "we",
        "you",
        "he",
        "she",
        "it",
        "they",
        "i",
        "his",
        "her",
        "their",
        "its",
        "our",
        "your",
        "my",
        "me",
        "him",
        "them",
    }
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _expand_to_or_query(query: str) -> str | None:
    """Lower-case, regex-tokenise, drop stopwords and <3-char tokens, OR-join.

    Returns None when nothing meaningful remains — caller treats that as
    "no fallback possible, return empty". Tokens are pure ``[a-z0-9]+``,
    safe to pass to ``to_tsquery`` without further escaping. Postgres
    lexes them at query time (e.g. ``retrieval`` -> ``retriev``).
    """
    tokens = _WORD_RE.findall(query.lower())
    meaningful = [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]
    if not meaningful:
        return None
    seen: dict[str, None] = {}
    for t in meaningful:
        seen[t] = None
    return " | ".join(seen.keys())


def _to_result(row: tuple, idx: int) -> ChannelResult:
    return ChannelResult(
        chunk_id=row[0],
        video_id=row[1],
        start_sec=float(row[2]),
        end_sec=float(row[3]),
        text=row[4],
        score=float(row[5]),
        rank=idx + 1,
    )


def retrieve(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int = 30,
) -> list[ChannelResult]:
    """Run a Postgres FTS query against chunks.tsv, return top-k by ts_rank.

    Tries the precise ``websearch_to_tsquery`` path first; falls back to an
    OR-expanded ``to_tsquery`` over meaningful tokens when precise returns
    zero rows.
    """
    rows = conn.execute(_PRECISE_SQL, (query, query, top_k)).fetchall()
    if rows:
        return [_to_result(row, i) for i, row in enumerate(rows)]

    expanded = _expand_to_or_query(query)
    if expanded is None:
        return []

    rows = conn.execute(_FALLBACK_SQL, (expanded, expanded, top_k)).fetchall()
    return [_to_result(row, i) for i, row in enumerate(rows)]
