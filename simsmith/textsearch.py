"""Tiny dependency-free keyword retrieval used for docs and the script library."""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9_]+")


def tokens(text: str) -> list:
    return _WORD.findall(text.lower())


def rank(query: str, docs: list, k: int = 5) -> list:
    """Return indices of the top-k docs by a simple TF-IDF-like score."""
    q = Counter(tokens(query))
    if not q or not docs:
        return []
    doc_tokens = [Counter(tokens(d)) for d in docs]
    n = len(docs)
    df = Counter()
    for dt in doc_tokens:
        for w in dt:
            df[w] += 1
    scores = []
    for i, dt in enumerate(doc_tokens):
        s = 0.0
        for w, qc in q.items():
            if w in dt:
                s += qc * (1 + math.log(dt[w])) * math.log(1 + n / df[w])
        scores.append((s, i))
    scores.sort(reverse=True)
    return [i for s, i in scores[:k] if s > 0]
