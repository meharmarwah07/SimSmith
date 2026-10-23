"""Version-pinned documentation index.

Put plain-text/markdown excerpts of API notes and gotchas for your pinned
simulator (MuJoCo, Newton, ...) in docs_index/. Mixing versions is the main
cause of hallucinated APIs.
"""
from __future__ import annotations

import os

from .textsearch import rank


class DocIndex:
    def __init__(self, directory: str = "docs_index", chunk_chars: int = 1200):
        self.chunks = []
        if os.path.isdir(directory):
            for name in sorted(os.listdir(directory)):
                path = os.path.join(directory, name)
                if not os.path.isfile(path) or not name.endswith((".md", ".txt", ".py", ".rst")):
                    continue
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                for i in range(0, len(text), chunk_chars):
                    self.chunks.append(f"[{name}]\n{text[i:i + chunk_chars]}")

    def retrieve(self, query: str, k: int = 6) -> list:
        return [self.chunks[i] for i in rank(query, self.chunks, k)]
