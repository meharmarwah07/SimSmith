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
        self.directory = directory
        self.chunks = []
        self._scene_cards: dict = {}  # template -> full file content, cached
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

    def scene_card(self, template: str) -> str:
        """Full content of docs_index/scene_<template>.md (ground-truth facts about the
        official scene XML for this template: bodies, joints, actuators, keyframes --
        see scripts/gen_scene_cards.py). Read whole, not chunked/ranked, so a caller can
        always include it regardless of what the keyword scorer in retrieve() picks."""
        if template in self._scene_cards:
            return self._scene_cards[template]
        path = os.path.join(self.directory, f"scene_{template}.md")
        text = ""
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        self._scene_cards[template] = text
        return text
