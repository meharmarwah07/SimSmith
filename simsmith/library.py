"""Library of verified scripts, stored as JSON lines.

Only scripts that passed execution verification are saved. Retrieval feeds
them to the generator as few-shot examples on later tasks.
"""
from __future__ import annotations

import json
import os
import time

from .textsearch import rank


class Library:
    def __init__(self, path: str = "library.jsonl"):
        self.path = path
        self.records = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def add(self, task: str, template: str, script: str, success_rate: float,
            isaac_version: str = "unset") -> dict:
        rec = {"task": task, "template": template, "script": script,
               "success_rate": success_rate, "isaac_version": isaac_version,
               "ts": time.time()}
        self.records.append(rec)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def retrieve(self, task: str, template: str, k: int = 2) -> list:
        pool = [r for r in self.records if r["template"] == template]
        idx = rank(task, [r["task"] for r in pool], k)
        return [pool[i] for i in idx]
