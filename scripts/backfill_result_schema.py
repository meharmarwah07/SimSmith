"""Adds the new result-row schema fields (see simsmith/run_meta.py) to COPIES of
older eval/results/*.json report files, so historical runs can at least be partially
compared against new ones without hand cross-referencing. Never modifies an original
file -- writes each backfilled copy to eval/results/backfilled/<same filename>.

Only adds a field when it can be genuinely INFERRED from what the old report already
recorded. Several schema fields simply aren't recoverable after the fact and are
written as null rather than guessed:
  * git_commit    -- never recorded historically; no way to know it now.
  * max_tokens    -- depends on SIMSMITH_MAX_TOKENS_* env vars in effect at run time,
                     never recorded; today's defaults are not assumed to have applied.
  * workers       -- the --workers CLI value was never recorded in the report.
  * classifier_version -- see CLASSIFIER_VERSION_NOTE below; left null, not guessed.

Inferred fields (from data the old report already has):
  * condition      -- "single_shot" for a baseline_*.json file; for a best_of_5_*.json
                      file, derived from its own stored top-level max_repairs (>0 ->
                      "best_of_5_repair", else "best_of_5") -- NOT trusted from the
                      file's own "condition" field, which was hardcoded to the literal
                      string "best_of_5" regardless of max_repairs until this same
                      2026-10-25 change (see git history on scripts/baseline_best_of_5.py).
  * n_candidates   -- 1 for single_shot, 5 for best_of_5(_repair).
  * max_repairs    -- the file's own top-level field, or 0 if absent (pre-repair runs).
  * model          -- the file's own top-level "model".
  * prompt_version -- the file's own top-level "prompt_version".
  * reasoning      -- derived from the file's own top-level "extra_payload" via the
                      same simsmith.run_meta.reasoning_mode() logic current runs use.
  * seeds          -- the file's own top-level "sim_seeds".
  * temperature    -- inferred as prompts.TEMPERATURES (best_of_5) or [0.3]
                      (single_shot) -- these constants are not known to have changed
                      across any historical run, but this is an assumption, not a
                      stored fact, so it's flagged with "temperature_inferred": true.
  * date           -- from the file's own top-level "timestamp" (unix epoch), if present.

CLASSIFIER_VERSION_NOTE: whether a given historical row's classification used the
buggy pre-fix hallucinated_api regex (see docs/diagnosis_2026-10-24.md Part 1) depends
on which classify_failure() was in scripts/baseline_dev.py AT THE TIME the run
happened -- not recorded in the report, and not safely inferable from the report's
*.json files alone. scripts/reclassify_results.py already recomputes corrected
classifications for the *_progress.jsonl files it knows about; this backfill script
does not duplicate that -- it only stamps a "classifier_version": null placeholder
plus this note, pointing at reclassify_results.py for the rows that tool covers.

Usage:
  python scripts/backfill_result_schema.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simsmith import prompts  # noqa: E402
from simsmith.run_meta import reasoning_mode  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "eval", "results")
OUT_DIR = os.path.join(RESULTS, "backfilled")

CLASSIFIER_VERSION_NOTE = (
    "not recorded at run time and not safely inferable from this report alone -- see "
    "scripts/reclassify_results.py for *_progress.jsonl files it can recompute a "
    "corrected classification for."
)


def _infer_top_level(report: dict, is_best_of_5: bool) -> dict:
    max_repairs = report.get("max_repairs", 0) if is_best_of_5 else 0
    condition = ("best_of_5_repair" if max_repairs > 0 else "best_of_5") if is_best_of_5 else "single_shot"
    n_candidates = 5 if is_best_of_5 else 1
    extra_payload = report.get("extra_payload") or {}
    date = None
    ts = report.get("timestamp")
    if isinstance(ts, (int, float)):
        date = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return {
        "condition": condition,
        "model": report.get("model"),
        "reasoning": reasoning_mode(extra_payload),
        "prompt_version": report.get("prompt_version"),
        "classifier_version": None,
        "classifier_version_note": CLASSIFIER_VERSION_NOTE,
        "n_candidates": n_candidates,
        "max_repairs": max_repairs,
        "max_tokens": None,
        "temperature": list(prompts.TEMPERATURES) if n_candidates > 1 else [0.3],
        "temperature_inferred": True,
        "seeds": report.get("sim_seeds"),
        "workers": None,
        "git_commit": None,
        "date": date,
    }


def backfill_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    fname = os.path.basename(path)
    is_best_of_5 = fname.startswith("best_of_5_")
    meta = _infer_top_level(report, is_best_of_5)

    # Direct assignment at top level, not setdefault: the old best_of_5_*.json
    # "condition" field is known-wrong for max_repairs>0 runs (hardcoded to the
    # literal string "best_of_5" regardless -- see module docstring), so it must be
    # corrected here, not preserved.
    report.update(meta)
    n_rows_touched = 0
    rows_by_template = report.get("rows") or {}
    for template, rows in rows_by_template.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Old best_of_5_*.json rows already have an "n_candidates" key, but it
            # means the ACTUAL count generated (len(res.candidates), which can be < 5
            # if generation stopped early) -- a different thing from the new schema's
            # "n_candidates" (the condition's configured target). Rename the old
            # field before stamping the new one in, so neither is silently lost or
            # misread as the other.
            if is_best_of_5 and "n_candidates" in row and "n_candidates_generated" not in row:
                row["n_candidates_generated"] = row.pop("n_candidates")
            for k, v in meta.items():
                row.setdefault(k, v)  # never overwrite any other field the row already has
            n_rows_touched += 1

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return {"file": fname, "out_path": out_path, "n_rows_touched": n_rows_touched,
            "condition": meta["condition"]}


def main() -> int:
    paths = sorted(glob.glob(os.path.join(RESULTS, "baseline_*.json")) +
                   glob.glob(os.path.join(RESULTS, "best_of_5_*.json")))
    paths = [p for p in paths if os.path.dirname(p) == RESULTS]  # skip OUT_DIR itself
    print(f"backfilling {len(paths)} file(s) -> {OUT_DIR}/ (originals untouched)\n")
    for path in paths:
        result = backfill_file(path)
        print(f"  {result['file']:40s} condition={result['condition']:<18} "
              f"rows_touched={result['n_rows_touched']:3d} -> {result['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
