"""Re-classify stored eval results with the fixed classify_failure() (see
scripts/baseline_dev.py and docs/diagnosis_2026-10-24.md Part 1: the previous
hallucinated_api check matched "attributeerror" + "module" as substrings, which fires
on ANY AttributeError because every traceback frame literally renders "in <module>" --
not just real module-attribute hallucinations). Does NOT rerun anything -- reads
existing eval/results/*.jsonl and recomputes each row's classification from its OWN
previously-saved failure text, bit for bit. Original files are never modified; a
corrected sibling "<name>.reclassified.jsonl" is written next to each one that could be
re-derived.

Three row shapes, three strategies:
  * single-shot rows (baseline_dev.py: baseline_progress*.jsonl, part3_singleshot_*,
    part4_*_progress.jsonl, smoke_*_progress.jsonl): each row already stores
    preflight_ok/preflight_problems/score_ok/crashes/failure_text directly --
    classify_failure() is just re-applied to those same saved fields.
  * best_of_5-style rows WITH per-candidate candidate_log (the "runs": [...] field,
    added during the 2026-10-24 diagnosis -- see simsmith/pipeline.py's
    candidate_log()): a candidate's failure text is reconstructed from its own stored
    per-seed RunResult.error, in exactly verify()'s own "seed N: crashed: ..." format
    (simsmith/verifier.py), then re-classified the same way. This is sufficient to fix
    the hallucinated_api bug specifically: that bug can only fire when a stored
    AttributeError traceback is present, which only happens on an actual subprocess
    crash (run["ok"] is False) -- exactly what's reconstructed here. It is NOT a full
    reconstruction of verify()'s other crash reasons (missing result keys / simulation
    too short / task-param mismatch), which depend on final_state that candidate_log
    doesn't retain; those are rare, don't interact with this bug, and are flagged if
    they appear to change class anyway.
  * best_of_5-style rows WITHOUT candidate_log (older runs: baseline_progress_v2_bof5,
    part3_bof5_norepair/_repair, partC_A/B/C): no raw failure text was ever stored for
    these, so they genuinely cannot be re-classified from what's on disk. Reported as
    "not reclassifiable", left untouched -- old numbers for these stand, with that
    caveat.

Usage:
  python scripts/reclassify_results.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_dev import classify_failure, _MODULE_ATTR_ERROR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "eval", "results")

SINGLESHOT_FILES = [
    "baseline_progress.jsonl",
    "baseline_progress_v2.jsonl",
    "part3_singleshot_progress.jsonl",
    "part4_super_reasoning_on_progress.jsonl",
    "part4_ultra_progress.jsonl",
    "smoke_super_progress.jsonl",
    "smoke_ultra_progress.jsonl",
]
BOF5_WITH_LOG_FILES = [
    "part1_diagnosis_progress.jsonl",
    "part3_super_reach_run1_progress.jsonl",
    "part3_super_reach_run2_progress.jsonl",
    "part3_super_reach_run3_progress.jsonl",
    "part3_super_pick_place_progress.jsonl",
    "part3_ultra_pick_place_progress.jsonl",
]
BOF5_WITHOUT_LOG_FILES = [
    "baseline_progress_v2_bof5.jsonl",
    "part3_bof5_norepair_progress.jsonl",
    "part3_bof5_repair_progress.jsonl",
    "partC_A_progress.jsonl",
    "partC_B_progress.jsonl",
    "partC_C_progress.jsonl",
]


def _load(fname: str) -> list:
    path = os.path.join(RESULTS, fname)
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _write(fname: str, rows: list) -> str:
    out_name = fname.replace(".jsonl", ".reclassified.jsonl")
    out_path = os.path.join(RESULTS, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out_path


def reconstruct_candidate_failure_text(cand: dict) -> str:
    """Reconstructed crash text from a candidate_log's "runs" list, built the same way
    verify() builds fail_lines for an actual subprocess crash (run["ok"] is False).
    This does NOT cover verify()'s other crash reasons (missing result keys, too-short
    simulation, task-param mismatch) -- those depend on final_state, which
    candidate_log never retained, so they are NOT reconstructable from stored data.
    That's fine for THIS fix: the hallucinated_api bug can only ever have fired on an
    AttributeError traceback, which only exists when there WAS a real subprocess crash
    -- exactly what this does recover -- so it's sufficient to confirm or demote a
    hallucinated_api label without needing the rest."""
    runs = cand.get("runs") or []
    fail_lines = [f"seed {r['seed']}: crashed: {r['error'][-600:]}" for r in runs if not r["ok"]]
    return "\n".join(fail_lines)


def reclassify_singleshot(fname: str) -> dict:
    rows = _load(fname)
    changes = []
    for r in rows:
        old = r["classification"]
        new = classify_failure(r["preflight_ok"], r["preflight_problems"] or [],
                               r["score_ok"], r["crashes"], r["failure_text"])
        if new != old:
            changes.append({"task_id": r["task_id"], "template": r["template"],
                            "i": r.get("i"), "old": old, "new": new})
        r["classification"] = new
        r["classification_orig"] = old
    out_path = _write(fname, rows)
    return {"file": fname, "kind": "single_shot", "n_rows": len(rows),
            "n_changed": len(changes), "changes": changes, "out_path": out_path}


def reclassify_bof5_with_log(fname: str) -> dict:
    """Only ever touches a candidate currently labeled "hallucinated_api" -- that is
    the one bucket the old classify() is known to mislabel into (see module
    docstring). Confirms it (stays hallucinated_api) if the reconstructed crash text
    still contains a genuine ModuleNotFoundError/ImportError/module-attribute pattern,
    otherwise demotes it to "crash" (the old bug's false-positive case). Every other
    original label (crash, wrong_logic, no_simulation, task_param_mismatch, pass,
    preflight_other) is left exactly as originally computed -- full reconstruction of
    those from candidate_log alone would be unreliable (see
    reconstruct_candidate_failure_text()'s docstring) and isn't needed to fix this bug."""
    rows = _load(fname)
    row_changes = []
    for r in rows:
        old_hallu_by_round = dict(r.get("hallucinated_by_round") or {})
        new_hallu_by_round = Counter()
        cand_changes = []
        for cand in r["candidates"]:
            old_cls = cand["classification"]
            new_cls = old_cls
            if old_cls == "hallucinated_api":
                text = reconstruct_candidate_failure_text(cand).lower()
                genuine = ("modulenotfounderror" in text or "no module named" in text
                          or "importerror" in text or _MODULE_ATTR_ERROR.search(text))
                new_cls = "hallucinated_api" if genuine else "crash"
            if new_cls != old_cls:
                cand_changes.append({"idx": cand["idx"], "round": cand["round"],
                                     "old": old_cls, "new": new_cls})
            cand["classification"] = new_cls
            cand["classification_orig"] = old_cls
            if new_cls == "hallucinated_api":
                new_hallu_by_round[str(cand["round"])] += 1

        # Recompute the task-run-level failure_classification ONLY if it was
        # "hallucinated_api" -- any other old label is untouched (see docstring), and
        # by construction (the row's old label is exactly classify(winner), see
        # baseline_best_of_5.py) the actual winner candidate must be among the ones
        # whose OLD per-candidate classification was "hallucinated_api". There can be
        # several such candidates sharing the winner's (stage, round) (up to 5 tied at
        # stage="generated", round=0 -- repair rounds 1/2 are each unique, so no
        # ambiguity there) and stored data doesn't retain which exact idx won, so:
        #   - if ALL of them demote to "crash" after the fix, the true winner (being
        #     one of them) must have too -- certain, not a guess.
        #   - if ALL of them stay "hallucinated_api" (confirmed genuine), unchanged.
        #   - if the group is mixed, which exact one was the winner is genuinely
        #     unrecoverable from stored data -- flagged rather than guessed.
        old_failure_classification = r.get("failure_classification")
        new_failure_classification = old_failure_classification
        ambiguous = False
        if old_failure_classification == "hallucinated_api":
            same_group_orig_hallu = [c for c in r["candidates"] if c["classification_orig"] == "hallucinated_api"]
            outcomes = {c["classification"] for c in same_group_orig_hallu}
            if outcomes == {"crash"}:
                new_failure_classification = "crash"
            elif outcomes == {"hallucinated_api"}:
                new_failure_classification = "hallucinated_api"
            else:
                ambiguous = True  # mixed outcomes; which exact candidate won isn't recoverable

        if cand_changes or new_failure_classification != old_failure_classification or ambiguous:
            row_changes.append({
                "task_id": r["task_id"], "run_idx": r.get("run_idx"), "template": r["template"],
                "candidate_changes": cand_changes,
                "failure_classification_old": old_failure_classification,
                "failure_classification_new": new_failure_classification,
                "ambiguous": ambiguous,
            })
        r["failure_classification_orig"] = old_failure_classification
        r["failure_classification"] = new_failure_classification
        r["hallucinated_by_round_orig"] = old_hallu_by_round
        r["hallucinated_by_round"] = dict(new_hallu_by_round)
    out_path = _write(fname, rows)
    return {"file": fname, "kind": "best_of_5_with_log", "n_rows": len(rows),
            "n_changed": len(row_changes), "changes": row_changes, "out_path": out_path}


def main() -> int:
    reports = []
    for fname in SINGLESHOT_FILES:
        if os.path.exists(os.path.join(RESULTS, fname)):
            reports.append(reclassify_singleshot(fname))
    for fname in BOF5_WITH_LOG_FILES:
        if os.path.exists(os.path.join(RESULTS, fname)):
            reports.append(reclassify_bof5_with_log(fname))

    not_reclassifiable = [f for f in BOF5_WITHOUT_LOG_FILES if os.path.exists(os.path.join(RESULTS, f))]

    print("=" * 100)
    print("RECLASSIFICATION REPORT (classify_failure() hallucinated_api fix)")
    print("=" * 100)
    total_changed = 0
    for rep in reports:
        total_changed += rep["n_changed"]
        print(f"\n--- {rep['file']} ({rep['kind']}, {rep['n_rows']} rows) ---")
        print(f"changed: {rep['n_changed']}/{rep['n_rows']}")
        for ch in rep["changes"]:
            print(f"  {json.dumps(ch)}")
        print(f"  -> wrote {rep['out_path']}")

    print(f"\n--- NOT reclassifiable (no raw failure text stored) ---")
    for f in not_reclassifiable:
        print(f"  {f}  (left untouched; old numbers stand with this caveat)")

    print(f"\nTotal rows/candidates whose label changed: {total_changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
