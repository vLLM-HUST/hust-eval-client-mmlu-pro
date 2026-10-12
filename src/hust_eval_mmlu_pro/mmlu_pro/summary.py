"""Summarise an MMLU-Pro accuracy run (and an optional bench run) into one report. Read-only.

If the run is unfinished the report covers only finished requests and says PARTIAL: the plan is sorted by
category, so a partial run covers only the first categories and is NOT a score.
Scoring uses scoring.score_results.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
from typing import Optional

from . import plan as plan_mod
from . import scoring


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q / 100 * len(xs) + 0.5)) - 1))]  # nearest-rank


def _dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs),
            "p95": _pct(xs, 95), "p99": _pct(xs, 99), "max": max(xs)}


def build_report(plan: str, accuracy_dir: str, bench_json: Optional[str] = None, allow_legacy: bool = False) -> dict:
    contract, _ = plan_mod.load_plan(plan, allow_legacy=allow_legacy)  # verifies the plan before trusting it
    with open(os.path.join(plan, "tasks.json")) as f:
        tasks = json.load(f)  # cases minus prompt token IDs
    by_id = {t["case_id"]: t for t in tasks}
    results = []
    for fn in sorted(glob.glob(os.path.join(accuracy_dir, "requests", "*.json"))):
        with open(fn) as f:
            results.append(json.load(f))
    results = [r for r in results if r["case_id"] in by_id]
    cases = [by_id[r["case_id"]] for r in results]
    scores = scoring.score_results(results, cases)
    outcomes = {o["case_id"]: o for o in scores["outcomes"]}

    ok = [r for r in results if r["success"]]
    stop = [r for r in results if r.get("finish_reason") != "length"]
    cut = [r for r in results if r.get("finish_reason") == "length"]

    def acc(group):
        return {"n": len(group), "correct": sum(outcomes[r["case_id"]]["correct"] for r in group),
                "stage_counts": {str(s): sum(1 for r in group if scoring.extraction_stage(r["output"]) == s) for s in (1, 2, 3, 0)}}

    report = {
        "legacy_plan": bool(contract.get("_legacy_plan")),
        "complete": len(results) == len(tasks),
        "WARNING": None if len(results) == len(tasks) else
        "PARTIAL: %d of %d finished; plan order is sorted by category, so this is NOT a score" % (len(results), len(tasks)),
        "accuracy": {
            "cases": scores["count"], "correct": scores["correct"], "accuracy": scores["accuracy"],
            "invalid_answers": scores["invalid_answers"], "transport_failures": scores["transport_failures"],
            "length_limited": scores["length_limited"],
            "length_limited_but_correct": sum(1 for o in scores["outcomes"] if o["length_limited"] and o["correct"]),
            "finish_reasons": {str(k): sum(1 for r in results if r.get("finish_reason") == k)
                               for k in {r.get("finish_reason") for r in results}},
        },
        "by_termination": {"finished_normally": acc(stop), "length_limited": acc(cut),
                           "note": "stage 3 = fallback that takes the last standalone A-J letter; for length-limited "
                                   "outputs it often picks a letter that is just text (e.g. 'A', 'I')"},
        "per_category": {c: {"n": v["count"], "correct": v["correct"], "accuracy": v["accuracy"]}
                         for c, v in scores["categories"].items()},
        "tokens": {"prompt": _dist([r["usage"]["prompt_tokens"] for r in ok if r.get("usage")]),
                   "completion": _dist([r["usage"]["completion_tokens"] for r in ok if r.get("usage")])},
        "client_timing_seconds": {"ttft": _dist([r.get("ttft_seconds") for r in ok]),
                                  "e2e": _dist([r.get("e2e_seconds") for r in ok])},
    }
    runs = sorted(glob.glob(os.path.join(accuracy_dir, "run.*.json")))
    if runs:
        with open(runs[0]) as f:
            meta = json.load(f)
        report["run_meta"] = {k: meta.get(k) for k in ("concurrency", "upstream_concurrency", "generation", "n_cases",
                                                      "runner_sha256", "scorer_sha256", "adapter_sha256",
                                                      "contract_sha256", "deviations")}
    if bench_json and os.path.exists(bench_json):
        with open(bench_json) as f:
            b = json.load(f)
        keys = ["date", "num_prompts", "completed", "failed", "duration", "request_throughput", "output_throughput",
                "total_token_throughput", "total_input_tokens", "total_output_tokens", "mean_ttft_ms", "median_ttft_ms",
                "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "p99_itl_ms",
                "max_concurrent_requests"]
        report["bench"] = {k: b.get(k) for k in keys}
    else:
        report["bench"] = "not run yet"
    return report
