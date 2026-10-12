"""MMLU-Pro answer extraction and scoring.

Extraction is the three-stage procedure of the reference runner of TIGER-AI-Lab/MMLU-Pro (Apache-2.0),
`evaluate_from_local.py` at f418b116db00b065c2aea046518d8fcf74d39872:
  1. "answer is (X)"           2. "answer: X"           3. the last standalone letter A-J in the text.

Scoring differs from the reference in exactly one place: when nothing can be extracted the reference guesses a
random option; here the answer is simply incorrect (extraction failures count as failures). Everything else —
what counts as a correct prediction and the micro-averaged accuracy — is the same.

This file's SHA-256 is written into every frozen plan (`scorer_sha256`) and re-checked when the plan is loaded,
so the scorer cannot change silently between freezing a plan and scoring a run.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

LETTERS = "ABCDEFGHIJ"

_STAGE1 = r"answer is \(?([A-J])\)?"
_STAGE2 = r".*[aA]nswer:\s*([A-J])"
_STAGE3 = r"\b[A-J]\b(?!.*\b[A-J]\b)"


def extract_answer(text: str) -> Optional[str]:
    for pattern in (_STAGE1, _STAGE2):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    match = re.search(_STAGE3, text, re.DOTALL)
    return match.group(0) if match else None


def extraction_stage(text: str) -> int:
    """Which stage produced the answer: 1, 2, 3, or 0 when nothing was extracted."""
    if re.search(_STAGE1, text):
        return 1
    if re.search(_STAGE2, text):
        return 2
    return 3 if re.search(_STAGE3, text, re.DOTALL) else 0


def _summary(rows: List[dict]) -> dict:
    correct = sum(r["correct"] for r in rows)
    return {"count": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None}


def score_results(results: List[dict], cases: List[dict]) -> dict:
    """Score request results against the frozen cases. Every case must have exactly one result."""
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Expected nonempty unique task manifest")
    lookup: Dict[str, dict] = {c["case_id"]: c for c in cases}
    if len(results) != len(cases) or {r["case_id"] for r in results} != set(lookup):
        raise ValueError("Missing or duplicate task outcomes")
    outcomes = []
    for result in results:
        case = lookup[result["case_id"]]
        for key in ("prompt_sha256", "prompt_tokens", "max_tokens", "ignore_eos"):
            if result[key] != case[key]:
                raise ValueError("Unmatched request contract: %s" % key)
        prediction = extract_answer(result.get("output", "")) if result["success"] else None
        valid = prediction is not None and prediction in LETTERS[: case["option_count"]]
        outcomes.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "prediction": prediction,
            "valid": valid,
            "correct": valid and prediction == case["answer"],
            "transport_success": result["success"],
            "length_limited": result.get("finish_reason") == "length",
        })
    return {
        **_summary(outcomes),
        "invalid_answers": sum(not o["valid"] for o in outcomes),
        "transport_failures": sum(not o["transport_success"] for o in outcomes),
        "length_limited": sum(o["length_limited"] for o in outcomes),
        "categories": {c: _summary([o for o in outcomes if o["category"] == c])
                       for c in sorted({o["category"] for o in outcomes})},
        "outcomes": outcomes,
    }
