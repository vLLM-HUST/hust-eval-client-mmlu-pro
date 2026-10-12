"""Freeze an MMLU-Pro plan (see plan.py). Needs the optional dependencies pyarrow and transformers.

By default every test question is selected (the full test set). `per_category=N` selects the first N question
IDs of each category instead (useful for a quick, clearly-labelled subset).
"""
from __future__ import annotations

from typing import Optional

from . import plan


def prepare(assets: str, model_path: str, reference: str, output: str, runtime_contract: Optional[str] = None,
            per_category: Optional[int] = None) -> dict:
    return plan.prepare(assets, model_path, reference, output, runtime_contract, per_category)
