"""Freezing and loading an MMLU-Pro plan.

A plan is a directory:
  cases.json            every task with its frozen prompt as token IDs, gold answer, option count, limits
  tasks.json            the same without the prompt token IDs (small; what reports and summaries read)
  inventory.jsonl.gz    one canonical JSON line per TEST question (all of them, selected or not), gzip mtime=0
  contract.json         what was frozen and the SHA-256 of the three files above, plus the scorer's SHA-256

`prepare` needs pyarrow and transformers (optional dependencies). `build_plan` and `load_plan` do not.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..hashing import sha256_file
from . import prompt as prompt_mod
from . import scoring

DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
REFERENCE_REVISION = "f418b116db00b065c2aea046518d8fcf74d39872"
DATA_HASHES = {
    "test": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
    "validation": "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078",
}
REFERENCE_SCRIPT_SHA256 = "d0a4ae51b80efa8f99713a21253ad5542913ef2ac9915e95d40c8a62174e9401"
INITIAL_PROMPT_SHA256 = "d7337806b4d84eef7a86bcb20bf3d64b65e72736c1603e16d08cadebac436f0a"
TEST_ROWS, VALIDATION_ROWS = 12032, 70
MAX_OUTPUT = 2048
MAX_CONTEXT = 32768
CONTRACT_VERSION = "mmlu-pro-plan-v1"
PLAN_FILES = ("cases.json", "tasks.json", "inventory.jsonl.gz")
LEGACY_CONTRACT_VERSION = "pyramidkv-mmlu-pro-subset-v1"   # plans made by the upstream adapter


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    """Canonical JSON bytes: sorted keys, no whitespace, UTF-8 (used for row hashes and inventory lines)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _dump(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def _select(inventory: List[dict], per_category: Optional[int]) -> set:
    if per_category is None:
        return {x["question_id"] for x in inventory}
    if per_category < 1:
        raise ValueError("per_category must be positive")
    chosen = set()
    for category in sorted({x["category"] for x in inventory}):
        members = sorted((x for x in inventory if x["category"] == category), key=lambda x: x["question_id"])
        chosen.update(x["question_id"] for x in members[:per_category])
    return chosen


def build_plan(test_rows: List[dict], validation_rows: List[dict], instruction: str,
               encode: Callable[[str], List[int]], per_category: Optional[int] = None,
               max_output: int = MAX_OUTPUT, max_context: int = MAX_CONTEXT) -> Tuple[List[dict], List[dict]]:
    """Pure function: (inventory, cases). Needs no files, no tokenizer library (pass any text -> token IDs)."""
    subjects = sorted({r["category"] for r in test_rows})
    prefixes = prompt_mod.build_prefixes(instruction, validation_rows, subjects)
    by_id: Dict[int, dict] = {}
    inventory: List[dict] = []
    prompts: Dict[int, List[int]] = {}
    for row in test_rows:
        qid = row["question_id"]
        if qid in by_id:
            raise ValueError("Duplicate immutable task ID")
        by_id[qid] = row
        ids = encode(prompt_mod.user_text(prefixes, row))
        if len(ids) + max_output > max_context:
            raise ValueError("Prompt exceeds the frozen context; do not silently truncate")
        prompts[qid] = ids
        inventory.append({
            "question_id": qid,
            "category": row["category"],
            "source_row_sha256": sha(canonical(row)),
            "prompt_tokens": len(ids),
            "prompt_sha256": sha(json.dumps(ids, separators=(",", ":")).encode()),
        })
    chosen = _select(inventory, per_category)
    cases = []
    for meta in sorted(inventory, key=lambda x: (x["category"], x["question_id"])):
        if meta["question_id"] not in chosen:
            continue
        row = by_id[meta["question_id"]]
        cases.append(dict(
            meta,
            case_id="mmlu-pro-%05d" % row["question_id"],
            kind="mmlu-pro",
            prompt=prompts[meta["question_id"]],
            answer=row["answer"],
            option_count=len([o for o in row["options"] if o != "N/A"]),
            max_tokens=max_output,
            ignore_eos=False,
        ))
    return inventory, cases


def write_plan(output, inventory: List[dict], cases: List[dict], extra: Optional[dict] = None,
               per_category: Optional[int] = None, max_output: int = MAX_OUTPUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)  # never write into an existing plan
    (out / "cases.json").write_text(_dump(cases))
    (out / "tasks.json").write_text(_dump([{k: v for k, v in c.items() if k != "prompt"} for c in cases]))
    with gzip.GzipFile(filename=str(out / "inventory.jsonl.gz"), mode="wb", mtime=0) as f:
        f.write(b"".join(canonical(row) + b"\n" for row in inventory))
    contract = {
        "contract_version": CONTRACT_VERSION,
        "dataset": "mmlu-pro",
        "dataset_revision": DATASET_REVISION,
        "data_sha256": DATA_HASHES,
        "reference_revision": REFERENCE_REVISION,
        "reference_script_sha256": REFERENCE_SCRIPT_SHA256,
        "prompt_template_sha256": INITIAL_PROMPT_SHA256,
        "full_test_count": len(inventory),
        "tasks": len(cases),
        "selection": "all test tasks" if per_category is None
        else "first %d test IDs per category by question_id" % per_category,
        "scorer": "reference three-stage A-J extraction; invalid/out-of-range/transport failures count incorrect; "
                  "no random guessing",
        "scorer_sha256": sha256_file(scoring.__file__),
        "prompt_builder_sha256": sha256_file(prompt_mod.__file__),
        "generation": {"temperature": 0, "seed": 17, "thinking": False, "max_tokens": max_output, "stop": None,
                       "retries": 0, "concurrency": 1, "tools": [], "truncation": "reject"},
        "prompt_contract": "reference five-shot CoT text wrapped in the model chat template, thinking disabled; "
                           "all five shots kept; no 'Question:' stop string; over-long prompts rejected",
        "files_sha256": {name: sha256_file(out / name) for name in PLAN_FILES},
    }
    contract.update(extra or {})
    (out / "contract.json").write_text(_dump(contract))
    return contract


def _read_parquet_rows(path: Path, digest: str) -> List[dict]:
    import pyarrow.parquet as pq  # optional dependency

    if sha256_file(path) != digest:
        raise ValueError("SHA-256 mismatch: %s" % path)
    return pq.read_table(path).to_pylist()


def load_assets(assets) -> Tuple[List[dict], List[dict]]:
    """Read the pinned parquet files; refuses anything whose SHA-256 differs from the pinned revision."""
    assets = Path(assets)
    test = _read_parquet_rows(assets / "test-00000-of-00001.parquet", DATA_HASHES["test"])
    val = _read_parquet_rows(assets / "validation-00000-of-00001.parquet", DATA_HASHES["validation"])
    if len(test) != TEST_ROWS or len(val) != VALIDATION_ROWS:
        raise ValueError("Unexpected pinned dataset cardinality")
    return test, val


def load_instruction(reference) -> str:
    """The reference initial prompt, from a checkout whose two files must match the pinned SHA-256."""
    ref = Path(reference)
    if sha256_file(ref / "evaluate_from_local.py") != REFERENCE_SCRIPT_SHA256:
        raise ValueError("SHA-256 mismatch: reference evaluate_from_local.py (wrong checkout revision?)")
    prompt_file = ref / "cot_prompt_lib" / "initial_prompt.txt"
    if sha256_file(prompt_file) != INITIAL_PROMPT_SHA256:
        raise ValueError("SHA-256 mismatch: reference initial_prompt.txt")
    return prompt_file.read_text()


def prepare(assets, model_path, reference, output, runtime_contract: Optional[str] = None,
            per_category: Optional[int] = None) -> dict:
    """Freeze a plan from the pinned dataset files. Full test set unless per_category is given."""
    from transformers import AutoTokenizer  # optional dependency

    model_path = Path(model_path)
    test, val = load_assets(assets)
    instruction = load_instruction(reference)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    inventory, cases = build_plan(test, val, instruction, lambda t: prompt_mod.encode_prompt(tokenizer, t),
                                  per_category=per_category)
    extra = {"model_files_sha256": {n: sha256_file(model_path / n) for n in
                                    ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
                                    if (model_path / n).is_file()}}
    if runtime_contract:
        rc = Path(runtime_contract)
        extra["runtime"] = json.loads(rc.read_text())
        extra["runtime_contract_sha256"] = sha256_file(rc)
    return write_plan(output, inventory, cases, extra, per_category)


def load_plan(plan_dir, allow_legacy: bool = False) -> Tuple[dict, List[dict]]:
    """Load and verify a plan: scorer unchanged since freezing, file hashes, task manifest, prompt identity.

    Plans made by the upstream adapter (contract_version pyramidkv-mmlu-pro-subset-v1) have no scorer hash of
    ours; they load only with allow_legacy=True, in which case the scorer check is skipped and
    contract["_legacy_plan"] is set so callers can say so in their reports.
    """
    p = Path(plan_dir)
    contract = json.loads((p / "contract.json").read_text())
    legacy = contract.get("contract_version") == LEGACY_CONTRACT_VERSION
    if legacy:
        if not allow_legacy:
            raise ValueError("plan was made by the upstream adapter; pass allow_legacy=True to load it "
                             "(the scorer integrity check cannot be applied)")
        contract["_legacy_plan"] = True
    elif contract.get("scorer_sha256") != sha256_file(scoring.__file__):
        raise ValueError("Scorer changed after contract freeze")
    for name, digest in contract["files_sha256"].items():
        if sha256_file(p / name) != digest:
            raise ValueError("SHA-256 mismatch: %s" % name)
    cases = json.loads((p / "cases.json").read_text())
    if len(cases) != contract["tasks"] or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Task manifest mismatch")
    for row in cases:
        if len(row["prompt"]) != row["prompt_tokens"] or \
                sha(json.dumps(row["prompt"], separators=(",", ":")).encode()) != row["prompt_sha256"]:
            raise ValueError("Prompt IDs differ from frozen manifest")
    return contract, cases
