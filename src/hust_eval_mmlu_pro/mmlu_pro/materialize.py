"""Make a `vllm bench serve --dataset-name custom` prompt file from a frozen MMLU-Pro plan.

Every emitted prompt is re-encoded exactly as when the plan was frozen and compared with the frozen token IDs;
any difference aborts. So the performance prompts are provably the accuracy prompts.

Note: `vllm bench serve --backend openai-chat` applies the chat template on the server with the server's default
thinking setting, while the accuracy run sends pre-rendered token IDs with thinking disabled. Output lengths may
therefore differ between the two runs; record that difference when you report both.

--sample N takes N evenly spaced cases in plan order (the plan is sorted by category, so "first N" would be one
category).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import plan as plan_mod
from . import prompt


def materialize(assets: str, reference: str, model_path: str, plan: str, output: str,
                sample: Optional[int] = None, allow_legacy: bool = False) -> int:
    from transformers import AutoTokenizer  # optional dependency

    out = Path(output)
    if out.exists():
        raise FileExistsError("refusing to overwrite %s" % output)
    test, val = plan_mod.load_assets(assets)
    instruction = plan_mod.load_instruction(reference)
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    prefixes = prompt.build_prefixes(instruction, val, sorted({r["category"] for r in test}))
    by_id = {r["question_id"]: r for r in test}

    _, cases = plan_mod.load_plan(plan, allow_legacy=allow_legacy)
    chosen = cases if not sample else [cases[i * len(cases) // sample] for i in range(sample)]
    with open(out, "w", encoding="utf8") as f:
        for c in chosen:
            row = by_id[int(c["case_id"].rsplit("-", 1)[1])]
            text = prompt.user_text(prefixes, row)
            if prompt.encode_prompt(tok, text) != c["prompt"]:
                f.close()
                out.unlink()
                raise ValueError("prompt mismatch for %s - aborted" % c["case_id"])
            f.write(json.dumps({"prompt": text}, ensure_ascii=False) + "\n")
    out.with_suffix(out.suffix + ".cases.json").write_text(json.dumps([c["case_id"] for c in chosen]))
    return len(chosen)
