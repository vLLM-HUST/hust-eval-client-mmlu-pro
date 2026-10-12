"""MMLU-Pro prompt construction.

Behaviour follows the reference runner of TIGER-AI-Lab/MMLU-Pro (Apache-2.0), `evaluate_from_local.py` at
revision f418b116db00b065c2aea046518d8fcf74d39872 (its text layout and few-shot CoT format). This is a
reimplementation, not a copy; `tests/` and the equivalence check in the README pin it to the original's output.

Differences from the reference runner, on purpose:
  * the five validation examples are always kept (the reference drops shots to fit a 4096-token window),
  * the text is wrapped in the model's chat template with thinking disabled and turned into token IDs,
  * over-long prompts are rejected, never truncated.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List

CHOICE_LETTERS = "ABCDEFGHIJ"
SHOTS = 5


def format_example(row: dict, answered: bool) -> str:
    """One question block. `answered=True` appends the reference chain-of-thought (few-shot example);
    `answered=False` ends with the cue the model must continue from."""
    options = [o for o in row["options"] if o != "N/A"]
    index = row["answer_index"]
    # filtering "N/A" options must not move the gold answer
    if not 0 <= index < len(options) <= len(CHOICE_LETTERS) or options[index] != row["options"][index]:
        raise ValueError("Option filtering would change the frozen answer index")
    if row["answer"] != CHOICE_LETTERS[index]:
        raise ValueError("Answer letter/index mismatch")
    text = "Question:\n" + row["question"] + "\nOptions:\n"
    text += "".join("%s. %s\n" % (CHOICE_LETTERS[i], o) for i, o in enumerate(options))
    if answered:
        cot = row["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        return text + cot + "\n\n"
    return text + "Answer: Let's think step by step."


def build_prefixes(instruction: str, validation_rows: Iterable[dict], subjects: Iterable[str]) -> Dict[str, str]:
    """Per-subject few-shot prefix: the instruction (with the subject filled in) plus the five answered examples
    from the validation split, in source order."""
    validation_rows = list(validation_rows)
    prefixes = {}
    for subject in subjects:
        shots = [r for r in validation_rows if r["category"] == subject]
        if len(shots) != SHOTS:
            raise ValueError("Expected exactly %d validation examples for %r, found %d" % (SHOTS, subject, len(shots)))
        prefixes[subject] = instruction.replace("{$}", subject) + "\n" + "".join(
            format_example(r, answered=True) for r in shots)
    return prefixes


def user_text(prefixes: Dict[str, str], row: dict) -> str:
    """The full user message for one test question."""
    return prefixes[row["category"]] + format_example(row, answered=False)


def encode_prompt(tokenizer, text: str) -> List[int]:
    """Chat-template the user message (thinking disabled, generation prompt added) and return real token IDs."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not isinstance(ids, list) or len(ids) < 16 or not all(type(x) is int for x in ids):
        raise ValueError("Expected actual prompt token IDs")
    return ids


Encoder = Callable[[str], List[int]]
