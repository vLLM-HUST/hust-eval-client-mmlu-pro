import gzip
import json
import tempfile
import unittest
from pathlib import Path

from hust_eval_mmlu_pro.hashing import sha256_file
from hust_eval_mmlu_pro.mmlu_pro import plan as plan_mod
from hust_eval_mmlu_pro.mmlu_pro import prompt, scoring

from helpers import INSTRUCTION, fake_encode, make_plan, synthetic_rows


class ExtractionTests(unittest.TestCase):
    def test_stage1_answer_is(self):
        self.assertEqual(scoring.extract_answer("so the answer is (C)."), "C")
        self.assertEqual(scoring.extract_answer("The answer is B"), "B")
        self.assertEqual(scoring.extraction_stage("the answer is (C)"), 1)

    def test_stage2_answer_colon(self):
        self.assertEqual(scoring.extract_answer("Final.\nAnswer: D"), "D")
        self.assertEqual(scoring.extraction_stage("Answer: D"), 2)

    def test_stage3_last_standalone_letter_picks_text_too(self):
        self.assertEqual(scoring.extract_answer("maybe B is right, then perhaps E"), "E")
        # documented weakness: an English article is a standalone letter
        self.assertEqual(scoring.extract_answer("A mutation changes things"), "A")
        self.assertEqual(scoring.extraction_stage("maybe B then E"), 3)

    def test_nothing_extractable(self):
        self.assertIsNone(scoring.extract_answer("no capital letters at all here"))
        self.assertEqual(scoring.extraction_stage("no capital letters at all here"), 0)


def _case(i, answer="A", options=4):
    return {"case_id": "c%d" % i, "category": "law", "answer": answer, "option_count": options,
            "prompt_sha256": "h%d" % i, "prompt_tokens": 10, "max_tokens": 64, "ignore_eos": False}


def _result(i, output="The answer is (A)", success=True, finish="stop"):
    return {"case_id": "c%d" % i, "prompt_sha256": "h%d" % i, "prompt_tokens": 10, "max_tokens": 64,
            "ignore_eos": False, "success": success, "output": output, "finish_reason": finish}


class ScoringTests(unittest.TestCase):
    def test_correct_wrong_invalid_and_failure(self):
        cases = [_case(0, "A"), _case(1, "B"), _case(2, "A", 4), _case(3, "A")]
        results = [_result(0), _result(1), _result(2, "The answer is (F)"), _result(3, "", success=False, finish=None)]
        s = scoring.score_results(results, cases)
        self.assertEqual((s["count"], s["correct"]), (4, 1))
        self.assertEqual(s["invalid_answers"], 2)          # 'F' with 4 options, and the failed request
        self.assertEqual(s["transport_failures"], 1)
        self.assertEqual(s["accuracy"], 0.25)

    def test_no_random_guess_when_nothing_extracted(self):
        s = scoring.score_results([_result(0, "nothing to see")], [_case(0)])
        self.assertEqual(s["correct"], 0)
        self.assertEqual(s["invalid_answers"], 1)

    def test_length_limited_counted_and_can_still_be_correct(self):
        s = scoring.score_results([_result(0, "so the answer is (A)", finish="length")], [_case(0)])
        self.assertEqual(s["length_limited"], 1)
        self.assertEqual(s["correct"], 1)

    def test_missing_or_mismatched_results_are_rejected(self):
        with self.assertRaises(ValueError):
            scoring.score_results([_result(0)], [_case(0), _case(1)])
        bad = _result(0)
        bad["prompt_sha256"] = "other"
        with self.assertRaises(ValueError):
            scoring.score_results([bad], [_case(0)])
        with self.assertRaises(ValueError):
            scoring.score_results([], [])


class PromptTests(unittest.TestCase):
    def test_format_example_layout(self):
        test, _ = synthetic_rows()
        text = prompt.format_example(test[0], answered=False)
        self.assertTrue(text.startswith("Question:\nQuestion number 0?\nOptions:\nA. opt a\nB. opt b\n"))
        self.assertTrue(text.endswith("Answer: Let's think step by step."))
        shot = prompt.format_example(test[0], answered=True)
        self.assertIn("Answer: Let's think step by step. Reasoning.", shot)  # "A: " cue rewritten
        self.assertTrue(shot.endswith("\n\n"))

    def test_na_options_are_dropped_but_must_not_move_the_answer(self):
        test, _ = synthetic_rows()
        row = dict(test[0], options=["x", "N/A", "z"], answer_index=0, answer="A")
        self.assertIn("B. z", prompt.format_example(row, answered=False))
        moved = dict(test[0], options=["x", "N/A", "z"], answer_index=2, answer="C")
        with self.assertRaises(ValueError):
            prompt.format_example(moved, answered=False)

    def test_letter_index_mismatch_rejected(self):
        test, _ = synthetic_rows()
        with self.assertRaises(ValueError):
            prompt.format_example(dict(test[0], answer="C"), answered=False)

    def test_prefix_needs_exactly_five_shots(self):
        _, val = synthetic_rows()
        with self.assertRaises(ValueError):
            prompt.build_prefixes(INSTRUCTION, val[:-1], ["law", "math"])
        pre = prompt.build_prefixes(INSTRUCTION, val, ["law", "math"])
        self.assertTrue(pre["law"].startswith("The following are multiple choice questions about law."))
        self.assertEqual(pre["law"].count("Question:\n"), 5)


class PlanTests(unittest.TestCase):
    def test_build_plan_shape_order_and_identity(self):
        test, val = synthetic_rows(6)
        inventory, cases = plan_mod.build_plan(test[::-1], val, INSTRUCTION, fake_encode)
        self.assertEqual(len(inventory), 6)
        self.assertEqual([c["case_id"] for c in cases],
                         ["mmlu-pro-%05d" % i for i in range(6)])  # sorted by (category, question_id)
        c = cases[0]
        self.assertEqual(list(c)[:5], ["question_id", "category", "source_row_sha256", "prompt_tokens", "prompt_sha256"])
        self.assertEqual(c["max_tokens"], 2048)
        self.assertFalse(c["ignore_eos"])
        self.assertEqual(c["option_count"], 4)

    def test_per_category_subset(self):
        test, val = synthetic_rows(6)
        _, cases = plan_mod.build_plan(test, val, INSTRUCTION, fake_encode, per_category=2)
        self.assertEqual(len(cases), 4)
        with self.assertRaises(ValueError):
            plan_mod.build_plan(test, val, INSTRUCTION, fake_encode, per_category=0)

    def test_duplicates_and_overlong_prompts_rejected(self):
        test, val = synthetic_rows(4)
        with self.assertRaises(ValueError):
            plan_mod.build_plan(test + [test[0]], val, INSTRUCTION, fake_encode)
        with self.assertRaises(ValueError):
            plan_mod.build_plan(test, val, INSTRUCTION, lambda t: [1] * 40000)

    def test_write_is_deterministic_and_refuses_existing_dir(self):
        with tempfile.TemporaryDirectory() as d:
            a = make_plan(Path(d))
            test, val = synthetic_rows()
            inv, cases = plan_mod.build_plan(test, val, INSTRUCTION, fake_encode)
            plan_mod.write_plan(Path(d) / "second", inv, cases)
            for name in plan_mod.PLAN_FILES:
                self.assertEqual(sha256_file(a / name), sha256_file(Path(d) / "second" / name), name)
            with self.assertRaises(FileExistsError):
                plan_mod.write_plan(a, inv, cases)
            first_line = gzip.open(a / "inventory.jsonl.gz").readline()
            self.assertEqual(json.loads(first_line)["question_id"], 0)

    def test_tasks_json_is_cases_without_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_plan(Path(d))
            cases = json.loads((p / "cases.json").read_text())
            tasks = json.loads((p / "tasks.json").read_text())
            self.assertTrue(all("prompt" in c for c in cases))
            self.assertEqual(tasks, [{k: v for k, v in c.items() if k != "prompt"} for c in cases])

    def test_load_plan_verifies_everything(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_plan(Path(d))
            contract, cases = plan_mod.load_plan(p)
            self.assertEqual(contract["tasks"], len(cases))
            # changed scorer hash -> refused
            c = json.loads((p / "contract.json").read_text())
            c["scorer_sha256"] = "0" * 64
            (p / "contract.json").write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, "Scorer changed"):
                plan_mod.load_plan(p)

    def test_load_plan_detects_tampered_files_and_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_plan(Path(d))
            cases = json.loads((p / "cases.json").read_text())
            cases[0]["prompt"][0] += 1
            (p / "cases.json").write_text(json.dumps(cases))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                plan_mod.load_plan(p)

    def test_legacy_plans_need_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_plan(Path(d))
            c = json.loads((p / "contract.json").read_text())
            c["contract_version"] = plan_mod.LEGACY_CONTRACT_VERSION
            c.pop("scorer_sha256")
            (p / "contract.json").write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                plan_mod.load_plan(p)
            contract, _ = plan_mod.load_plan(p, allow_legacy=True)
            self.assertTrue(contract["_legacy_plan"])


if __name__ == "__main__":
    unittest.main()
