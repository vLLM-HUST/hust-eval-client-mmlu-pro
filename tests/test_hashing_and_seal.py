import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hust_eval_mmlu_pro import hashing, seal


class HashingTests(unittest.TestCase):
    def test_sha256_file_known_value(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_bytes(b"abc")
            self.assertEqual(hashing.sha256_file(p), hashlib.sha256(b"abc").hexdigest())

    def test_hash_tree_exclusions_and_bundle_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "results").mkdir()
            (r / "results" / "t.xlsx").write_text("table")
            (r / "a").mkdir()
            (r / "a" / "x.txt").write_text("x")
            (r / "MANIFEST.json").write_text("{}")
            files = hashing.hash_tree(r, exclude_top=("results",), exclude_files=("MANIFEST.json",))
            self.assertEqual(sorted(files), ["a/x.txt"])
            self.assertEqual(hashing.bundle_hash(files), hashing.bundle_hash(dict(reversed(list(files.items())))))

    def test_hash_model_dir_small_only_and_expected(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "model"
            m.mkdir()
            (m / "config.json").write_text("{}")
            (m / "model-00001.safetensors").write_bytes(b"weights")
            exp = Path(d) / "exp.json"
            exp.write_text(json.dumps({"files": [
                {"file": "config.json", "sha256": hashlib.sha256(b"{}").hexdigest()},
                {"file": "model-00001.safetensors", "sha256": hashlib.sha256(b"weights").hexdigest()}]}))
            small = hashing.hash_model_dir(m, small_only=True, expected_file=str(exp))
            self.assertEqual(small["compare_with_expected"]["model-00001.safetensors"], "not_hashed")
            self.assertFalse(small["all_expected_matched"])
            full = hashing.hash_model_dir(m, expected_file=str(exp))
            self.assertTrue(full["all_expected_matched"])
            exp.write_text(json.dumps({"files": [{"file": "config.json", "sha256": "0" * 64}]}))
            self.assertEqual(hashing.hash_model_dir(m, expected_file=str(exp))["compare_with_expected"]["config.json"], "MISMATCH")


class SealTests(unittest.TestCase):
    def _run_dir(self, d):
        r = Path(d) / "run1"
        (r / "receipt").mkdir(parents=True)
        (r / "results").mkdir()
        (r / "outputs").mkdir()
        (r / "profile.json").write_text("{}")
        (r / "receipt" / "effective.json").write_text('{"ok": true}')
        (r / "receipt" / "startup.log").write_text("log")
        (r / "outputs" / "a.json").write_text("x")
        (r / "results" / "results.xlsx").write_text("table")
        return r

    def test_full_flow_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._run_dir(d)
            bundle = seal.seal_evidence(r)
            res = seal.seal_final(r, r / "results" / "results.xlsx", ledger=str(Path(d) / "ledger.jsonl"))
            self.assertEqual(res["table_sha256"], hashlib.sha256(b"table").hexdigest())
            self.assertEqual(json.loads((Path(d) / "ledger.jsonl").read_text())["manifest_sha256"], res["manifest_sha256"])
            self.assertTrue(seal.seal_check(r)["ok"])
            self.assertEqual(json.loads((r / "MANIFEST.json").read_text())["evidence_bundle_sha256"], bundle["evidence_bundle_sha256"])
            (r / "outputs" / "a.json").write_text("changed")
            self.assertEqual(seal.seal_check(r)["evidence_files_differ"], ["outputs/a.json"])
            (r / "outputs" / "a.json").write_text("x")
            (r / "results" / "results.xlsx").write_text("edited table")
            self.assertFalse(seal.seal_check(r)["table_matches_manifest"])

    def test_sealed_once_and_final_needs_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._run_dir(d)
            with self.assertRaises(seal.SealError):
                seal.seal_final(r, r / "results" / "results.xlsx")
            seal.seal_evidence(r)
            with self.assertRaises(seal.SealError):
                seal.seal_evidence(r)

    def test_refuses_failed_run_and_changed_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._run_dir(d)
            (r / "receipt" / "FAILED.txt").write_text("boom")
            with self.assertRaises(seal.SealError):
                seal.seal_evidence(r)
            (r / "receipt" / "FAILED.txt").unlink()
            seal.seal_evidence(r)
            (r / "outputs" / "a.json").write_text("late edit")
            with self.assertRaises(seal.SealError):
                seal.seal_final(r, r / "results" / "results.xlsx")


if __name__ == "__main__":
    unittest.main()
