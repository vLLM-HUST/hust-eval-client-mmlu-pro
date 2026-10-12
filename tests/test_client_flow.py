import json
import re
import tempfile
import unittest
from pathlib import Path

from hust_eval_mmlu_pro import receipt
from hust_eval_mmlu_pro.mmlu_pro import bench, run, summary

from helpers import StubServer, make_plan

PROFILE = Path(__file__).resolve().parents[1] / "examples" / "profile.b0_35b.json"


class RunTests(unittest.TestCase):
    def test_run_scores_and_records_deviations(self):
        with tempfile.TemporaryDirectory() as d, StubServer() as srv:
            plan = make_plan(Path(d))
            out = Path(d) / "out"
            s = run.run(str(plan), str(out), srv.url, "stub-model", concurrency=3, log=lambda m: None)
            self.assertEqual(s["cases"], 4)
            self.assertEqual(s["correct"], 2)  # stub always says A; gold is A,B,A,B
            self.assertEqual(s["transport_failures"], 0)
            self.assertEqual(srv.last_body["temperature"], 0)
            self.assertEqual(srv.last_body["seed"], 17)
            self.assertTrue(srv.last_body["stream"])
            meta = json.loads(next(out.glob("run.*.json")).read_text())
            self.assertEqual(meta["concurrency"], 3)
            self.assertEqual(meta["upstream_concurrency"], 1)
            self.assertTrue(meta["deviations"])
            self.assertEqual(len(meta["scorer_sha256"]), 64)
            self.assertFalse(meta["legacy_plan"])
            with self.assertRaises(FileExistsError):
                run.run(str(plan), str(out), srv.url, "stub-model", log=lambda m: None)

    def test_failed_requests_are_kept_and_scored_incorrect(self):
        with tempfile.TemporaryDirectory() as d:
            plan = make_plan(Path(d))
            out = Path(d) / "out"
            s = run.run(str(plan), str(out), "http://127.0.0.1:1", "x", timeout=2, log=lambda m: None)
            self.assertEqual(s["transport_failures"], 4)
            self.assertEqual(s["correct"], 0)
            self.assertEqual(len(list((out / "requests").glob("*.json"))), 4)

    def test_resume_reuses_successful_requests(self):
        with tempfile.TemporaryDirectory() as d, StubServer() as srv:
            plan = make_plan(Path(d))
            out = Path(d) / "out"
            run.run(str(plan), str(out), srv.url, "stub-model", limit=2, log=lambda m: None)
            msgs = []
            run.run(str(plan), str(out), srv.url, "stub-model", resume=True, log=msgs.append)
            self.assertIn("already_done=2", msgs[0])

    def test_api_key_env_must_be_set(self):
        with tempfile.TemporaryDirectory() as d:
            plan = make_plan(Path(d))
            with self.assertRaises(RuntimeError):
                run.run(str(plan), str(Path(d) / "o"), "http://127.0.0.1:1", "x",
                        api_key_env="HUST_EVAL_SURELY_UNSET_VAR", log=lambda m: None)

    def test_summary_flags_partial_runs(self):
        with tempfile.TemporaryDirectory() as d, StubServer() as srv:
            plan = make_plan(Path(d))
            out = Path(d) / "out"
            run.run(str(plan), str(out), srv.url, "stub-model", limit=2, log=lambda m: None)
            rep = summary.build_report(str(plan), str(out))
            self.assertFalse(rep["complete"])
            self.assertIn("PARTIAL", rep["WARNING"])
            self.assertEqual(rep["bench"], "not run yet")


class ReceiptAndBenchTests(unittest.TestCase):
    def test_receipt_records_models_and_compares_profile(self):
        with tempfile.TemporaryDirectory() as d, StubServer() as srv:
            log = Path(d) / "startup.log"
            log.write_text("max_seq_len=32768, tensor_parallel_size=2\n")
            rd = Path(d) / "run"
            res = receipt.capture(rd, srv.url, profile=str(PROFILE), startup_log=str(log))
            self.assertEqual(res["models"]["data"][0]["max_model_len"], 32768)
            rc = rd / "receipt"
            for name in ("models.json", "startup.log", "effective.json", "client.json", "profile.sha256"):
                self.assertTrue((rc / name).exists(), name)
            eff = json.loads((rc / "effective.json").read_text())
            self.assertEqual(eff["fields"]["tensor_parallel_size"]["status"], "log_only")
            self.assertEqual(json.loads((rc / "client.json").read_text())["image_id"], "NOT_PROVIDED")
            with self.assertRaises(FileExistsError):
                receipt.capture(rd, srv.url)

    def test_bench_dry_run_writes_command_and_hashes_only(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "m.jsonl"
            data.write_text('{"prompt": "a"}\n{"prompt": "b"}\n')
            rd = Path(d) / "run"
            res = bench.bench(rd, "mmlu-pro", str(data), "http://h:1", "m", "/tok", dry_run=True)
            self.assertEqual(res["dataset_rows"], 2)
            self.assertIn("--request-rate", res["command"])
            self.assertTrue((rd / "inputs" / "mmlu-pro.materialized.sha256").exists())
            self.assertFalse((rd / "outputs" / "mmlu-pro.json").exists())


class ClientOnlyGuard(unittest.TestCase):
    """The package must stay a client: it may call `vllm bench serve`, never `vllm serve`."""

    def test_no_server_launching_code(self):
        src = Path(__file__).resolve().parents[1] / "src" / "hust_eval_mmlu_pro"
        for p in src.rglob("*.py"):
            text = p.read_text()
            self.assertNotIn("Popen", text, "%s starts background processes" % p.name)
            self.assertFalse(re.search(r'\[\s*"vllm"\s*,\s*"serve"', text), "%s launches a server" % p.name)
            self.assertNotIn("start_new_session", text, "%s detaches a process" % p.name)
            self.assertNotIn("api_server", text, "%s references the server entrypoint" % p.name)


if __name__ == "__main__":
    unittest.main()
