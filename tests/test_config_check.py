import json
import tempfile
import unittest
from pathlib import Path

from hust_eval_mmlu_pro import config_check

PROFILE = json.loads((Path(__file__).resolve().parents[1] / "examples" / "profile.b0_35b.json").read_text())
ARGV = PROFILE["launcher"] + [PROFILE["model"]["path"]] + PROFILE["serve_args"]

# a log in the formats seen on a real vLLM 0.28.1 / vllm-ascend start (dict-style args + engine config line)
LOG = (
    "non-default args: {'dtype': 'bfloat16', 'tensor_parallel_size': 2, 'enable_expert_parallel': True, "
    "'gpu_memory_utilization': 0.85, 'max_num_seqs': 16, 'compilation_config': {'mode': <CompilationMode.VLLM_COMPILE: 3>}}\n"
    "Initializing a V1 LLM engine with config: speculative_config=None, dtype=torch.bfloat16, max_seq_len=32768, "
    "tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, "
    "disable_custom_all_reduce=True, enforce_eager=False, enable_prefix_caching=False, enable_chunked_prefill=True, "
    "max_num_batched_tokens=8192, cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>, "
    "cudagraph_capture_sizes=[1, 2, 4, 8, 16]\n"
)
MODELS = {"data": [{"id": "qwen3.5-35b", "max_model_len": 32768, "root": "/models/x"}]}


class ConfigCheckTests(unittest.TestCase):
    def test_everything_verified_when_argv_and_log_agree(self):
        r = config_check.verify(PROFILE, ARGV, LOG, MODELS)
        self.assertTrue(r["ok"], r["mismatches"])
        self.assertEqual(r["unverified"], [])
        self.assertEqual(len(r["verified"]), len(PROFILE["expect"]))

    def test_dcp_unset_equals_engine_default_one(self):
        r = config_check.verify(PROFILE, ARGV, LOG, MODELS)
        self.assertEqual(r["fields"]["decode_context_parallel_size"]["status"], "verified")

    def test_log_only_when_argv_not_provided(self):
        r = config_check.verify(PROFILE, None, LOG, MODELS)
        self.assertTrue(r["ok"])
        self.assertEqual(r["verified"], [])
        self.assertIn("tensor_parallel_size", r["log_only"])

    def test_tampering_is_detected_in_log_and_api(self):
        bad = LOG.replace("tensor_parallel_size=2", "tensor_parallel_size=1")
        r = config_check.verify(PROFILE, ARGV, bad, {"data": [{"id": "qwen3.5-35b", "max_model_len": 65536}]})
        self.assertFalse(r["ok"])
        self.assertIn("tensor_parallel_size", r["mismatches"])
        self.assertIn("api.max_model_len", r["mismatches"])

    def test_wrong_argv_is_detected(self):
        argv = [("1" if a == "16" else a) for a in ARGV]  # max-num-seqs 16 -> 1 (also capture sizes); still a mismatch
        r = config_check.verify(PROFILE, argv, LOG, MODELS)
        self.assertFalse(r["ok"])

    def test_no_sources_means_unverified_not_verified(self):
        r = config_check.verify(PROFILE, None, None, None)
        self.assertTrue(r["ok"])  # nothing disagreed ...
        self.assertEqual(r["verified"], [])  # ... but nothing was proven either
        self.assertEqual(len(r["unverified"]), len(PROFILE["expect"]))

    def test_engine_default_is_not_an_explicit_setting(self):
        argv = [a for a in ARGV if a != "--no-enable-prefix-caching"]
        r = config_check.verify(PROFILE, argv, None, None)
        self.assertEqual(r["fields"]["enable_prefix_caching"]["status"], "mismatch")

    def test_load_argv_formats(self):
        with tempfile.TemporaryDirectory() as d:
            nul = Path(d) / "cmdline"
            nul.write_bytes(b"vllm\x00serve\x00/m\x00--port\x00123\x00")
            js = Path(d) / "argv.json"
            js.write_text(json.dumps(["vllm", "serve", "--port", "123"]))
            self.assertEqual(config_check.load_argv(str(nul))[-2:], ["--port", "123"])
            self.assertEqual(config_check.load_argv(str(js))[-2:], ["--port", "123"])


if __name__ == "__main__":
    unittest.main()
