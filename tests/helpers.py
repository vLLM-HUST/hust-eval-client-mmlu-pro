"""Shared test fixtures: a small synthetic plan built with the real plan code, and a stub OpenAI-compatible server."""
import http.server
import json
import threading
from pathlib import Path

from hust_eval_mmlu_pro.mmlu_pro import plan as plan_mod


def _row(qid, category, gold_index):
    letters = "ABCD"
    return {"question_id": qid, "question": "Question number %d?" % qid, "options": ["opt a", "opt b", "opt c", "opt d"],
            "answer": letters[gold_index], "answer_index": gold_index, "category": category,
            "cot_content": "A: Let's think step by step. Reasoning. The answer is (%s)." % letters[gold_index]}


def synthetic_rows(n=4):
    """n test rows split over two categories (gold answers alternate A,B,A,B...) plus 5 validation shots each."""
    cats = ("law", "math")
    test = [_row(i, cats[0] if i < n // 2 else cats[1], i % 2) for i in range(n)]
    val = [_row(1000 + 10 * ci + k, c, k % 4) for ci, c in enumerate(cats) for k in range(5)]
    return test, val


def fake_encode(text):
    return [ord(ch) % 5000 for ch in text]


INSTRUCTION = "The following are multiple choice questions about {$}. Think step by step.\n"


def make_plan(root: Path, n: int = 4, per_category=None):
    test, val = synthetic_rows(n)
    inventory, cases = plan_mod.build_plan(test, val, INSTRUCTION, fake_encode, per_category=per_category)
    out = Path(root) / "plan"
    plan_mod.write_plan(out, inventory, cases, per_category=per_category)
    return out


class StubServer:
    """Always answers 'answer is (A)' with a streamed completion; also serves /v1/models and /metrics."""

    def __init__(self):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/v1/models"):
                    body = json.dumps({"data": [{"id": "stub-model", "max_model_len": 32768, "root": "/models/x"}]}).encode()
                elif self.path.startswith("/metrics"):
                    body = b"# stub metrics\n"
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                outer.last_body = json.loads(self.rfile.read(n))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in ("Thinking. ", "The answer is (A)"):
                    self.wfile.write(("data: " + json.dumps({"choices": [{"text": piece, "finish_reason": None}]}) + "\n\n").encode())
                self.wfile.write(("data: " + json.dumps({"choices": [{"text": "", "finish_reason": "stop"}]}) + "\n\n").encode())
                self.wfile.write(("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 7}}) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.last_body = None
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
