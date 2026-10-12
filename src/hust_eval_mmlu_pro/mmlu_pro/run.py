"""Send a frozen MMLU-Pro plan to an already-running OpenAI-compatible server and score the answers.

This is a pure client: streaming POST /v1/completions with the frozen token IDs of a plan, no retries (a failed
request stays in the denominator and is scored incorrect), scoring with `scoring.score_results`.

Deviation recorded in run.<ts>.json: concurrency is configurable, while the frozen plan declares 1.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from .. import __version__
from ..hashing import sha256_file
from . import plan as plan_mod
from . import scoring

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def one_request(base_url: str, model: str, case: dict, gen: dict, timeout: int, api_key: Optional[str] = None) -> dict:
    body = {
        "model": model, "prompt": case["prompt"], "max_tokens": case["max_tokens"],
        "temperature": gen["temperature"], "seed": gen["seed"], "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": case["ignore_eos"],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/completions", json.dumps(body).encode(), headers)
    rec = {"case_id": case["case_id"], "prompt_sha256": case["prompt_sha256"], "prompt_tokens": case["prompt_tokens"],
           "max_tokens": case["max_tokens"], "ignore_eos": case["ignore_eos"], "success": False,
           "output": "", "finish_reason": None, "error": None}
    t0 = time.monotonic()
    t_first = None
    parts = []
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    rec["usage"] = chunk["usage"]
                for ch in chunk.get("choices", []):
                    if ch.get("text"):
                        if t_first is None:
                            t_first = time.monotonic()
                        parts.append(ch["text"])
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
        rec["output"] = "".join(parts)
        rec["success"] = rec["finish_reason"] is not None
        if not rec["success"]:
            rec["error"] = "stream ended without finish_reason"
    except Exception as e:  # noqa: BLE001 - recorded, scored incorrect, never retried
        rec["output"] = "".join(parts)
        rec["error"] = repr(e)
    t1 = time.monotonic()
    rec["e2e_seconds"] = t1 - t0
    rec["ttft_seconds"] = (t_first - t0) if t_first else None
    return rec


def run(plan: str, output: str, base_url: str = "http://127.0.0.1:8000",
        served_model: str = "model", concurrency: int = 16, timeout: int = 1800, limit: Optional[int] = None,
        resume: bool = False, api_key_env: Optional[str] = None, allow_legacy: bool = False, log=print) -> dict:
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise RuntimeError("environment variable %s is not set" % api_key_env)

    contract, cases = plan_mod.load_plan(Path(plan), allow_legacy=allow_legacy)  # scorer, file and prompt hashes
    gen = contract["generation"]
    if limit:
        cases = cases[:limit]
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError("output exists; use a new directory (or resume=True to continue the same run)")
    (out / "requests").mkdir(parents=True, exist_ok=True)

    meta = {
        "tool": "hust-eval-mmlu-pro %s" % __version__,
        "runner_sha256": sha256_file(__file__),
        "scorer_sha256": sha256_file(scoring.__file__),
        "legacy_plan": bool(contract.get("_legacy_plan")),
        "contract_sha256": sha256_file(Path(plan) / "contract.json"),
        "concurrency": concurrency, "upstream_concurrency": gen.get("concurrency"), "generation": gen,
        "base_url": base_url, "served_model": served_model, "n_cases": len(cases), "limit": limit,
        "resume": resume, "started_unix": time.time(),
        "deviations": ["concurrency differs from contract.generation.concurrency"],
    }
    if limit:
        meta["WARNING"] = "smoke test only - not a score"
    (out / ("run.%d.json" % int(time.time()))).write_text(json.dumps(meta, indent=1))

    done, lock, todo = {}, threading.Lock(), []
    for c in cases:
        f = out / "requests" / (c["case_id"] + ".json")
        if resume and f.exists():
            r = json.loads(f.read_text())
            if r.get("success"):
                done[c["case_id"]] = r
                continue
        todo.append(c)
    log("cases=%d already_done=%d to_run=%d concurrency=%d" % (len(cases), len(done), len(todo), concurrency))

    t_start = time.monotonic()

    def work(c):
        r = one_request(base_url, served_model, c, gen, timeout, api_key)
        (out / "requests" / (c["case_id"] + ".json")).write_text(json.dumps(r, ensure_ascii=False))
        with lock:
            done[c["case_id"]] = r
            n = len(done)
            if n % 100 == 0 or n == len(cases):
                ok = sum(1 for x in done.values() if x["success"])
                log("%d/%d ok=%d elapsed=%.0fs" % (n, len(cases), ok, time.monotonic() - t_start))

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(work, todo))
    wall = time.monotonic() - t_start

    results = [done[c["case_id"]] for c in cases]  # plan order
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False))
    scores = scoring.score_results(results, cases)
    (out / "score.json").write_text(json.dumps(scores, indent=1, ensure_ascii=False))
    ok = [r for r in results if r["success"]]
    summary = {
        "cases": len(cases), "correct": scores["correct"], "accuracy": scores["accuracy"],
        "invalid_answers": scores["invalid_answers"], "transport_failures": scores["transport_failures"],
        "length_limited": scores["length_limited"], "wall_seconds_this_invocation": wall,
        "mean_ttft_s": sum(r["ttft_seconds"] for r in ok if r["ttft_seconds"]) / max(1, len(ok)),
        "mean_e2e_s": sum(r["e2e_seconds"] for r in ok) / max(1, len(ok)),
        "smoke_only": bool(limit),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
