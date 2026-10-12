"""hust-eval-mmlu-pro command line. A pure client: it never starts, stops or configures a server."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False))


def _cmd_hash_file(a):
    from .hashing import sha256_file
    for p in a.paths:
        print("%s  %s" % (sha256_file(p), p))


def _cmd_hash_model(a):
    import os
    from .hashing import hash_model_dir, sha256_file
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite " + a.out)
    res = hash_model_dir(a.model, a.small_only, a.expected,
                         progress=lambda n, h, s: print("%-60s %s %.0fs" % (n, h[:12], s), flush=True))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    if "compare_with_expected" in res:
        print("expected-compare:", res["compare_with_expected"])
    print("written", a.out, "sha256", sha256_file(a.out))


def _cmd_contract(a):
    from . import runtime_contract
    c = runtime_contract.build(a.profile, a.out, a.image_id)
    print("written", a.out)
    _print({k: c[k] for k in ("model_revision", "container_image_digest", "sources")})


def _cmd_receipt(a):
    import os
    from . import receipt
    key = os.environ.get(a.api_key_env) if a.api_key_env else None
    res = receipt.capture(a.run_dir, a.base_url, a.profile, a.startup_log, a.server_argv, a.image_id,
                          not a.no_metrics, key)
    eff = res["effective"]
    if eff is None:
        print("captured /v1/models; no profile given, nothing compared")
    else:
        print("ok=%s mismatches=%s verified=%d log_only=%d argv_only=%d unverified=%d" % (
            eff["ok"], eff["mismatches"], len(eff["verified"]), len(eff["log_only"]), len(eff["argv_only"]),
            len(eff["unverified"])))
        if not eff["ok"]:
            sys.exit(2)


def _cmd_prepare(a):
    from .mmlu_pro.prepare import prepare
    c = prepare(a.assets, a.model_path, a.reference, a.output, a.runtime_contract, a.per_category)
    print("plan written to %s: %d tasks of %d (%s)" % (a.output, c["tasks"], c["full_test_count"], c["selection"]))


def _cmd_run(a):
    from .mmlu_pro.run import run
    s = run(a.plan, a.output, a.base_url, a.served_model, a.concurrency, a.timeout, a.limit, a.resume,
            a.api_key_env, a.allow_legacy_plan, log=lambda m: print(m, flush=True))
    _print(s)
    if s["transport_failures"]:
        sys.exit("transport failures present (kept, scored incorrect) - inspect requests/*.json")


def _cmd_materialize(a):
    from .mmlu_pro.materialize import materialize
    n = materialize(a.assets, a.reference, a.model_path, a.plan, a.output, a.sample, a.allow_legacy_plan)
    print("wrote %d prompts to %s (all verified identical to the frozen token IDs)" % (n, a.output))


def _cmd_bench(a):
    from .mmlu_pro.bench import bench
    _print(bench(a.run_dir, a.name, a.dataset, a.base_url, a.served_model, a.tokenizer, a.num_prompts,
                 a.output_len, a.vllm_bin, a.dry_run))


def _cmd_summary(a):
    import os
    from .mmlu_pro.summary import build_report
    rep = build_report(a.plan, a.accuracy_dir, a.bench_json, a.allow_legacy_plan)
    if a.out:
        if os.path.exists(a.out):
            sys.exit("refusing to overwrite " + a.out)
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
    _print(rep)


def _cmd_seal(a):
    from . import seal
    try:
        if a.action == "evidence":
            b = seal.seal_evidence(a.run_dir)
            print("evidence_bundle_sha256 =", b["evidence_bundle_sha256"])
            print("files hashed:", len(b["files"]))
            print("-> put this value in the table's Provenance sheet BEFORE running `seal final`.")
        elif a.action == "final":
            if not a.table:
                sys.exit("--table is required for `final`")
            r = seal.seal_final(a.run_dir, a.table, a.ledger)
            print("table_sha256    =", r["table_sha256"])
            print("manifest_sha256 =", r["manifest_sha256"], "  <- record this outside the run dir")
        else:
            r = seal.seal_check(a.run_dir, a.table)
            _print(r)
            sys.exit(0 if r["ok"] else 1)
    except seal.SealError as e:
        sys.exit("ERROR: %s" % e)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hust-eval-mmlu-pro", description=__doc__)
    p.add_argument("--version", action="version", version="hust-eval-mmlu-pro " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("hash-file", help="SHA-256 of one or more files")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=_cmd_hash_file)

    s = sub.add_parser("hash-model", help="hash a model directory, optionally compare with expected hashes")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--expected")
    s.add_argument("--small-only", action="store_true", help="skip weight files (seconds instead of hours)")
    s.set_defaults(fn=_cmd_hash_model)

    s = sub.add_parser("contract", help="build runtime-contract.json (needed by `mmlu-pro prepare`) from a profile")
    s.add_argument("--profile", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--image-id", default="NOT_PROVIDED")
    s.set_defaults(fn=_cmd_contract)

    s = sub.add_parser("receipt", help="capture client-side evidence about a RUNNING server")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--base-url", required=True)
    s.add_argument("--profile", help="expected configuration; enables the comparison")
    s.add_argument("--startup-log", help="frozen copy source of the server's startup log (operator supplied)")
    s.add_argument("--server-argv", help="the server's ACTUAL argv: JSON list or NUL/newline text (/proc cmdline)")
    s.add_argument("--image-id")
    s.add_argument("--api-key-env")
    s.add_argument("--no-metrics", action="store_true")
    s.set_defaults(fn=_cmd_receipt)

    mp = sub.add_parser("mmlu-pro", help="MMLU-Pro tools").add_subparsers(dest="mmlu_cmd", required=True)

    s = mp.add_parser("prepare", help="freeze a plan (needs pyarrow, transformers)")
    s.add_argument("--assets", required=True, help="dir with test-/validation-00000-of-00001.parquet (pinned SHA-256)")
    s.add_argument("--model-path", required=True, help="model dir (tokenizer + chat template)")
    s.add_argument("--reference", required=True, help="TIGER-AI-Lab/MMLU-Pro checkout at the pinned revision")
    s.add_argument("--output", required=True, help="new plan directory (must not exist)")
    s.add_argument("--runtime-contract", help="optional runtime-contract.json (see `contract`), embedded in the plan")
    s.add_argument("--per-category", type=int, help="subset: first N question IDs per category (default: all tasks)")
    s.set_defaults(fn=_cmd_prepare)

    s = mp.add_parser("run", help="send the plan to a running server and score it")
    s.add_argument("--plan", required=True)
    s.add_argument("--allow-legacy-plan", action="store_true",
                   help="accept a plan frozen by the upstream adapter (scorer integrity check not applicable)")
    s.add_argument("--base-url", default="http://127.0.0.1:8000")
    s.add_argument("--served-model", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--concurrency", type=int, default=16)
    s.add_argument("--timeout", type=int, default=1800)
    s.add_argument("--limit", type=int, help="smoke test: first N cases only (not a score)")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--api-key-env")
    s.set_defaults(fn=_cmd_run)

    s = mp.add_parser("materialize", help="make a `vllm bench serve` prompt file from the plan")
    s.add_argument("--allow-legacy-plan", action="store_true")
    s.add_argument("--assets", required=True)
    s.add_argument("--reference", required=True)
    s.add_argument("--model-path", required=True)
    s.add_argument("--plan", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--sample", type=int)
    s.set_defaults(fn=_cmd_materialize)

    s = mp.add_parser("bench", help="run `vllm bench serve` as a client against a running server")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--base-url", required=True)
    s.add_argument("--served-model", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--num-prompts", type=int, default=200)
    s.add_argument("--output-len", type=int, default=256)
    s.add_argument("--vllm-bin", default="vllm")
    s.add_argument("--dry-run", action="store_true", help="write the command and hashes, do not run")
    s.set_defaults(fn=_cmd_bench)

    s = mp.add_parser("summary", help="summarise an accuracy run (and optional bench json)")
    s.add_argument("--allow-legacy-plan", action="store_true")
    s.add_argument("--plan", required=True)
    s.add_argument("--accuracy-dir", required=True)
    s.add_argument("--bench-json")
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_summary)

    s = sub.add_parser("seal", help="bind evidence to a result table by SHA-256")
    s.add_argument("action", choices=["evidence", "final", "check"])
    s.add_argument("--run-dir", required=True)
    s.add_argument("--table")
    s.add_argument("--ledger")
    s.set_defaults(fn=_cmd_seal)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
