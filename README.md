# hust-eval-client-mmlu-pro

MMLU-Pro client for an **already-running** OpenAI-compatible vLLM server.

It freezes a plan, sends requests, scores answers, computes SHA-256 hashes and seals evidence. **It never starts,
stops or configures a server**, on purpose: whoever operates the server hands over what a client cannot see
(startup log, actual argv, image digest), and this tool records and checks it. A test (`ClientOnlyGuard`) fails
if any module starts background processes or launches `vllm serve`.

It is self-contained: no other project's code is imported or needed at run time.

**New user? Start with [docs/QUICKSTART.md](docs/QUICKSTART.md)** (Chinese): prerequisites, step-by-step commands, what to
report, common errors.

This project is the **MMLU-Pro** one. The other designated datasets (HLE-Verified, SWE-bench-Pro, FrontierScience,
Terminal-Bench 2.1) are meant to become separate projects of the same shape, merged later if worthwhile.
The hashing, receipt, config-check and seal modules are dataset-independent; for now each project carries its own
copy, so a fix made in one must be copied to the others until they are merged.

Names are per project so several can be installed in one environment: this project installs the Python package
`hust_eval_mmlu_pro` and the console command `hust-eval-mmlu-pro`. The other projects should take their own
(`hust_eval_<dataset>` / `hust-eval-<dataset>`).

## What it is not

- Not the upstream leaderboard configuration. Prompt text and answer extraction follow the reference runner of
  TIGER-AI-Lab/MMLU-Pro, but prompts are sent as pre-rendered chat-template token IDs with thinking disabled, all five
  shots are kept, there is no `Question:` stop string and the context is 32K. Report results as non-official.
- One scoring rule differs from the reference runner: when no answer letter can be extracted the reference guesses a
  random option; here the answer is incorrect.
- Not a signed evidence bundle. `seal` writes a plain SHA-256 listing (not RFC 8785 canonical JSON, no signature).
- Not a V5.5 formal run: single lifecycle, no B1, client contract differs. See "Known limits".

## Install

```bash
cd hust-eval-client-mmlu-pro
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip                # REQUIRED on the system Python of macOS (pip 21): `pip install -e` fails without it
pip install -e .                  # core is standard library only
pip install -e '.[prepare]'       # only for `mmlu-pro prepare` and `mmlu-pro materialize` (pyarrow, transformers)
hust-eval-mmlu-pro --help         # or: python -m hust_eval_mmlu_pro --help
```

Python 3.9+. `mmlu-pro bench` additionally needs the `vllm` command on the client machine (it runs
`vllm bench serve`, which is a client).

**Pin `transformers` to the version you froze with.** The frozen prompts are token IDs produced by the model's chat
template through `transformers`; another version may render the template differently and give different IDs, i.e.
a different plan. The first plan here was frozen with `transformers==5.14.1` and `pyarrow==19.0.1`; after
`prepare`, compare the `cases.json` SHA-256 with the one you expect (for the same model, parquet and reference, the
first plan's was `f77a7e5e6fd96fbbf46be7ab11f360309554105e97921805f02d1c93fc05fcf3`).

Installation was verified with `pip install -e .` and `pip install .` into a fresh venv on Python 3.9 after
upgrading pip. The `[prepare]` extra was not installed from scratch in a clean environment.

## Inputs you must provide

| What | Why |
|---|---|
| MMLU-Pro parquet files `test-00000-of-00001.parquet`, `validation-00000-of-00001.parquet` (HF revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`) | SHA-256 checked against the pinned values; anything else is refused |
| a checkout of `TIGER-AI-Lab/MMLU-Pro` at `f418b116db00b065c2aea046518d8fcf74d39872` | the reference initial prompt; `cot_prompt_lib/initial_prompt.txt` and `evaluate_from_local.py` are SHA-256 checked |
| the model directory | tokenizer and chat template used to freeze the prompts |

## Typical flow

```bash
RUN=runs/mmlu-pro/mmlu-pro-<profile>-$(date +%Y%m%dT%H%M%S)

# 1. model identity (once). --small-only is seconds; without it the weights are hashed too
hust-eval-mmlu-pro hash-model --model /models/Qwen3.5-35B-A3B --out model_hashes.json --small-only

# 2. freeze the plan: every test task unless --per-category N is given
hust-eval-mmlu-pro contract --profile examples/profile.b0_35b.json --out runtime-contract.json --image-id sha256:...   # optional
hust-eval-mmlu-pro mmlu-pro prepare --assets <parquet dir> --model-path <model dir> \
    --reference <MMLU-Pro checkout> --runtime-contract runtime-contract.json --output plan

# 3. server already running (started by someone/something else). Record what a client can observe:
hust-eval-mmlu-pro receipt --run-dir $RUN --base-url http://127.0.0.1:18180 \
    --profile examples/profile.b0_35b.json \
    --startup-log /path/to/serve.log --server-argv /path/to/argv.json --image-id sha256:...

# 4. smoke, then the full accuracy run (a pure client)
hust-eval-mmlu-pro mmlu-pro run --plan plan --base-url http://127.0.0.1:18180 \
    --served-model qwen3.5-35b --output $RUN/outputs/smoke --limit 20
hust-eval-mmlu-pro mmlu-pro run --plan plan --base-url http://127.0.0.1:18180 \
    --served-model qwen3.5-35b --output $RUN/outputs/mmlu-pro-accuracy --concurrency 16   # --resume to continue

# 5. optional throughput run (client: vllm bench serve)
hust-eval-mmlu-pro mmlu-pro materialize --assets <parquet dir> --reference <MMLU-Pro checkout> \
    --model-path <model dir> --plan plan --output $RUN/inputs/materialized.200.jsonl --sample 200
hust-eval-mmlu-pro mmlu-pro bench --run-dir $RUN --name mmlu-pro --dataset $RUN/inputs/materialized.200.jsonl \
    --base-url http://127.0.0.1:18180 --served-model qwen3.5-35b --tokenizer <model dir>

# 6. read it, then seal and bind the table
hust-eval-mmlu-pro mmlu-pro summary --plan plan --accuracy-dir $RUN/outputs/mmlu-pro-accuracy \
    --bench-json $RUN/outputs/mmlu-pro.json
hust-eval-mmlu-pro seal evidence --run-dir $RUN      # -> evidence_bundle_sha256: write it into the table's Provenance sheet
hust-eval-mmlu-pro seal final --run-dir $RUN --table $RUN/results/results.xlsx --ledger runs/ledger.jsonl
hust-eval-mmlu-pro seal check --run-dir $RUN
```

Commands that hash: `hash-file`, `hash-model`, `contract` (profile hash), `mmlu-pro prepare` (plan files, scorer),
`receipt` (profile, startup log, argv), `mmlu-pro bench` (dataset + result JSON), `seal` (everything in the run
dir, then the table).

## The plan

`prepare` writes `cases.json` (frozen prompt token IDs, gold answer, option count, limits), `tasks.json` (the same
without token IDs), `inventory.jsonl.gz` (every test question, canonical JSON, gzip `mtime=0` so it is
reproducible) and `contract.json` with the SHA-256 of those three files **and of `scoring.py`**. Loading a plan
re-checks all of it, so the scorer cannot change silently between freezing a plan and scoring a run, and a
modified prompt or file is refused.

Plans frozen by the earlier upstream adapter (`contract_version: pyramidkv-mmlu-pro-subset-v1`) load only with
`--allow-legacy-plan`; the scorer-integrity check cannot be applied to them and the reports say so
(`legacy_plan: true`).

## Provenance and equivalence

Prompt layout and answer extraction are a reimplementation of the reference runner of TIGER-AI-Lab/MMLU-Pro
(Apache-2.0), and the plan format and scoring semantics follow the `mmlu_pro_pyramidkv.py` adapter of
vLLM-HUST/vllm-hust-benchmark (commit `9e7f83388d72a37767fd9f24636219093ecbe662`). Nothing was copied; equivalence
was checked on real data on 2026-10-12:

- Freezing the full test set with this project produced `cases.json` `f77a7e5e…fcf3`, `tasks.json` `5d23a790…2873`
  and `inventory.jsonl.gz` `c696840c…8b4b`, byte-for-byte the files the upstream adapter produced for the same
  inputs.
- Scoring 6,999 real model outputs with both scorers gave identical answer extraction on every output, identical
  totals (5,449 correct; 27 invalid; 1,049 length-limited) and **0** differing per-case outcomes.
- The HTTP request bodies this client and the earlier script send for the same case are byte-identical.

These checks cover the plan, the requests and the scoring. They say nothing about the model's outputs being
reproducible (see the next section).

## Outputs are not reproducible run to run

On the vLLM 0.28.1 / vllm-ascend server used for the first runs, sending the same frozen prompts with
`temperature=0` and a fixed `seed` did **not** give the same text twice: for the same 5 questions, two runs of the
earlier script and one run of this client were pairwise 0/5 identical (completion lengths differed, e.g. 351 vs 340
vs 652 tokens), and so were all of them against the first full run. This was observed once, on 5 questions, under
concurrent load; it is not a measurement of the size of the effect. Practical consequences: a single run's accuracy
carries run-to-run noise that this tool does not estimate, and "the outputs differ" is not evidence that two clients
differ. To estimate the noise, repeat the whole run (the V5.5 plan asks for three independent lifecycles and the
median).

## Evidence model

- **Run directory** holds `profile.json`, `receipt/`, `inputs/`, `outputs/`, `results/` (the table).
- **Seal order matters.** A table cannot contain its own hash, so: `seal evidence` hashes every file except
  `results/` and the seal files -> put that bundle hash into the table -> `seal final` hashes the table and
  writes `MANIFEST.json`. Keep `MANIFEST.json`'s own SHA-256 outside the run dir (ledger, git, chat).
- **Freeze the server log before sealing.** The tool cannot see whether a server is still writing. Hand it a copy
  of the startup log (`receipt --startup-log`), not the live file.
- **Field status** in `receipt/effective.json` (from `config_check`):
  `verified` (argv and log agree), `log_only`, `argv_only` (a command-line fact, not proof of the engine's value),
  `unverified`, `mismatch`. `ok=true` means *nothing disagreed*, not *everything was proven*; read the lists.
- Never overwritten: `prepare`, `run`, `bench`, `hash-model`, `contract`, `seal` all refuse to write over existing
  output. Re-runs go to a new directory; `run --resume` continues the same one and only reuses requests that succeeded.
- Failed requests are not retried; they stay in the denominator and are scored incorrect.

## Known limits

- Log parsing was calibrated on one real vLLM 0.28.1.post1.dev143 / vllm-ascend startup log. Other versions may
  print differently; fields then degrade to `argv_only`/`unverified` instead of `verified`.
- Extraction stage 3 takes the last standalone A-J letter in the text. For outputs cut off by the token limit it often
  picks a letter that is just prose (e.g. the article "A"); such outputs are scored like any other. `summary` reports
  correctness separately for outputs that finished normally and outputs that hit the limit.
- `mmlu-pro run` uses `/v1/completions` with token IDs; V5.5-style chat-completions clients are not implemented.
- No RFC 8785 canonicalisation, no signing.

## Layout

```
src/hust_eval_mmlu_pro/
  cli.py hashing.py seal.py receipt.py config_check.py runtime_contract.py     # dataset-independent
  mmlu_pro/  prompt.py scoring.py plan.py prepare.py run.py materialize.py bench.py summary.py
tests/  examples/
```

## Tests

```bash
PYTHONPATH=src:tests python3 -m unittest discover -s tests -v
```
