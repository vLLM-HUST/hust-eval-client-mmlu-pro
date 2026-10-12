"""hust_eval_mmlu_pro: client-side evaluation tooling.

Scope (on purpose): this package talks to a server that somebody else started. It sends requests,
scores answers, computes SHA-256 hashes and seals evidence. It never launches, stops or configures a server.
"""

__version__ = "0.1.0"
