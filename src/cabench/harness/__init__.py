"""Pinned evaluation harness: every run is tied to exact model, data and code versions.

Modules (build them in this order):
  pins.py      the three pin dataclasses + the pinned values       (no GPU)
  verify.py    hashing, git and version checks                     (no GPU)
  manifest.py  per-run manifest + config_id / run_id               (no GPU)
  parsers.py   versioned answer parsers for generative tasks       (no GPU)
  run.py       the only module that imports lm_eval / vllm         (GPU)
"""
