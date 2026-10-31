# Random-subset baselines on MMLU

How well does a score on k random questions stand in for the full 14,042-question score?
This is what IRT has to beat. The functions are in `src/cabench/baselines.py`.

## Data
`Y`: 428 models × 14,042 MMLU items (5-shot, Open LLM Leaderboard), built by
`scripts/irt/efficbench.py` from `mmlu_fields.pickle`. It says right or wrong only, and has
one run per model, so `noise_reference` needs our own repeat runs and is not run here.

`answer_choices.py` adds which option each model picked, for a sample of 24 models spread
across the score range. It reads the public `open-llm-leaderboard-old` details and the
answer key from `cais/mmlu`.

## Run
From the repository root:

```bash
uv sync --group irt
# put mmlu_fields.pickle in data/raw/
uv run python scripts/irt/efficbench.py
uv run python scripts/baselines/run_baselines.py     # -> results/baselines/  (~15 min)
uv run python scripts/baselines/answer_choices.py    # needs internet
uv run --group irt python scripts/baselines/plot_results.py
```

## Method
- **All models:** uniform and subject-stratified subsets, 1,000 repetitions per k, checked
  against the closed-form standard errors. `checks.csv` lists every check, pass or fail.
- **Held-out models:** the same 5 family splits as `scripts/irt`, with train as calibration
  and val + test as held-out. Selection methods: uniform, subtask, difficulty, systematic,
  anchors. 200 repetitions per split, method and k; anchors are deterministic, so one.
- **Correlation** is Spearman across held-out models between subset and full score, the
  same measure `scripts/irt` reports, so the two sets of results can be compared.
