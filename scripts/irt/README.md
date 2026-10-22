# IRT baseline on MMLU (efficbench)

How well can a model's full MMLU score be estimated from k of its answers?
A from-scratch, tinyBenchmarks-style baseline: fit multidimensional IRT on known models,
pick k anchor items, and estimate unseen models' scores.

## Data
`Y`: 428 models × 14,042 MMLU items (5-shot, Open LLM Leaderboard), `Y[m, i] = 1` if correct.
Source: `mmlu_fields.pickle` from [efficbench](https://github.com/felipemaiapolo/efficbench/tree/master/data).
This is a stand-in until the MMLU-Pro matrix from `scripts/matrix/` is ready.

## Method
- **Split** models 70/15/15 by behavioural family, so no held-out model has a near-duplicate in train
  (`cabench.splits`).
- **MIRT** on train models: P(correct) = σ(θ_m·α_i − β_i), L2-regularised maximum likelihood (`cabench.irt`).
- **Selection**: random, subject-stratified (`cabench.baselines`), or IRT anchors (K-means on
  standardised [α, β], one item per cluster, weighted by cluster size).
- **Estimators**: anchor-weighted accuracy; p-IRT (infer θ from the k answers, predict the rest);
  gp-IRT (blend of the two, weight fitted on validation).
- **Protocol**: D ∈ {1, 2, 3, 5} and gp-IRT weights chosen on validation; test evaluated once.
  5 splits × 20 seeds × k ∈ {10, 20, 50, 100, 200, 500}.

## Run
From the repository root:

```bash
uv sync --group irt
# put mmlu_fields.pickle in data/raw/
uv run python scripts/irt/efficbench.py                  # -> data/processed/efficbench_mmlu/
uv run --group irt python scripts/irt/run_experiment.py  # -> results/irt/  (~30-60 min on a laptop CPU)
uv run --group irt pytest
```

Notebooks `notebooks/irt_01_inspect_data`, `irt_02_fit_mirt` and `irt_03_results` look at the
matrix, a single MIRT fit, and the experiment results.
