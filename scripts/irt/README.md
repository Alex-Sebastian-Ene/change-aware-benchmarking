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

## Minimal IRT baseline
`notebooks/irt_04_baseline` is the reference point for later IRT changes. It uses a 1-D 2PL model.
It checks that the fit is sound (convergence, parameter sanity, held-out answer prediction,
calibration), then evaluates score estimation on test and saves `results/irt/baseline.csv`.

The protocol lives in `protocol.py`:
- **Methods** are (selector, estimator) pairs registered in `SELECTORS` / `ESTIMATORS`.
- **`evaluate`** runs any list of methods on the same splits, k values and seeds as the baseline.

A new method (for example information- or cost-aware selection) is one new function plus one
dictionary entry.

## Progression through V-optimality

The next three notebooks separate estimator, selector and model changes:

| Notebook | Question | Selectors | Dimensions |
|---|---|---|---|
| `irt_05_information_selection` | Does information beat representativeness? | random, anchors, static/adaptive Fisher | 1 |
| `irt_06_mirt` | Does multidimensional structure reduce error? | random only | 1, 2, 3, 5, 8 |
| `irt_07_voptimal` | Does matrix-aware selection help MIRT? | random, anchors, static/adaptive V-optimal | validation choice from 06 |

Each supports accuracy, p-IRT and the existing gp-IRT blend. Fit the item model on training
models and the gp-IRT weight separately for each split, selector and k on validation. In 06,
choose D using random + gp-IRT mean log validation MAE across budgets, then freeze it before
the test dimensionality sweep. Training-only PCA / Bernoulli parallel analysis and hidden-answer
validation log loss, Brier score and calibration ECE provide separate model diagnostics.

Run the notebooks in order, or run the same experiments from the repository root:

```bash
uv run --group irt python scripts/irt/run_progression.py
# Individual stages (run mirt before voptimal):
uv run --group irt python scripts/irt/run_progression.py --stage fisher
uv run --group irt python scripts/irt/run_progression.py --stage mirt
uv run --group irt python scripts/irt/run_progression.py --stage voptimal
# Reduced real-data integration check, NOT research results:
uv run --group irt python scripts/irt/run_progression.py --smoke
```

Full defaults remain five family splits, 20 seeds and k = 10, 20, 50, 100, 200, 500.
Outputs go to `results/irt/progression/{information,mirt,voptimal}/`; `--smoke` uses a separate
`results/irt/progression/smoke/` tree (one split, two seeds, k=10/20, 100 training epochs).
The original `baseline.csv` and legacy experiment are untouched.

Each stage saves `test.csv` (the existing metrics plus `n_dims`), `predictions.csv` (one row per
split/seed/k/selector/estimator/dimension/model), `validation_predictions.csv`, `gp_weights.csv`
and `paired.csv`. Per-model rows retain `model_id`, `true_score`, `pred_score`, `abs_error` and
`signed_error`. Scores are fractions; multiply errors and confidence limits by 100 for pp.
06 also saves `dimensionality.csv`, `response_validation.csv` and `chosen_dimension.csv`.

Implementation details that matter when interpreting results:

- The parameterisation is `sigmoid(theta @ alpha.T - beta)`: beta is an intercept. In 1D,
  Fisher information is `alpha**2 * p * (1-p)`. Static Fisher averages over training abilities.
- Adaptive methods start at the training mean, choose one item, reveal only that answer,
  and refit ability with item parameters frozen. One trajectory yields every k checkpoint.
  These selectors and static information designs are deterministic; rows repeated over seeds
  enable paired comparisons but are not independent samples.
- New experiments use a float64 Newton MAP solver on selected answers, with a small training
  covariance ridge. The baseline still defaults to its original Adam solver. Tests check
  solver agreement. p-IRT retains the baseline definition: actual seen answers, IRT predictions
  elsewhere, averaged across the full benchmark.
- Accuracy on intentionally informative subsets is usually biased. Keep it as an ablation;
  the validation-fitted blend can put zero weight on it. gp-IRT does not guarantee improvement.
- V-optimality uses `M = alpha.T @ alpha / Q`, targeting mean **logit prediction variance**
  over all MMLU items, following [Krumdick et al., Eq. 6](https://arxiv.org/html/2604.01418v1#S3.SS2).
  It is a local covariance approximation with fixed item parameters, not a calibrated score
  interval. Static designs accumulate expected training information; adaptive designs
  recompute accumulated information at the updated ability. The Fisher adaptation follows
  [Fluid Language Model Benchmarking](https://arxiv.org/abs/2509.11106).
- Paired comparisons match model-level errors, average seeds within models and models within
  splits, then bootstrap paired split means. Negative delta MAE means improvement. Five
  overlapping family splits provide limited evidence; these CIs describe split variability.

This implementation stops at V-optimal MIRT. It adds no token-cost selection, cross-benchmark
pool, task-holdout prediction or fine-tuning delta analysis. The data still needs a complete
0/1 matrix; the team's future NaN-containing MMLU-Pro matrix needs separate handling.

## Use a teammate's fitted model

Selection is independent of the fitting backend. `cabench.design` uses only NumPy and SciPy;
it can be imported without PyTorch or scikit-learn. The original array functions in `cabench.irt`
remain available as compatibility wrappers. There are two ways to supply a frozen model.

**Parameter adapter**, for Rasch / 2PL / linear-logit MIRT fitted by any library:

```python
from cabench.design import parameter_context, static_design, adaptive_design

ctx = parameter_context(train_theta, item_alpha, item_beta, item_ids=item_ids)
# Prediction is sigmoid(theta @ alpha.T - beta).
# For sigmoid(a * (theta-b)), convert beta=a*b; for Rasch, alpha=ones((Q,1)).
# NumPy arrays must be in response-column order and the same latent coordinates.

static_items = static_design(ctx, k=20, criterion="voptimal")
trajectory = adaptive_design(Y_new, ctx, ks=[10, 20, 50], criterion="voptimal")
for k, items_per_model, theta in trajectory:
    ...

# Optional: use the teammate's ability estimator instead of the default Newton MAP.
ctx["infer"] = teammate_infer
```

**Callback adapter**, retaining a teammate's predictor and inference implementation:

```python
from cabench.design import model_context

ctx = model_context(
    train_theta, item_loadings,
    predict=teammate_predict,  # predict(theta, items=None) -> probabilities (models, items)
    infer=teammate_infer,     # infer(responses, items, initial=None) -> theta (models, D)
    embeddings=item_embeddings,  # optional; required only for the anchor selector
    item_ids=item_ids,
)
```

Both callbacks support shared `(k,)` and per-model `(models,k)` item indices. Inference gets
only the selected response array `(models,k)`, never a full response matrix. Keep item
parameters frozen during evaluation. Detach PyTorch tensors and move them to CPU before
converting to NumPy. A context's training abilities must come from that split's training
models; fitting on the full population would invalidate the holdout experiment.

The design contract is rank-one item information `I_i(theta)=w_i(theta)*a_i*a_i.T`, with
fixed loading directions `a_i`. The default `w=p(1-p)` assumes a linear-logit model. For a
different link, pass its correct `information_weights(theta, items=None)` callback returning
nonnegative `(models,items)` weights. Models with ability-dependent information directions
need a further adapter; arbitrary nonlinear IRT is not silently treated as 2PL.
Optional `prior_cov` and positive-semidefinite `target` matrices let the teammate supply
their training prior and V-optimal prediction objective. Match the inference prior and
ability coordinates to these matrices. Default target remains `A.T @ A / Q`.

The notebooks have explicit plug-in cells:

- **05:** supply `contexts=[ctx_for_split_0, ...]`, or replace `build_context`.
- **06:** supply `contexts_by_dim={1: contexts_D1, 2: contexts_D2, ...}`, or replace
  `build_context`. `run_mirt` takes `context_factory` / `contexts_by_dim`; supplied models
  are never refitted. The notebook uses the supplied mapping's available dimensions.
  Paired comparisons use the smallest supplied D and record it in `reference_dims`;
  include D=1 to retain the original unidimensional comparison.
- **07:** supply `contexts` directly. No notebook-06 file is read in this path. Alternatively
  set a validation-chosen `n_dims` and a fitter; the 06 choice file is only a fallback.
- **08:** `irt_08_visualise_methods.ipynb` reads any stage's `predictions.csv` without fitting.
  It provides selector curves, estimator heatmaps, prediction parity / signed-error plots,
  paired differences, and optional static/adaptive item geometry. Its default synthetic
  demo works without previous experiment outputs and is labelled separately from real data.

`run_selection` and `evaluate` take `item_ids` to check model item IDs against the response
column order when both are supplied. Supply these in shared experiments, and use a distinct
output directory per fitted model so result files are not mixed or overwritten. The same
validation-only gp-IRT calibration and paired-record format work with either adapter.
