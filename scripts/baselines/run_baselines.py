"""Random-subset baselines on the MMLU matrix, with the checks that validate them.

Run from the repository root (~15 min on a laptop CPU), after scripts/irt/efficbench.py:

    uv run python scripts/baselines/run_baselines.py

Part 1 uses every model: uniform and subject-stratified subsets against the closed forms.
Part 2 compares the selection methods on held-out models, on the same 5 family splits as
scripts/irt, so what difficulty and anchors learn never comes from the models they score.

Writes results/baselines/{uniform,stratified,analytic_ratio,methods,checks}.csv and
run_info.json. Seeds are fixed; results/ is git-ignored.
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from cabench import baselines as b
from cabench.splits import behaviour_families, split_models

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "data/processed/efficbench_mmlu"
OUT = ROOT / "results/baselines"

KS = (10, 20, 50, 100, 200, 500, 1000, 2000)
METHODS = ("uniform", "subtask", "difficulty", "systematic", "anchors")
SEED = 0
N_REP = 1000  # part 1: all models
N_REP_HELD_OUT = 200  # part 2: per split and method
N_REP_CALIBRATED = 50
SPLIT_SEEDS = range(5)
FAMILY_DIST = 0.3  # as in scripts/irt/run_experiment.py

checks = []


def check(name, passed, detail):
    """Record one validation; every result is printed and kept, pass or fail."""
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def load_matrix():
    """The response matrix as baselines.py expects it, and item_id -> subject."""
    Y = np.load(MATRIX / "responses.npy")
    models = pd.read_csv(MATRIX / "models.csv", index_col="model_idx")
    items = pd.read_csv(MATRIX / "items.csv", index_col="item_idx")
    M = pd.DataFrame(Y.astype("float32"), index=models["model_id"], columns=items["item_id"])
    M.index.name, M.columns.name = None, None
    return M, pd.Series(items["subject"].to_numpy(), index=items["item_id"].to_numpy())


def check_data(M, subjects):
    print("Data")
    values = np.unique(M.to_numpy())
    check("matrix shape", M.shape == (428, 14042), f"{M.shape[0]} models x {M.shape[1]} items")
    check("values are 0/1 only", set(values) <= {0.0, 1.0}, f"values {values.tolist()}")
    check("no missing answers", M.notna().all(axis=None), "complete, so analytic_se applies")
    check("model ids unique", M.index.is_unique, f"{M.index.nunique()} ids")
    check("item ids unique", M.columns.is_unique, f"{M.columns.nunique()} ids")
    full = b.full_scores(M)
    check(
        "accuracies are plausible",
        full.between(0.25, 0.95).mean() > 0.95,
        f"min {full.min():.3f}, median {full.median():.3f}, max {full.max():.3f}; "
        f"4 options, so chance is 0.25",
    )
    check("57 subjects", subjects.nunique() == 57, f"{subjects.nunique()} subjects")


def check_reproducible(M):
    print("Reproducibility")
    first = b.random_subset_errors(M, [50], n_rep=5, seed=1)
    second = b.random_subset_errors(M, [50], n_rep=5, seed=1)
    third = b.random_subset_errors(M, [50], n_rep=5, seed=2)
    check("same seed, same result", first.equals(second), "identical tables")
    check("new seed, new result", not first.equals(third), "tables differ")


def check_full_benchmark(M):
    n = M.shape[1]
    errors = b.random_subset_errors(M, [n], n_rep=2, seed=SEED)
    worst = errors["err"].abs().max()
    check("k = N recovers the full score", worst < 1e-12, f"largest |err| {worst:.1e}")
    check("k = N has no sampling error", b.analytic_se(0.5, n, n) == 0, "analytic_se is 0")


def check_bias(errors, label):
    """z-test per k on the mean over models. Models share subsets, so test the rep means."""
    rep_means = errors.groupby(["k", "rep"])["err"].mean()
    grouped = rep_means.groupby("k")
    z = grouped.mean() / (grouped.std() / np.sqrt(grouped.count()))
    check(
        f"{label}: unbiased at every k",
        z.abs().max() < 3.5,
        f"largest |z| {z.abs().max():.2f} over {len(z)} k values; "
        f"largest |bias| {grouped.mean().abs().max():.5f}",
    )


def check_analytic(ratio, n_rep):
    # the sd of a sample sd is about sd / sqrt(2(n-1))
    noise = 1 / np.sqrt(2 * (n_rep - 1))
    by_k = ratio.groupby("k")["ratio"]
    medians = by_k.median()
    inside = ratio["ratio"].between(1 - 4 * noise, 1 + 4 * noise).mean()
    check(
        "uniform: simulated sd matches analytic_se",
        (medians - 1).abs().max() < 3 * noise,
        f"median empirical/analytic per k {medians.min():.3f}..{medians.max():.3f} "
        f"(allowed 1 +/- {3 * noise:.3f})",
    )
    check(
        "uniform: holds for each model, not just on average",
        inside > 0.99,
        f"{inside:.1%} of {len(ratio)} (model, k) pairs within 4 Monte Carlo sds",
    )


def stratified_se(M, subjects, k):
    """Closed-form SE of a proportional stratified mean, per model."""
    n = M.shape[1]
    within = []
    for _, group in subjects.groupby(subjects):
        p = M[group.index].mean(axis=1).to_numpy(dtype=float)
        size = len(group)
        within.append(size / n * p * (1 - p) * size / (size - 1))
    return pd.Series(np.sqrt((1 - k / n) / k * np.sum(within, axis=0)), index=M.index)


def check_stratified_se(errors, M, subjects, n_rep):
    noise = 1 / np.sqrt(2 * (n_rep - 1))
    empirical = errors.groupby(["k", "model"])["est"].std()
    medians = {}
    for k in (200, 500, 1000, 2000):  # below 57 x a few items the shares are mostly 0 or 1
        medians[k] = (empirical.loc[k] / stratified_se(M, subjects, k)).median()
    check(
        "stratified: simulated sd matches the closed form",
        all(abs(m - 1) < 3 * noise for m in medians.values()),
        f"median ratio at k=200..2000: {', '.join(f'{m:.3f}' for m in medians.values())}",
    )


def check_allocation(subjects):
    """The mean allocation per subject must equal its exact share k * size / N."""
    rng = np.random.default_rng(SEED)
    k, n_draws = 137, 20000
    sizes = subjects.value_counts().sort_index()
    share = k * sizes / len(subjects)

    draws = pd.DataFrame(
        [b.proportional_allocation(subjects, k, rng) for _ in range(n_draws)], columns=sizes.index
    )
    within = (draws >= np.floor(share)).all(axis=None) and (draws <= np.ceil(share)).all(axis=None)
    check("allocation sums to k", (draws.sum(axis=1) == k).all(), f"{n_draws} draws at k={k}")
    check("allocation is floor or ceil of the share", within, "every subject, every draw")
    sd = np.sqrt((share % 1) * (1 - share % 1) / n_draws).clip(lower=1e-3)
    z = ((draws.mean() - share) / sd).abs().max()
    check("mean allocation equals the exact share", z < 4, f"largest |z| {z:.2f} over 57 subjects")

    # scripts/irt draws through stratified_items, so the shares have to hold there as well
    labels = subjects.to_numpy()
    picks = pd.DataFrame(
        [
            pd.Series(labels[b.stratified_items(labels, k, rng)]).value_counts()
            for _ in range(n_draws)
        ]
    ).reindex(columns=sizes.index)
    z = ((picks.fillna(0).mean() - share) / sd).abs().max()
    check("stratified_items gives the same shares", z < 4, f"largest |z| {z:.2f} over 57 subjects")


def check_against_irt_helper(M):
    """Same seed, same subsets: scripts/irt's numpy path must give the very same scores."""
    X = M.to_numpy(dtype=float)
    ours, theirs = np.random.default_rng(SEED), np.random.default_rng(SEED)
    worst = 0.0
    for k in (10, 100, 1000):
        for _ in range(100):
            mine = b.subset_scores(M, b.sample_items(M.columns, k, ours)).to_numpy()
            other = X[:, b.random_items(X.shape[1], k, theirs)].mean(axis=1)
            worst = max(worst, np.abs(mine - other).max())
    check("same subsets as scripts/irt random_items", worst < 1e-12, f"largest gap {worst:.1e}")


def check_budget(M, uniform):
    """budget_for_error against the k solved from the closed form."""
    full = b.full_scores(M).to_numpy(dtype=float)
    n = M.shape[1]
    variance = np.mean(full * (1 - full))
    # targets sit well inside the grid: at the edge, Monte Carlo noise decides the answer
    for target in (0.05, 0.03, 0.02, 0.012):
        budget = b.budget_for_error(uniform, target)
        exact = n / (1 + (n - 1) * target**2 / variance)
        expected = min(k for k in KS if k >= exact)
        check(
            f"budget for RMSE {target}",
            budget == expected,
            f"grid gives k={budget}, closed form k={exact:.0f}, so k={expected} on the grid",
        )
    check("unreachable target gives None", b.budget_for_error(uniform, 1e-4) is None, "None")


def check_bootstrap(M):
    """A single subset's 95% interval should contain the full score about 95% of the time."""
    rng = np.random.default_rng(SEED)
    sample = M.iloc[rng.choice(len(M), size=60, replace=False)]
    full = b.full_scores(sample)
    for k in (100, 500):
        hits = []
        for _ in range(200):
            items = b.sample_items(M.columns, k, rng)
            se = b.bootstrap_se(sample, items, n_boot=200, rng=rng)
            hits.append(((b.subset_scores(sample, items) - full).abs() <= 1.96 * se).mean())
        coverage = float(np.mean(hits))
        check(f"bootstrap 95% coverage, k={k}", 0.92 <= coverage <= 0.98, f"{coverage:.1%}")


def spearman(errors, by):
    """Rank correlation across models between subset and full score, per group."""
    return errors.groupby(by)[["est", "full"]].apply(
        lambda g: g["est"].corr(g["full"], method="spearman")
    )


def held_out_splits(M):
    families = behaviour_families(M.to_numpy(dtype=float), FAMILY_DIST)
    for seed in SPLIT_SEEDS:
        train, val, test = split_models(len(M), seed, groups=families)
        held_out = np.concatenate([val, test])
        yield seed, train, held_out, set(families[train]) & set(families[held_out])


def compare_methods(M, subjects):
    print("Held-out comparison")
    all_errors, spearmans, calibrated = [], [], []
    leaks = 0
    for seed, train, held_out, shared in held_out_splits(M):
        leaks += len(shared) + len(set(train) & set(held_out))
        calib, held = M.iloc[train], M.iloc[held_out]
        print(f"  split {seed}: {len(calib)} calibration, {len(held)} held-out", flush=True)
        for method in METHODS:
            errors = b.subset_errors(
                held, method, KS, N_REP_HELD_OUT, seed, M_calib=calib, subtasks=subjects
            ).assign(split=seed)
            all_errors.append(errors)
            rho = spearman(errors, ["method", "split", "k", "rep"])
            spearmans.append(rho.rename("spearman").reset_index())
            print(f"    {method} done", flush=True)

        rng = np.random.default_rng(seed)
        full = b.full_scores(held)
        for k in KS:
            for _ in range(N_REP_CALIBRATED):
                items = b.sample_items(M.columns, k, rng)
                err = b.calibrated_scores(calib, held, items) - full
                calibrated.append(
                    pd.DataFrame({"method": "calibrated", "k": k, "err": err.to_numpy()})
                )
    check("no model or family on both sides of a split", leaks == 0, f"{leaks} overlaps, 5 splits")

    errors = pd.concat(all_errors, ignore_index=True)
    rho = pd.concat(spearmans, ignore_index=True)
    grouped = rho.groupby(["method", "k"])["spearman"]
    table = pd.DataFrame(
        {
            "spearman": grouped.mean(),
            "spearman_p10": grouped.quantile(0.1),
            "spearman_p90": grouped.quantile(0.9),
            "n_subsets": grouped.count(),
        }
    )
    summary = b.summarise_errors(errors, by=["method", "k"])
    calibrated = b.summarise_errors(pd.concat(calibrated, ignore_index=True), by=["method", "k"])
    table = pd.concat([table.join(summary), calibrated]).sort_index()
    return table


def check_methods(table):
    rho = table["spearman"].unstack("method").drop(columns="calibrated")
    climbing = (rho.diff().iloc[1:] > -0.01).all(axis=None)
    check("correlation rises with k for every method", climbing, "never falls by more than 0.01")
    biggest = table.groupby("method")["bias"].apply(lambda s: s.abs().max())
    random_methods = biggest[["uniform", "subtask", "difficulty", "systematic"]]
    check(
        "held-out random methods are unbiased",
        random_methods.max() < 0.01,
        f"largest |bias| {random_methods.max():.4f}; anchors {biggest['anchors']:.4f}",
    )


def check_anchors(M):
    """Anchors beat random by a lot at small k, so try to break that result.

    Repeat with new clustering seeds, then shuffle each question's answers across the
    calibration models: that keeps every question's difficulty but destroys the shared
    answer patterns the clusters are built from.
    """
    _, train, held_out, _ = next(held_out_splits(M))
    calib, held = M.iloc[train], M.iloc[held_out]
    full = b.full_scores(held)
    rng = np.random.default_rng(SEED)

    def rho(M_calib, k, seed):
        anchors, weights = b.cluster_anchors(M_calib, k, seed)
        return b.weighted_scores(held, anchors, weights).corr(full, method="spearman")

    shuffled = M.iloc[train].copy()
    shuffled.iloc[:, :] = rng.permuted(calib.to_numpy(), axis=0)
    random_rho = [
        b.subset_scores(held, b.sample_items(M.columns, 10, rng)).corr(full, method="spearman")
        for _ in range(200)
    ]
    real = [rho(calib, 10, seed) for seed in range(30)]
    control = [rho(shuffled, 10, seed) for seed in range(30)]
    check(
        "anchors at k=10 hold up over 30 clustering seeds",
        np.quantile(real, 0.1) > np.mean(random_rho),
        f"anchors mean {np.mean(real):.3f}, 10th-90th percentile "
        f"{np.quantile(real, 0.1):.3f}..{np.quantile(real, 0.9):.3f}; "
        f"random mean {np.mean(random_rho):.3f}",
    )
    check(
        "anchor gain disappears when answer patterns are shuffled",
        np.mean(control) < np.mean(real) - 0.1,
        f"shuffled calibration mean {np.mean(control):.3f}, real {np.mean(real):.3f}",
    )
    return {
        "anchors_k10_seeds": float(np.mean(real)),
        "anchors_k10_shuffled": float(np.mean(control)),
        "random_k10": float(np.mean(random_rho)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M, subjects = load_matrix()
    check_data(M, subjects)
    check_reproducible(M)

    print("All models: uniform", flush=True)
    check_full_benchmark(M)
    uniform_errors = b.random_subset_errors(M, KS, N_REP, SEED)
    uniform = b.summarise_errors(uniform_errors)
    ratio = b.empirical_vs_analytic(uniform_errors, M.shape[1])
    check_bias(uniform_errors, "uniform")
    check_analytic(ratio, N_REP)
    check_against_irt_helper(M)
    check_budget(M, uniform)
    uniform.to_csv(OUT / "uniform.csv")
    ratio.to_csv(OUT / "analytic_ratio.csv", index=False)
    del uniform_errors

    print("All models: stratified by subject", flush=True)
    stratified_errors = b.random_subset_errors(M, KS, N_REP, SEED, subjects)
    stratified = b.summarise_errors(stratified_errors)
    check_bias(stratified_errors, "stratified")
    check_stratified_se(stratified_errors, M, subjects, N_REP)
    check_allocation(subjects)
    stratified.to_csv(OUT / "stratified.csv")
    del stratified_errors

    print("Single-subset intervals", flush=True)
    check_bootstrap(M)

    table = compare_methods(M, subjects)
    check_methods(table)
    table.to_csv(OUT / "methods.csv")
    anchor_checks = check_anchors(M)

    results = pd.DataFrame(checks)
    results.to_csv(OUT / "checks.csv", index=False)
    info = {
        "seed": SEED,
        "ks": KS,
        "n_rep": N_REP,
        "n_rep_held_out": N_REP_HELD_OUT,
        "split_seeds": list(SPLIT_SEEDS),
        "family_dist": FAMILY_DIST,
        "matrix": M.shape,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        **anchor_checks,
    }
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\n{results['passed'].sum()} of {len(results)} checks passed; wrote {OUT}")


if __name__ == "__main__":
    main()
