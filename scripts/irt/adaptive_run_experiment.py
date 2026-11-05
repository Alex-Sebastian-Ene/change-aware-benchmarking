"""Runs adaptive testing experiments and logs results.
Implements Fluid Benchmarking (Hofmann et al., 2025). See src/cabench/adaptive_irt.py

Run from the repository root (roughly 5 mins on a laptop CPU):
    uv run --group irt python scripts/irt/adaptive_run_experiment.py

Uses the same family splits as run_experiment.py (SPLIT_SEEDS, FAMILY_DIST). 
For each split:
  1. fit a, b on the train models with adaptive_irt.fit_2pl and freeze them
  2. estimate each test model's reference ability (true_theta) using all items 
  3. run Fluid on each test model for max(KS) items (one run gives every smaller k too)
  4. at each k, compare Fluid with random_ability on two scales: 
        ability (theta vs true_theta) and accuracy (expected_accuracy vs real accuracy)

Writes results/irt/mmlu_efficbench/:
  adaptive_test.csv    metrics per split, k, method, seed and scale
                       plus a "full ability" row per split (the error floor of the accuracy conversion)
  adaptive_thetas.csv  per-model theta and expected accuracy for every split, k, method and seed
  adaptive_items.csv   Fluid's item sequence per test model: item, subject, a, b, answer, theta after it
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from efficbench import ROOT, load

from cabench.baselines import random_items
from cabench.adaptive_irt import (
    fit_2pl,
    ability_estimate,
    run_fluid
)
from run_experiment import evaluate_scores
from cabench.splits import behaviour_families, split_models

KS = [10, 20, 50, 100, 200, 500]
SPLIT_SEEDS = range(5)
SEEDS = range(20)
FAMILY_DIST = 0.3

def expected_accuracy(theta, a, b):
    """ Predicted accuracy on full benchmark: mean over all items of P(correct | theta).
    Not included in Fluid-benchmarks but added here for comparison with static benchmarking method
    """
    theta = np.asarray(theta) 
    p = expit(a * (theta[:, None] - b)) # [n_models, n_items]
    return p.mean(1) # mean over items


def full_ability(Y, a, b):
    """ Ability estimate based on ALL responses, used as the reference "true" ability.
    This is the best estimate the frozen a, b allow, so it is the target that k-item
    estimates are scored against on the ability scale.
    """
    full_abilities = np.full([Y.shape[0]], np.nan)

    for i in range(Y.shape[0]):
        full_abilities[i] = ability_estimate(Y[i], a, b)

    return full_abilities 


def random_ability(Y, a, b, k, seed):
    """ Ability estimate based on k random items
    """
    rng = np.random.default_rng(seed)
    idx = random_items(Y.shape[1], k, rng)
    R = np.full(Y.shape, np.nan)
    R[:, idx] = Y[:, idx]
    random_abilities = np.full([R.shape[0]], np.nan)

    for i in range(R.shape[0]):
        random_abilities[i] = ability_estimate(R[i], a, b)

    return random_abilities


def fluid_abilities(Y, a, b, start_ability, n_max):
    """ Run Fluid on each model for n_max items
    Returns abilities (n_models, n_max), where abilities[:, k - 1] is theta after k items,
    and items (n_models, n_max), the item indices asked, in order.
    """
    abilities = np.full((Y.shape[0], n_max), np.nan)
    items = np.zeros((Y.shape[0], n_max), dtype=int)

    for i in range(Y.shape[0]):
        abilities[i], items[i] = run_fluid(Y[i], a, b, start_ability, n_max) 
    return abilities, items


def main():
    Y, models, items = load() # load data from efficbench 
    Y = Y.astype(np.float32)
    families = behaviour_families(Y, FAMILY_DIST)
    splits = {s: split_models(len(Y), s, groups=families) for s in SPLIT_SEEDS}

    test_rows = []
    theta_rows = []
    item_rows = []
    
    for s, (train, _, test) in splits.items():

        # 1. fit item params on train models
        a, b, theta_train = fit_2pl(Y[train])

        start_ability = theta_train.mean()

        # true values
        Y_test = Y[test]
        true_acc = Y_test.mean(1) # true accuracy of all items
        true_theta = full_ability(Y_test, a, b) # true ability estimate based on all responses

        # 2. Fluid run per test model
        fluid_thetas, fluid_items = fluid_abilities(Y_test, a, b, start_ability, max(KS))

        # 3. Logs 
        for k in KS:
            # method, seed, theta of every test model after k questions
            estimates = [("fluid", None, fluid_thetas[:, k - 1])]

            # Random ability 
            for seed in SEEDS: 
                estimates += [("random_ability", seed, random_ability(Y_test, a, b, k, seed))]

            for method, seed, theta in estimates:
                acc = expected_accuracy(theta, a, b) # predicting accuracy based on model's theta (abilty)

                # summary per method
                keys = {"split": s, "k": k, "method": method, "seed": seed}
                test_rows.append({**keys, "scale": "ability", **evaluate_scores(true_theta, theta)})
                test_rows.append({**keys, "scale": "accuracy", **evaluate_scores(true_acc, acc)})

                # per-model values: Fluid, and random_ability (20 seeds)
                for j, m in enumerate(test):
                    theta_rows.append({
                        **keys,
                        "model_idx": m, # the model's row in Y
                        "model_id": models.model_id.iloc[m], # model name
                        "theta": theta[j], 
                        "expected_accuracy": acc[j], 
                        "true_theta": true_theta[j], # theta from all questions
                        "true_accuracy": true_acc[j], 
                    })

    # evaluate score between true accuracy and best possible theta from all items - the error floor
        test_rows.append({
            "split": s, "k": Y.shape[1], "method": "full ability", "seed": None,
            "scale": "accuracy",
            **evaluate_scores(true_acc, expected_accuracy(true_theta, a, b)),
        })

        # which items Fluid asked each test model, in order
        for j, m in enumerate(test):
            for step, i in enumerate(fluid_items[j], start=1):
                item_rows.append({
                    "split": s,
                    "model_idx": m, # model index in Y
                    "step": step, # order of question
                    "item_idx": i, # item's index in Y
                    "item_id": items.item_id.iloc[i], # which benchmark quesion from items.csv
                    "subject": items.subject.iloc[i],
                    "a": a[i],
                    "b": b[i],
                    "correct": Y[m, i], # whether the model got it right
                    "theta": fluid_thetas[j, step - 1], # theta predicted after this answer
                })

        print(f"split {s} done", flush=True)

    out = ROOT / "results/irt/mmlu_efficbench"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(test_rows).to_csv(out / "adaptive_test.csv", index=False)
    pd.DataFrame(theta_rows).to_csv(out / "adaptive_thetas.csv", index=False)
    pd.DataFrame(item_rows).to_csv(out / "adaptive_items.csv", index=False)


if __name__ == "__main__":
    main()
