"""Split definitions (in-distribution, stronger/weaker models, held-out family, extrapolation).

Core objective 7. Owners: S. Ene, J. Hall.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def split_models(n_models, seed, groups=None):
    """70/15/15 split of model indices into train / val / test.

    With `groups`, each group stays on one side (held-out family split).
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.permutation(n_models)
        a, b = int(0.70 * n_models), int(0.85 * n_models)
        return idx[:a], idx[a:b], idx[b:]
    groups = np.asarray(groups)
    order = np.concatenate(
        [np.flatnonzero(groups == g) for g in rng.permutation(np.unique(groups))]
    )
    cum = np.arange(1, n_models + 1)
    # cut only at group boundaries
    ends = np.flatnonzero(np.r_[groups[order][1:] != groups[order][:-1], True])
    a = ends[np.abs(cum[ends] - 0.70 * n_models).argmin()] + 1
    b = ends[np.abs(cum[ends] - 0.85 * n_models).argmin()] + 1
    return order[:a], order[a:b], order[b:]


def behaviour_families(Y, max_dist=0.3):
    """Cluster models by answer correlation (average linkage on 1 - corr).

    shortcut: the efficbench data has no base-model lineage, so a family is behavioural;
    use the `family` / `parent_model` columns of models.parquet once that matrix is used.
    """
    D = 1 - np.corrcoef(Y)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), "average"), max_dist, "distance")
