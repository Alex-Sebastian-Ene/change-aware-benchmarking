"""Noise reference (re-runs of the same model) and random-subset baselines at equal compute.

Core objective 2. Owner: S. Ene.
"""

import numpy as np


def random_items(n_items, k, rng):
    return rng.choice(n_items, size=k, replace=False)


def stratified_items(subjects, k, rng):
    """k random items allocated to subjects in proportion to subject size.

    Leftover slots after rounding down go to subjects drawn in proportion to their remainder.
    """
    names, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    quota = k * counts / counts.sum()
    alloc = np.floor(quota).astype(int)
    frac = quota - alloc
    extra = k - alloc.sum()
    if extra:
        alloc[rng.choice(len(names), size=extra, replace=False, p=frac / frac.sum())] += 1
    return np.concatenate(
        [
            rng.choice(np.flatnonzero(inverse == s), size=n, replace=False)
            for s, n in enumerate(alloc)
            if n
        ]
    )
