import numpy as np
import pytest

from cabench.baselines import stratified_items


def test_stratified_items():
    rng = np.random.default_rng(0)
    subjects = np.repeat(["a", "b", "c"], [700, 200, 100])
    for k in [1, 7, 10, 100]:
        idx = stratified_items(subjects, k, rng)
        assert len(idx) == k == len(set(idx)), k
    counts = np.unique(subjects[stratified_items(subjects, 100, rng)], return_counts=True)[1]
    assert (counts == [70, 20, 10]).all()


def test_stratified_items_match_exact_shares():
    # shares 5.9, 2.7, 1.4: two leftovers, and drawing them by weight shorts the first subject
    subjects = np.repeat(["a", "b", "c"], [590, 270, 140])
    rng = np.random.default_rng(0)
    counts = [
        np.unique(subjects[stratified_items(subjects, 10, rng)], return_counts=True)[1]
        for _ in range(5000)
    ]
    assert np.mean(counts, axis=0) == pytest.approx([5.9, 2.7, 1.4], abs=0.03)
