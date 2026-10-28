import numpy as np

from cabench.baselines import stratified_items


def test_stratified_items():
    rng = np.random.default_rng(0)
    subjects = np.repeat(["a", "b", "c"], [700, 200, 100])
    for k in [1, 7, 10, 100]:
        idx = stratified_items(subjects, k, rng)
        assert len(idx) == k == len(set(idx)), k
    counts = np.unique(subjects[stratified_items(subjects, 100, rng)], return_counts=True)[1]
    assert (counts == [70, 20, 10]).all()
