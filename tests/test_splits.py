import numpy as np

from cabench.splits import split_models


def test_family_split_keeps_groups_together():
    groups = np.repeat(np.arange(30), np.random.default_rng(0).integers(1, 8, 30))
    tr, va, te = split_models(len(groups), 0, groups=groups)
    assert len(tr) + len(va) + len(te) == len(groups) == len(set(np.r_[tr, va, te]))
    assert not set(groups[tr]) & set(groups[va])
    assert not set(groups[tr]) & set(groups[te])
    assert not set(groups[va]) & set(groups[te])
