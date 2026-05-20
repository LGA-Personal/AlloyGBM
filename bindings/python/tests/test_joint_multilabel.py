"""MultiLabelGBMRanker(training_mode='joint') tests (v0.10.1).

v0.10.0 shipped the Rust joint trainer (`fit_joint_multi_output`) and the
`MultiOutputLeafValues` artifact section, but `MultiLabelGBMRanker` still
routed every fit to the independent-per-label fallback.  v0.10.1 wires
`training_mode='joint'` through to the joint trainer.
"""
import numpy as np
import pytest

from alloygbm import MultiLabelGBMRanker


def _toy_ranking(n_queries=8, items_per_query=5, n_features=4, n_labels=3, seed=11):
    rng = np.random.default_rng(seed)
    n = n_queries * items_per_query
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, 4, size=(n, n_labels)).astype(np.float32)
    group = np.full(n_queries, items_per_query, dtype=np.int32)
    return X, y, group


def test_joint_mode_fits_and_predicts_with_correct_shape():
    X, y, group = _toy_ranking()
    m = MultiLabelGBMRanker(n_estimators=4, training_mode="joint")
    m.fit(X, y, group=group)
    preds = m.predict(X)
    assert preds.shape == (X.shape[0], y.shape[1])
    assert np.all(np.isfinite(preds))


def test_default_training_mode_is_independent():
    """Backward compatibility: callers that don't pass training_mode must
    still hit the v0.7.1 independent-per-label fallback (which uses
    GBMRanker under the hood)."""
    m = MultiLabelGBMRanker(n_estimators=2)
    assert m.training_mode == "independent"


def test_invalid_training_mode_rejected():
    with pytest.raises(ValueError, match="training_mode"):
        MultiLabelGBMRanker(training_mode="nonsense")
