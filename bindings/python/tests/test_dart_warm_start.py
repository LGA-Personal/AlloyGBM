"""DART + warm_start tests (v0.10.0).

v0.9.0 rejected this combination; v0.10.0 enables it. Continuation seeds
``dart_state.tree_weights`` from the prior model's per-stump ``tree_weight``
and starts fresh dropout bookkeeping for new rounds (historical
``dropped_per_round`` is not persisted by design — RNG-driven dropout
history cannot be replayed).
"""
import numpy as np
import pytest

from alloygbm import GBMRegressor


def _toy_regression(n_rows=80, n_features=3, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    y = (X @ rng.standard_normal(n_features) + 0.05 * rng.standard_normal(n_rows)).astype(
        np.float32
    )
    return X, y


def test_dart_warm_start_continues_training_without_error():
    """v0.9.0 raised NotImplementedError here; v0.10.0 must succeed."""
    X, y = _toy_regression()
    base = GBMRegressor(
        n_estimators=10,
        boosting_mode="dart",
        dart_drop_rate=0.1,
        seed=7,
    )
    base.fit(X, y)
    cont = GBMRegressor(
        n_estimators=10,  # 10 additional rounds on top of the prior 10
        boosting_mode="dart",
        dart_drop_rate=0.1,
        warm_start=True,
        seed=7,
    )
    # Should not raise. (v0.9.0 raised
    # "boosting_mode='dart' + warm_start is not yet supported".)
    cont.fit(X, y, init_model=base)
    # Continuation must have produced at least one new round on top of base.
    assert cont.n_estimators_ >= 1


def test_dart_warm_start_predictions_change_after_extra_rounds():
    X, y = _toy_regression()
    base = GBMRegressor(
        n_estimators=10, boosting_mode="dart", dart_drop_rate=0.1, seed=7,
    )
    base.fit(X, y)
    base_preds = np.asarray(base.predict(X), dtype=np.float32)
    cont = GBMRegressor(
        n_estimators=30,
        boosting_mode="dart",
        dart_drop_rate=0.1,
        warm_start=True,
        seed=7,
    )
    cont.fit(X, y, init_model=base)
    cont_preds = np.asarray(cont.predict(X), dtype=np.float32)
    # Extra rounds should change predictions noticeably.
    assert np.linalg.norm(cont_preds - base_preds) > 1e-4
