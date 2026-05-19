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


def test_dart_warm_start_preserves_prior_ensemble_contributions():
    """Review fix (Comment 1): a DART continuation must start from the prior
    model's weighted prediction surface, not from an all-weights-1 ensemble
    (or worse, an empty ensemble).

    We verify this by training a base DART model with enough rounds that
    several stumps end up with non-unit ``tree_weight`` after dropout
    normalization. Then we continue with a tiny ``learning_rate`` and a
    single new round so the new round's contribution is negligible (~lr per
    leaf). The continuation's predictions must equal ``base.predict(X)`` to
    within that negligible delta.

    If the warm-start path failed to seed ``dart_state.tree_weights`` from
    the prior fit (or if ``apply_round_stumps_tree_walk`` ignored
    ``stump.tree_weight``), the continuation's initial prediction surface
    would be the unweighted sum of prior leaves and the parity gap would
    be far larger than the lr-scaled new contribution.
    """
    X, y = _toy_regression(n_rows=120, seed=11)

    # Use settings that almost guarantee some non-unit DART tree weights:
    # higher drop_rate + enough rounds.
    base = GBMRegressor(
        n_estimators=30,
        boosting_mode="dart",
        dart_drop_rate=0.3,
        dart_max_drop=5,
        seed=11,
    )
    base.fit(X, y)
    base_preds = np.asarray(base.predict(X), dtype=np.float32)

    # Sanity: at least one stump should carry a non-1.0 tree weight after
    # dropout normalization. (Otherwise this test cannot distinguish the bug
    # from correct behaviour.) We can't probe the internal artifact directly
    # from Python, but we can probe indirectly: if all weights were 1.0,
    # base predictions for a deterministic seeded fit would equal a Standard
    # fit on the same seed. They typically diverge.
    standard_baseline = GBMRegressor(
        n_estimators=30, boosting_mode="standard", seed=11
    )
    standard_baseline.fit(X, y)
    std_preds = np.asarray(standard_baseline.predict(X), dtype=np.float32)
    assert np.linalg.norm(base_preds - std_preds) > 1e-3, (
        "test fixture failure: DART base did not produce a different "
        "prediction surface than Standard; cannot validate warm-start parity"
    )

    # Continue with 1 extra round at a tiny LR so the new contribution is
    # ~tiny_lr * one_leaf_value per row. The total delta from base must be
    # tiny if warm-start correctly carries over the prior weighted ensemble.
    # Use boosting_mode="standard" for the continuation to isolate the
    # warm-start prediction-seeding code path (which uses
    # `apply_round_stumps_tree_walk` and must multiply leaf values by the
    # prior stumps' `tree_weight`). The new round adds a single
    # tree_weight=1.0 stump scaled by tiny_lr.
    tiny_lr = 1e-6
    cont = GBMRegressor(
        n_estimators=1,
        boosting_mode="standard",
        warm_start=True,
        learning_rate=tiny_lr,
        seed=11,
    )
    cont.fit(X, y, init_model=base)
    cont_preds = np.asarray(cont.predict(X), dtype=np.float32)

    # The per-row delta from base should be dominated by the tiny-lr new
    # round contribution. Absolute target magnitudes are ~O(1) for the
    # synthetic dataset, so leaf values are bounded by O(1) and the delta
    # bound is roughly `tiny_lr * O(1) = 1e-6`. Allow a generous tolerance
    # to absorb numerical noise but reject the obviously-broken case where
    # cont_preds collapses toward zero (which is what would happen if the
    # warm-start prior were ignored).
    per_row_delta = np.abs(cont_preds - base_preds)
    max_delta = per_row_delta.max()
    assert max_delta < 1e-3, (
        f"DART warm-start lost prior ensemble contribution: "
        f"max |cont - base| = {max_delta:.6f} but expected << 1e-3 "
        f"(would be O(|base_preds|) if warm-start were broken; "
        f"base_preds range = [{base_preds.min():.3f}, {base_preds.max():.3f}])"
    )


def test_dart_warm_start_with_eval_set_early_stopping_stays_coherent():
    """Review follow-up: warm_start + eval_set + early_stopping_rounds must
    truncate against NEW-round stump counts, not prior-round counts. An
    earlier draft of the warm-start fix placed prior-round counts at the
    front of ``stumps_per_completed_round``, which made
    ``best_round``-indexed truncation consume the wrong counts and silently
    keep the wrong stumps. With the correct fix, this combination produces
    a coherent model that round-trips through pickle and predicts sanely.
    """
    import pickle

    X, y = _toy_regression(n_rows=120, seed=13)
    X_val = X[:40]
    y_val = y[:40]

    base = GBMRegressor(
        n_estimators=15,
        boosting_mode="dart",
        dart_drop_rate=0.2,
        seed=13,
    )
    base.fit(X, y)
    base_preds_before = np.asarray(base.predict(X[:5]), dtype=np.float32)

    cont = GBMRegressor(
        n_estimators=10,
        boosting_mode="dart",
        dart_drop_rate=0.2,
        warm_start=True,
        early_stopping_rounds=3,
        seed=13,
    )
    cont.fit(X, y, init_model=base, eval_set=(X_val, y_val))

    # Continuation must produce finite, non-pathological predictions —
    # a broken truncation would either panic above (no model emitted) or
    # produce a corrupted model whose predictions degrade obviously.
    cont_preds = np.asarray(cont.predict(X[:5]), dtype=np.float32)
    assert np.all(np.isfinite(cont_preds)), "continuation predictions not finite"

    # Round-trip through pickle: would expose any stump-count mismatch
    # between in-memory model and serialized artifact (a broken truncation
    # could keep extra/missing stumps that the artifact format rejects).
    blob = pickle.dumps(cont)
    restored = pickle.loads(blob)
    restored_preds = np.asarray(restored.predict(X[:5]), dtype=np.float32)
    np.testing.assert_allclose(cont_preds, restored_preds, rtol=1e-5)

    # Best-iteration bookkeeping should be sane. n_estimators_ reflects the
    # number of new rounds actually kept after early stopping; should be in
    # [0, 10] (we asked for up to 10 new rounds; early stopping may keep
    # zero if the base model already won validation). The key check is just
    # that the field is a plausible non-negative integer.
    assert 0 <= cont.n_estimators_ <= 10, (
        f"n_estimators_={cont.n_estimators_} out of expected [0, 10] range"
    )

    # Base model was not mutated.
    base_preds_after = np.asarray(base.predict(X[:5]), dtype=np.float32)
    np.testing.assert_allclose(base_preds_before, base_preds_after, rtol=1e-7)
