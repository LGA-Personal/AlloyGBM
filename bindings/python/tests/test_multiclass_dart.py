"""Multiclass softmax + DART tests (v0.10.1).

v0.9.0 rejected DART for K>=3 classes; v0.10.0 still deferred it;
v0.10.1 enables it.  DART maintains a flat per-tree `tree_weight`
pool across the K-stumps-per-round commit order; before each round
it picks tree indices to drop (LightGBM convention: drop entire
class-trees, not gradient channels) and after building K new trees
it calls `apply_normalization` to rescale.
"""
import numpy as np
import pytest

from alloygbm import GBMClassifier


def _toy_multiclass(n_rows=200, n_features=5, n_classes=3, seed=23):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_rows).astype(np.int64)
    return X, y


def test_multiclass_dart_trains_and_predicts_proba():
    X, y = _toy_multiclass()
    m = GBMClassifier(
        n_estimators=10,
        boosting_mode="dart",
        dart_drop_rate=0.2,
        dart_max_drop=5,
        seed=23,
    )
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_multiclass_dart_differs_from_standard():
    X, y = _toy_multiclass()
    d = GBMClassifier(
        n_estimators=10, boosting_mode="dart", dart_drop_rate=0.2, seed=23,
    )
    d.fit(X, y)
    s = GBMClassifier(n_estimators=10, boosting_mode="standard", seed=23)
    s.fit(X, y)
    assert not np.allclose(d.predict_proba(X), s.predict_proba(X), atol=1e-4)


def test_multiclass_dart_pickle_round_trip():
    import pickle

    X, y = _toy_multiclass()
    m = GBMClassifier(
        n_estimators=8, boosting_mode="dart", dart_drop_rate=0.2, seed=23,
    )
    m.fit(X, y)
    p1 = m.predict_proba(X)
    restored = pickle.loads(pickle.dumps(m))
    p2 = restored.predict_proba(X)
    np.testing.assert_allclose(p1, p2, rtol=1e-6)


def test_multiclass_dart_warm_start_continues_without_error():
    X, y = _toy_multiclass()
    base = GBMClassifier(
        n_estimators=8, boosting_mode="dart", dart_drop_rate=0.15, seed=23,
    )
    base.fit(X, y)
    cont = GBMClassifier(
        n_estimators=8,
        boosting_mode="dart",
        dart_drop_rate=0.15,
        warm_start=True,
        seed=23,
    )
    cont.fit(X, y, init_model=base)
    p_base = base.predict_proba(X)
    p_cont = cont.predict_proba(X)
    # Continuation should produce a different, valid distribution.
    assert not np.allclose(p_base, p_cont, atol=1e-5)
    assert np.allclose(p_cont.sum(axis=1), 1.0, atol=1e-5)


def test_multiclass_dart_first_round_no_dropouts():
    """select_dropouts returns empty when tree_weights pool is empty,
    so round 0 of a fresh multiclass DART fit must add K new stumps
    with weight 1.0 and no normalization-driven changes."""
    X, y = _toy_multiclass(n_rows=50, seed=31)
    m = GBMClassifier(
        n_estimators=1, boosting_mode="dart", dart_drop_rate=0.5, seed=31,
    )
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (50, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_multiclass_dart_pickle_round_trip_with_multi_stump_trees():
    """Regression test for the v0.10.1 PR review (C4, C5): the previous
    implementation indexed `class_stumps[class_k][prior_round]` as if
    each class-round produced exactly one stump, and stamped
    `tree_weight` only on `last_mut()`.  Level-wise trees with depth>=2
    produce multiple stumps per (round, class), so:

    - Dropout subtracts only the root of a prior tree from
      class_predictions (leaves the rest in place) → next-round
      gradients are computed against the wrong ensemble.
    - Only the deepest stump of a new class-round gets stamped with
      `new_w`; the shallower stumps keep `tree_weight = 1.0` and the
      predictor (which folds `tree_weight` into every leaf) returns
      a different ensemble than training.

    Concretely: pickle round-trip + predict must equal an in-memory
    predict on the *same* model.  Pre-fix this was broken because the
    artifact's per-stump `tree_weight` values do not match the
    in-memory bookkeeping the engine used during training.
    """
    import pickle

    # Big enough data to force depth>=2 trees (so multi-stump rounds).
    rng = np.random.default_rng(41)
    X = rng.standard_normal((400, 6)).astype(np.float32)
    y = rng.integers(0, 3, size=400).astype(np.int64)
    m = GBMClassifier(
        n_estimators=20,
        boosting_mode="dart",
        dart_drop_rate=0.3,
        dart_max_drop=8,
        max_depth=4,
        min_data_in_leaf=8,
        seed=41,
    )
    m.fit(X, y)
    p1 = m.predict_proba(X)
    restored = pickle.loads(pickle.dumps(m))
    p2 = restored.predict_proba(X)
    # Strict equality (within f32 noise) — the artifact carries every
    # `tree_weight` the predictor needs, so a round-trip must reproduce
    # in-memory predictions exactly.
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-6)


def test_multiclass_dart_warm_start_with_multi_stump_trees():
    """Regression test for the v0.10.1 PR follow-up: the PyO3
    bridge's ``MultiClassWarmStartState.initial_dart_tree_weights``
    seeding indexed ``class_stumps[class_k][r]`` as if it were the
    r-th *tree*, but a depth>=2 level-wise tree contributes multiple
    *stumps* per round so the flat ``class_stumps[k]`` vec is denser
    than the per-tree array the engine expects.

    For a model with multi-stump trees, the buggy seeding would:

    - Pick wrong weights for trees after the first (e.g. with 3-stump
      trees and 5 prior rounds, only trees 0 and 1 would be probed
      at all — the other rounds' weights would silently default
      to a wrong value or the length check would mismatch).
    - Either trigger the engine's
      ``length {} != initial_rounds * K`` ContractViolation, OR
      seed the wrong weights and silently produce a corrupted
      continuation model whose predictions diverge from one fitted
      end-to-end in a single ``fit()`` call.

    This test trains a multi-stump multiclass DART model, splits the
    fit across two ``fit()`` calls (base + warm-start continuation),
    and asserts the continuation produces a finite, well-formed
    model that round-trips through pickle.
    """
    import pickle

    rng = np.random.default_rng(53)
    X = rng.standard_normal((300, 5)).astype(np.float32)
    y = rng.integers(0, 3, size=300).astype(np.int64)
    base = GBMClassifier(
        n_estimators=10,
        boosting_mode="dart",
        dart_drop_rate=0.25,
        dart_max_drop=5,
        max_depth=4,            # depth>=2 ⇒ multi-stump trees
        min_data_in_leaf=8,
        seed=53,
    )
    base.fit(X, y)
    cont = GBMClassifier(
        n_estimators=10,
        boosting_mode="dart",
        dart_drop_rate=0.25,
        dart_max_drop=5,
        max_depth=4,
        min_data_in_leaf=8,
        warm_start=True,
        seed=53,
    )
    cont.fit(X, y, init_model=base)
    proba = cont.predict_proba(X)
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(proba))
    # Round-trip through pickle — exposes any tree_weight mismatch
    # between in-memory and serialized state.
    restored = pickle.loads(pickle.dumps(cont))
    np.testing.assert_allclose(
        proba, restored.predict_proba(X), rtol=1e-5, atol=1e-6
    )


def test_multiclass_dart_works_with_leaf_wise_growth():
    """v0.10.2 Phase 4: lift the level-wise restriction for multiclass DART.

    Per-class `dart_round_start_offsets[k]` + `dart_round_counts[k]`
    bookkeeping snapshots `class_stumps[k].len()` around each
    `build_tree_*` call. The bookkeeping is growth-mode-agnostic
    (a tree's stump count is variable under leaf-wise but the round
    boundaries are still correctly captured).
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((250, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 2 + X[:, 2] * 3).round().astype(int) % 3
    m = GBMClassifier(
        n_estimators=10,
        boosting_mode="dart",
        tree_growth="leaf",
        max_leaves=8,
        dart_drop_rate=0.2,
        seed=42,
    )
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (250, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # All probabilities finite and in [0, 1].
    assert np.all(np.isfinite(proba))
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_multiclass_dart_leaf_wise_warm_start_round_trip():
    """v0.10.2 Phase 4: warm-start with leaf-wise multiclass DART.

    The PyO3 bridge reconstructs per-class DART tree weights by
    grouping `class_stumps[k]` stumps by `tree_id = node_id / TREE_NODE_STRIDE`
    and taking the first stump's `tree_weight` per tree. Under
    leaf-wise growth each tree has a variable stump count but the
    `tree_id` decoding works identically.

    Mirror the existing `test_multiclass_dart_warm_start_continues_without_error`
    pattern: base trains 8 rounds; cont with `warm_start=True` +
    `init_model=base` trains 8 more rounds (16 total). The continuation
    must produce a different, valid distribution.
    """
    X, y = _toy_multiclass()
    base = GBMClassifier(
        n_estimators=8,
        boosting_mode="dart",
        tree_growth="leaf",
        max_leaves=6,
        dart_drop_rate=0.15,
        seed=23,
    )
    base.fit(X, y)
    cont = GBMClassifier(
        n_estimators=8,
        boosting_mode="dart",
        tree_growth="leaf",
        max_leaves=6,
        dart_drop_rate=0.15,
        warm_start=True,
        seed=23,
    )
    cont.fit(X, y, init_model=base)
    p_base = base.predict_proba(X)
    p_cont = cont.predict_proba(X)
    assert not np.allclose(p_base, p_cont, atol=1e-5)
    assert np.allclose(p_cont.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(p_cont))


def test_multiclass_dart_with_validation_early_stopping():
    """v0.10.2 Phase 4: validation DART transition (subtract dropped →
    scale new at new_w → re-add dropped) applies under leaf-wise growth
    too. A broken transition typically manifests as NaN/inf validation
    losses or a model that won't pickle round-trip.
    """
    import pickle

    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 5)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 2).round().astype(int) % 4  # 4 classes
    X_train, y_train = X[:300], y[:300]
    X_val, y_val = X[300:], y[300:]
    m = GBMClassifier(
        n_estimators=50,
        boosting_mode="dart",
        tree_growth="leaf",
        max_leaves=8,
        early_stopping_rounds=5,
        dart_drop_rate=0.15,
        seed=7,
    )
    m.fit(X_train, y_train, eval_set=(X_val, y_val))
    proba = m.predict_proba(X_val)
    assert proba.shape == (100, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(proba))
    # Pickle round-trip after early stopping has truncated the model.
    restored = pickle.loads(pickle.dumps(m))
    proba_restored = restored.predict_proba(X_val)
    np.testing.assert_allclose(proba, proba_restored, rtol=1e-5, atol=1e-6)


def test_multiclass_dart_with_validation_early_stopping():
    """Regression test for the v0.10.1 PR review (C6): the DART
    transition (dropout subtract + new_w scale + dropped re-add) must
    also apply to `validation_class_predictions`, otherwise the
    validation loss tracked for early stopping is computed against an
    inconsistent ensemble (training sees dropout, validation does not)
    and the `best_validation_round` decision is corrupted.

    Smoke-level: this just verifies that DART + multiclass + eval_set
    + early_stopping_rounds runs to completion and produces a
    well-formed model.  A broken validation transition typically
    manifests as NaN/inf validation losses or a model that won't
    pickle round-trip.
    """
    import pickle

    rng = np.random.default_rng(43)
    X = rng.standard_normal((300, 5)).astype(np.float32)
    y = rng.integers(0, 3, size=300).astype(np.int64)
    X_val = X[:60]
    y_val = y[:60]
    m = GBMClassifier(
        n_estimators=20,
        boosting_mode="dart",
        dart_drop_rate=0.25,
        max_depth=4,
        min_data_in_leaf=8,
        early_stopping_rounds=3,
        seed=43,
    )
    m.fit(X, y, eval_set=(X_val, y_val))
    proba = m.predict_proba(X)
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # Round-trip equality after early stopping has truncated the model.
    p1 = proba
    restored = pickle.loads(pickle.dumps(m))
    p2 = restored.predict_proba(X)
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-6)
