"""n_estimators="auto": round count and learning rate chosen by early stopping."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor, MultiLabelGBMRanker
from alloygbm._regressor import _auto_rounds


def _regression(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + rng.normal(scale=0.3, size=n)
    return X, y


def test_defaults_are_auto_for_every_single_output_estimator():
    for cls in (GBMRegressor, GBMClassifier, GBMRanker):
        params = cls().get_params()
        assert params["n_estimators"] == "auto"
        assert params["learning_rate"] == "auto"
        assert "n_estimators='auto'" in repr(cls())


def test_holdout_mode_refits_on_all_rows_with_scaled_rounds():
    X, y = _regression()
    model = GBMRegressor(seed=3).fit(X, y)
    info = model.auto_rounds_
    assert info["mode"] == "holdout"
    assert info["learning_rate"] == model.learning_rate_ == _auto_rounds.AUTO_LEARNING_RATE
    assert info["validation_rows"] == 60
    assert info["validation_fraction"] == pytest.approx(0.1)
    assert info["refit_rounds"] == _auto_rounds.refit_rounds(
        info["holdout_best_round"], info["validation_fraction"]
    )
    assert model.n_estimators_ == info["refit_rounds"]
    # Parameters are left as the caller set them.
    assert model.n_estimators == "auto" and model.learning_rate == "auto"


def test_holdout_mode_is_deterministic_per_seed():
    X, y = _regression()
    a = GBMRegressor(seed=5).fit(X, y).predict(X)
    b = GBMRegressor(seed=5).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_explicit_rounds_are_respected_with_legacy_learning_rate():
    X, y = _regression()
    model = GBMRegressor(n_estimators=37).fit(X, y)
    assert model.auto_rounds_["mode"] == "explicit"
    assert model.n_estimators_ == 37
    assert model.learning_rate_ == pytest.approx(_auto_rounds.LEGACY_LEARNING_RATE)
    explicit_rate = GBMRegressor(n_estimators=10, learning_rate=0.3).fit(X, y)
    assert explicit_rate.learning_rate_ == pytest.approx(0.3)


def test_explicit_learning_rate_is_used_by_the_holdout():
    X, y = _regression()
    model = GBMRegressor(learning_rate=0.2).fit(X, y)
    assert model.auto_rounds_["mode"] == "holdout"
    assert model.learning_rate_ == pytest.approx(0.2)


def test_eval_set_drives_early_stopping_without_refit():
    X, y = _regression(800)
    model = GBMRegressor(early_stopping_rounds=7).fit(
        X[:600], y[:600], eval_set=(X[600:], y[600:])
    )
    info = model.auto_rounds_
    assert info["mode"] == "eval_set"
    assert info["patience"] == 7
    assert info["round_cap"] == _auto_rounds.AUTO_ROUND_CAP
    assert model.best_iteration_ is not None
    assert model.n_estimators_ < _auto_rounds.AUTO_ROUND_CAP


def test_early_stopping_rounds_sets_holdout_patience():
    X, y = _regression()
    model = GBMRegressor(early_stopping_rounds=5).fit(X, y)
    assert model.auto_rounds_["mode"] == "holdout"
    assert model.auto_rounds_["patience"] == 5


@pytest.mark.parametrize(
    "params",
    [
        {"training_policy": "manual"},
        {"boosting_mode": "dart"},
        {"training_mode": "morph"},
        {"lr_schedule": "warmup_cosine"},
    ],
)
def test_unsupported_configurations_keep_fixed_defaults(params):
    X, y = _regression(300)
    model = GBMRegressor(**params).fit(X, y)
    assert model.auto_rounds_["mode"] == "legacy"
    assert model.learning_rate_ == pytest.approx(_auto_rounds.LEGACY_LEARNING_RATE)


def test_early_stopping_rounds_without_a_holdout_is_ignored_in_the_fallback():
    X, y = _regression(_auto_rounds.AUTO_MIN_ROWS - 1)
    model = GBMRegressor(early_stopping_rounds=5).fit(X, y)
    assert model.auto_rounds_["mode"] == "legacy"
    warm = GBMRegressor(early_stopping_rounds=5, warm_start=True)
    warm.fit(*_regression())
    warm.fit(*_regression())
    assert warm.auto_rounds_["mode"] == "legacy"
    # An explicit round count keeps the existing contract.
    with pytest.raises(ValueError, match="early_stopping_rounds requires eval_set"):
        GBMRegressor(n_estimators=10, early_stopping_rounds=5).fit(X, y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eval_sample_weight": np.ones(10)},
        {"eval_group": np.zeros(10)},
        {"eval_metric": lambda y_true, y_pred: 0.0},
    ],
)
def test_orphan_eval_arguments_fail_before_the_holdout(kwargs):
    X, y = _regression()
    with pytest.raises(ValueError, match="requires eval_set"):
        GBMRegressor().fit(X, y, **kwargs)


def test_row_count_mismatch_fails_before_the_holdout():
    X, y = _regression()
    for rows in (X[:-1], np.vstack([X, X[:1]])):
        with pytest.raises(ValueError, match="same number of rows"):
            GBMRegressor().fit(rows, y)
        with pytest.raises(ValueError, match="same number of rows"):
            GBMRegressor().fit(rows.tolist(), y)


def test_manual_policy_trains_exactly_the_legacy_round_count():
    X, y = _regression(300)
    model = GBMRegressor(training_policy="manual").fit(X, y)
    assert model.n_estimators_ == _auto_rounds.LEGACY_N_ESTIMATORS


def test_tiny_datasets_keep_fixed_defaults():
    X, y = _regression(_auto_rounds.AUTO_MIN_ROWS - 1)
    assert GBMRegressor().fit(X, y).auto_rounds_["mode"] == "legacy"


def test_warm_start_continues_with_the_first_fits_learning_rate():
    X, y = _regression()
    model = GBMRegressor(warm_start=True, seed=1)
    model.fit(X, y)
    first_rounds = model.n_estimators_
    assert model.auto_rounds_["mode"] == "holdout"
    model.fit(X, y)
    assert model.auto_rounds_["mode"] == "legacy"
    assert model.learning_rate_ == pytest.approx(_auto_rounds.AUTO_LEARNING_RATE)
    # A warm-started fit reports the rounds it added.
    assert first_rounds > 0
    assert model.n_estimators_ == _auto_rounds.LEGACY_N_ESTIMATORS


def test_warm_start_holdout_refit_starts_from_scratch():
    # Under warm_start=True the refit must not continue from the holdout model.
    X, y = _regression()
    warm = GBMRegressor(warm_start=True, seed=2).fit(X, y)
    cold = GBMRegressor(seed=2).fit(X, y)
    np.testing.assert_array_equal(warm.predict(X), cold.predict(X))


def test_binary_and_multiclass_classifiers_use_the_holdout():
    X, y = _regression(900)
    binary = GBMClassifier().fit(X, (y > 0).astype(int))
    assert binary.auto_rounds_["mode"] == "holdout"
    labels = np.digitize(y, [-1.0, 0.0, 1.0])
    multi = GBMClassifier().fit(X, labels)
    assert multi.auto_rounds_["mode"] == "holdout"
    proba = multi.predict_proba(X)
    assert proba.shape == (900, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_ranker_uses_a_group_holdout():
    X, y = _regression(600)
    group = np.repeat(np.arange(60), 10)
    model = GBMRanker().fit(X, (y > 0).astype(int), group=group)
    assert model.auto_rounds_["mode"] == "holdout"
    assert model.auto_rounds_["validation_rows"] % 10 == 0


def test_sample_weights_are_split_with_the_rows():
    X, y = _regression()
    weights = np.random.default_rng(0).uniform(0.5, 2.0, size=len(y))
    model = GBMRegressor().fit(X, y, sample_weight=weights)
    assert model.auto_rounds_["mode"] == "holdout"


def test_refit_scales_by_the_share_actually_held_out():
    # Every class keeps at least one validation row, so 25 two-row classes
    # hold out half the data, not 10%.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    labels = np.repeat(np.arange(25), 2)
    info = GBMClassifier(max_depth=2).fit(X, labels).auto_rounds_
    assert info["mode"] == "holdout"
    assert info["validation_rows"] == 25
    assert info["validation_fraction"] == pytest.approx(0.5)
    assert info["refit_rounds"] == _auto_rounds.refit_rounds(info["holdout_best_round"], 0.5)


def test_holdout_without_weight_on_one_side_keeps_fixed_defaults():
    X, y = _regression(_auto_rounds.AUTO_MIN_ROWS + 10)
    weights = np.zeros(len(y))
    weights[0] = 1.0
    model = GBMRegressor().fit(X, y, sample_weight=weights)
    assert model.auto_rounds_["mode"] == "legacy"
    assert np.isfinite(model.predict(X)).all()


def test_save_load_keeps_the_learning_rate_for_continuation(tmp_path):
    X, y = _regression()
    model = GBMRegressor(seed=4).fit(X, y)
    path = tmp_path / "model.alloy"
    model.save_model(str(path))
    loaded = GBMRegressor.load_model(str(path))
    assert loaded.learning_rate_ == pytest.approx(_auto_rounds.AUTO_LEARNING_RATE)
    assert loaded.auto_rounds_ == model.auto_rounds_
    resumed = GBMRegressor().fit(X, y, init_model=loaded)
    assert resumed.auto_rounds_["mode"] == "legacy"
    assert resumed.learning_rate_ == pytest.approx(_auto_rounds.AUTO_LEARNING_RATE)


def test_dataframe_input_is_split_by_position():
    pd = pytest.importorskip("pandas")
    X, y = _regression()
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])], index=np.arange(len(y)) * 3)
    from_frame = GBMRegressor().fit(frame, y).predict(X)
    from_array = GBMRegressor().fit(X, y).predict(X)
    np.testing.assert_allclose(from_frame, from_array)


def test_invalid_auto_values_are_rejected():
    with pytest.raises(ValueError, match="n_estimators"):
        GBMRegressor(n_estimators="fast").fit(*_regression(100))
    with pytest.raises(ValueError, match="learning_rate"):
        GBMRegressor(learning_rate="fast").fit(*_regression(100))
    with pytest.raises(ValueError, match="n_estimators"):
        GBMRegressor(n_estimators=0).fit(*_regression(100))
    model = GBMRegressor(n_estimators=5).set_params(n_estimators="auto", learning_rate="auto")
    assert model.get_params()["n_estimators"] == "auto"


def test_pickle_keeps_the_auto_rounds_record():
    X, y = _regression()
    model = GBMRegressor().fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.auto_rounds_ == model.auto_rounds_
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_multi_label_ranker_keeps_fixed_round_defaults():
    X, y = _regression(200)
    labels = np.column_stack([(y > 0).astype(int), (y > 1).astype(int)])
    group = np.repeat(np.arange(20), 10)
    for mode in ("independent", "joint"):
        model = MultiLabelGBMRanker(multi_label_mode=mode, max_depth=2).fit(
            X, labels, group=group
        )
        assert np.asarray(model.predict(X)).shape == (200, 2)


def test_holdout_split_stratifies_classes():
    labels = np.array([0] * 90 + [1] * 10)
    train, valid = _auto_rounds.holdout_split(100, 0.1, 0, class_labels=labels)
    assert len(valid) == 10
    assert sorted(np.bincount(labels[valid]).tolist()) == [1, 9]
    assert len(np.intersect1d(train, valid)) == 0
    assert _auto_rounds.holdout_split(3, 0.1, 0, class_labels=np.array([0, 0, 1])) is None


def test_holdout_split_keeps_groups_whole():
    groups = np.repeat(np.arange(25), 4)
    train, valid = _auto_rounds.holdout_split(100, 0.2, 7, group_ids=groups)
    assert set(groups[train]).isdisjoint(set(groups[valid]))
    assert len(set(groups[valid])) == 5
    assert _auto_rounds.holdout_split(4, 0.5, 0, group_ids=np.zeros(4)) is None


def test_relative_tolerance_ignores_a_flat_tail():
    curve = [1.0, 0.5, 0.4, 0.39999, 0.39998, 0.39997, 0.39996]
    assert _auto_rounds.best_round_with_relative_tolerance(curve, 3, 1e-3) == 3
    assert _auto_rounds.best_round_with_relative_tolerance(curve, 3, 0.0) == 7


def _record_native_relative_tolerance(monkeypatch):
    """Spy on the native summary trainers; record each call's relative tolerance."""
    import alloygbm._alloygbm as native

    seen: list[float] = []
    for name in (
        "train_regression_artifact_dense_with_summary_bytes",
        "train_regression_artifact_dense_with_summary",
        "train_regression_artifact_with_summary",
    ):
        original = getattr(native, name)

        def spy(*args, _original=original, **kwargs):
            seen.append(kwargs.get("early_stopping_relative_tolerance", 0.0))
            return _original(*args, **kwargs)

        monkeypatch.setattr(native, name, spy)
    return seen


def test_engine_stops_the_holdout_under_the_relative_rule(monkeypatch):
    seen = _record_native_relative_tolerance(monkeypatch)
    X, y = _regression()
    GBMRegressor(seed=3).fit(X, y)
    # Holdout pass, then the refit (which has no validation to stop on).
    assert seen == [_auto_rounds.AUTO_RELATIVE_TOLERANCE, 0.0]

    seen.clear()
    GBMRegressor(n_estimators=20).fit(X, y)
    GBMRegressor(seed=3).fit(X, y, eval_set=(X[:50], y[:50]))
    assert seen == [0.0, 0.0]


@pytest.mark.parametrize(
    ("cls", "make"),
    [
        (GBMRegressor, lambda: _regression(n=800, seed=4)),
        (
            GBMClassifier,
            lambda: (
                (lambda X: (X, (X[:, 0] + X[:, 1] > 0).astype(int)))(
                    np.random.default_rng(5).normal(size=(800, 6))
                )
            ),
        ),
        (
            GBMClassifier,
            lambda: (
                (lambda X: (X, np.digitize(X[:, 0] + 0.5 * X[:, 1], [-0.7, 0.0, 0.7])))(
                    np.random.default_rng(6).normal(size=(800, 6))
                )
            ),
        ),
    ],
)
def test_engine_relative_stop_keeps_the_chosen_rounds(monkeypatch, cls, make):
    """Stopping the holdout early in the engine must not move the round count."""
    X, y = make()
    fast = cls(seed=1).fit(X, y)
    monkeypatch.setattr(cls, "_early_stopping_relative_kwargs", lambda self: {})
    slow = cls(seed=1).fit(X, y)
    for key in ("holdout_best_round", "refit_rounds"):
        assert fast.auto_rounds_[key] == slow.auto_rounds_[key]
    predict = "predict_proba" if cls is GBMClassifier else "predict"
    np.testing.assert_array_equal(getattr(fast, predict)(X), getattr(slow, predict)(X))
