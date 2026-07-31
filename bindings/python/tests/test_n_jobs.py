from __future__ import annotations

import inspect
import pickle

import numpy as np
import pytest
from sklearn.base import clone

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor, MultiLabelGBMRanker
from alloygbm import _alloygbm as _native
from alloygbm._regressor import _base as _regressor_base


@pytest.mark.parametrize(
    "estimator_factory",
    [
        GBMRegressor,
        GBMClassifier,
        GBMRanker,
        MultiLabelGBMRanker,
    ],
)
@pytest.mark.parametrize("n_jobs", [None, -1, 1, 3])
def test_estimators_retain_supported_n_jobs(estimator_factory, n_jobs):
    estimator = estimator_factory(n_jobs=n_jobs)

    assert estimator.get_params()["n_jobs"] == n_jobs
    assert clone(estimator).get_params()["n_jobs"] == n_jobs
    assert pickle.loads(pickle.dumps(estimator)).get_params()["n_jobs"] == n_jobs
    assert "n_jobs=" in repr(estimator)


@pytest.mark.parametrize(
    "estimator_factory",
    [
        GBMRegressor,
        GBMClassifier,
        GBMRanker,
        MultiLabelGBMRanker,
    ],
)
@pytest.mark.parametrize(
    "n_jobs",
    [True, False, 0, -2, 1.5, "2", object()],
    ids=["true", "false", "zero", "below-minus-one", "float", "string", "object"],
)
def test_estimators_reject_invalid_n_jobs(estimator_factory, n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        estimator_factory(n_jobs=n_jobs)


@pytest.mark.parametrize(
    "estimator_factory",
    [
        GBMRegressor,
        GBMClassifier,
        GBMRanker,
        MultiLabelGBMRanker,
    ],
)
def test_set_params_validates_and_retains_n_jobs(estimator_factory):
    estimator = estimator_factory()

    assert estimator.set_params(n_jobs=2) is estimator
    assert estimator.get_params()["n_jobs"] == 2
    with pytest.raises(ValueError, match="n_jobs"):
        estimator.set_params(n_jobs=0)
    assert estimator.get_params()["n_jobs"] == 2


def test_regressor_classifier_and_ranker_signatures_expose_n_jobs():
    for estimator_type in (GBMRegressor, GBMClassifier, GBMRanker):
        parameter = inspect.signature(estimator_type).parameters["n_jobs"]
        assert parameter.default is None
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_native_training_signatures_expose_n_jobs():
    functions = (
        _native.train_regression_artifact,
        _native.train_regression_artifact_dense,
        _native.train_regression_artifact_with_summary,
        _native.train_regression_artifact_dense_with_summary,
        _native.train_regression_artifact_dense_with_summary_bytes,
        _native.train_joint_multi_label_ranker,
    )
    for function in functions:
        parameter = inspect.signature(function).parameters["n_jobs"]
        assert parameter.default in (None, Ellipsis)


@pytest.mark.parametrize("n_jobs", [True, False, 0, -2])
def test_native_training_bridge_rejects_invalid_n_jobs(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        _native.train_regression_artifact(
            rows=[[0.0], [1.0], [2.0], [3.0]],
            targets=[0.0, 1.0, 2.0, 3.0],
            learning_rate=0.1,
            max_depth=2,
            row_subsample=1.0,
            col_subsample=1.0,
            min_validation_improvement=0.0,
            seed=3,
            deterministic=True,
            rounds=1,
            n_jobs=n_jobs,
        )


@pytest.mark.parametrize(
    "estimator_type",
    [GBMRegressor, GBMClassifier, GBMRanker],
)
def test_legacy_estimator_pickle_state_defaults_n_jobs(estimator_type):
    estimator = estimator_type(n_jobs=2)
    state = estimator.__getstate__()
    state.pop("n_jobs")

    restored = estimator_type.__new__(estimator_type)
    restored.__setstate__(state)

    assert restored.n_jobs is None
    assert restored.get_params()["n_jobs"] is None
    assert "n_jobs=None" in repr(restored)


def test_legacy_multilabel_pickle_state_defaults_n_jobs():
    estimator = MultiLabelGBMRanker(n_jobs=2)
    state = estimator.__getstate__()
    state["_per_label_kwargs"].pop("n_jobs")

    restored = MultiLabelGBMRanker.__new__(MultiLabelGBMRanker)
    restored.__setstate__(state)

    assert restored.get_params()["n_jobs"] is None
    assert "n_jobs=None" in repr(restored)


def test_legacy_independent_multilabel_bundle_defaults_n_jobs(tmp_path):
    X = np.arange(16, dtype=np.float32).reshape(8, 2)
    y = np.column_stack(
        [
            np.linspace(0.0, 1.0, 8, dtype=np.float32),
            np.linspace(1.0, 0.0, 8, dtype=np.float32),
        ]
    )
    group = np.repeat(np.arange(4, dtype=np.int64), 2)
    estimator = MultiLabelGBMRanker(
        ranking_labels=["forward", "reverse"],
        n_estimators=2,
        max_depth=1,
        n_jobs=2,
    ).fit(X, y, group=group)
    estimator._per_label_kwargs.pop("n_jobs")
    for ranker in estimator._sub_rankers:
        del ranker.n_jobs
    path = tmp_path / "legacy-independent.mlranker"
    estimator.save_model(str(path))

    restored = MultiLabelGBMRanker.load_model(str(path))

    assert restored.get_params()["n_jobs"] is None
    assert all(ranker.n_jobs is None for ranker in restored.sub_rankers_)
    np.testing.assert_array_equal(restored.predict(X), estimator.predict(X))


def test_regressor_forwards_n_jobs_to_native_summary_bridge(monkeypatch):
    seen: list[int | None] = []
    original_loader = (
        _regressor_base._load_native_train_regression_artifact_with_summary
    )

    def recording_loader():
        native_function = original_loader()

        def recording_bridge(*args, **kwargs):
            seen.append(kwargs.get("n_jobs"))
            return native_function(*args, **kwargs)

        return recording_bridge

    monkeypatch.setattr(
        _regressor_base,
        "_load_native_train_regression_artifact_with_summary",
        recording_loader,
    )
    model = GBMRegressor(n_estimators=3, max_depth=2, n_jobs=2)
    model.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0])

    assert seen == [2]


def _assert_fit_equivalent(single, parallel, X, y, **fit_kwargs):
    single.fit(X, y, **fit_kwargs)
    parallel.fit(X, y, **fit_kwargs)

    if isinstance(single, MultiLabelGBMRanker):
        if single.multi_label_mode == "joint":
            assert single._joint_artifact_bytes == parallel._joint_artifact_bytes
        else:
            assert [
                bytes(ranker.artifact_bytes) for ranker in single.sub_rankers_
            ] == [
                bytes(ranker.artifact_bytes) for ranker in parallel.sub_rankers_
            ]
    else:
        assert bytes(single.artifact_bytes) == bytes(parallel.artifact_bytes)
    np.testing.assert_array_equal(single.predict(X), parallel.predict(X))


def test_standard_fit_paths_are_exact_across_thread_counts():
    X = np.arange(96, dtype=np.float32).reshape(32, 3)
    regression_y = (0.25 * X[:, 0] - 0.1 * X[:, 2]).astype(np.float32)
    binary_y = (np.arange(32) % 2).astype(np.int32)
    multiclass_y = (np.arange(32) % 4).astype(np.int32)
    group = np.repeat(np.arange(8), 4)
    ranking_y = (np.arange(32) % 4).astype(np.float32)

    _assert_fit_equivalent(
        GBMRegressor(n_estimators=4, max_depth=2, seed=9, n_jobs=1),
        GBMRegressor(n_estimators=4, max_depth=2, seed=9, n_jobs=2),
        X,
        regression_y,
    )
    _assert_fit_equivalent(
        GBMClassifier(n_estimators=4, max_depth=2, seed=9, n_jobs=1),
        GBMClassifier(n_estimators=4, max_depth=2, seed=9, n_jobs=2),
        X,
        binary_y,
    )
    _assert_fit_equivalent(
        GBMClassifier(n_estimators=4, max_depth=2, seed=9, n_jobs=1),
        GBMClassifier(n_estimators=4, max_depth=2, seed=9, n_jobs=2),
        X,
        multiclass_y,
    )
    _assert_fit_equivalent(
        GBMRanker(n_estimators=4, max_depth=2, seed=9, n_jobs=1),
        GBMRanker(n_estimators=4, max_depth=2, seed=9, n_jobs=2),
        X,
        ranking_y,
        group=group,
    )


def _eligible_multiclass_fixture():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(1_024, 4)).astype(np.float32)
    scores = np.column_stack(
        [
            1.2 * X[:, 0] - 0.4 * X[:, 1],
            -0.7 * X[:, 0] + X[:, 2],
            0.5 * X[:, 1] - X[:, 3],
            -0.4 * X[:, 2] + 0.8 * X[:, 3],
        ]
    )
    return X, np.argmax(scores, axis=1).astype(np.int32)


@pytest.mark.parametrize(
    "mode_kwargs",
    [
        {},
        {"tree_growth": "leaf", "max_leaves": 4},
        {
            "boosting_mode": "goss",
            "goss_top_rate": 0.2,
            "goss_other_rate": 0.2,
        },
        {
            "boosting_mode": "dart",
            "dart_drop_rate": 0.25,
            "dart_max_drop": 3,
        },
        {
            "training_mode": "morph",
            "morph_warmup_iters": 1,
        },
    ],
    ids=["level", "leaf", "goss", "dart", "morph"],
)
def test_eligible_multiclass_modes_are_exact_across_thread_counts(mode_kwargs):
    X, y = _eligible_multiclass_fixture()
    shared = {
        "n_estimators": 4,
        "max_depth": 2,
        "seed": 13,
        **mode_kwargs,
    }

    _assert_fit_equivalent(
        GBMClassifier(n_jobs=1, **shared),
        GBMClassifier(n_jobs=2, **shared),
        X,
        y,
    )


def test_eligible_multiclass_validation_is_exact_across_thread_counts():
    X, y = _eligible_multiclass_fixture()
    validation_X = X[768:]
    validation_y = y[768:]
    shared = {
        "n_estimators": 8,
        "max_depth": 2,
        "seed": 17,
        "early_stopping_rounds": 2,
    }

    _assert_fit_equivalent(
        GBMClassifier(n_jobs=1, **shared),
        GBMClassifier(n_jobs=2, **shared),
        X,
        y,
        eval_set=(validation_X, validation_y),
    )


def test_eligible_multiclass_warm_start_is_exact_across_thread_counts():
    X, y = _eligible_multiclass_fixture()
    prior_single = GBMClassifier(
        n_estimators=2,
        max_depth=2,
        seed=23,
        n_jobs=1,
    ).fit(X, y)
    prior_parallel = GBMClassifier(
        n_estimators=2,
        max_depth=2,
        seed=23,
        n_jobs=2,
    ).fit(X, y)
    np.testing.assert_array_equal(
        prior_single.artifact_bytes,
        prior_parallel.artifact_bytes,
    )

    _assert_fit_equivalent(
        GBMClassifier(
            n_estimators=4,
            max_depth=2,
            seed=23,
            warm_start=True,
            n_jobs=1,
        ),
        GBMClassifier(
            n_estimators=4,
            max_depth=2,
            seed=23,
            warm_start=True,
            n_jobs=2,
        ),
        X,
        y,
        init_model=prior_single,
    )


@pytest.mark.parametrize("mode", ["independent", "joint"])
def test_multi_label_fit_is_exact_across_thread_counts(mode):
    X = np.arange(96, dtype=np.float32).reshape(32, 3)
    group = np.repeat(np.arange(8), 4)
    targets = np.column_stack(
        [
            np.arange(32) % 4,
            (np.arange(32) * 3) % 5,
        ]
    ).astype(np.float32)

    _assert_fit_equivalent(
        MultiLabelGBMRanker(
            n_estimators=3,
            max_depth=2,
            seed=5,
            n_jobs=1,
            multi_label_mode=mode,
        ),
        MultiLabelGBMRanker(
            n_estimators=3,
            max_depth=2,
            seed=5,
            n_jobs=2,
            multi_label_mode=mode,
        ),
        X,
        targets,
        group=group,
    )
