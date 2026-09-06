from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from alloygbm import GBMClassifier, GBMRegressor, MultiLabelGBMRanker


def _continuous_fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[row + 0.25, (row % 5) + 0.125] for row in range(24)],
        dtype=np.float32,
    )
    y = np.asarray([0.2 * row + (row % 3) for row in range(24)], dtype=np.float32)
    return x, y


def test_quantile_sketch_parameter_round_trips_and_validates() -> None:
    model = GBMRegressor(quantile_sketch_max_rows=17)

    assert model.get_params()["quantile_sketch_max_rows"] == 17
    assert "quantile_sketch_max_rows=17" in repr(model)
    assert "quantile_sketch_max_rows" in inspect.signature(
        GBMClassifier.__init__
    ).parameters
    assert model.set_params(quantile_sketch_max_rows=None) is model
    assert model.quantile_sketch_max_rows is None

    x, y = _continuous_fixture()
    with pytest.raises(ValueError, match="quantile_sketch_max_rows"):
        GBMRegressor(quantile_sketch_max_rows=0).fit(x, y)
    model.set_params(quantile_sketch_max_rows=-1)
    with pytest.raises(ValueError, match="quantile_sketch_max_rows"):
        model.fit(x, y)
    with pytest.raises(ValueError, match="quantile_sketch_max_rows"):
        MultiLabelGBMRanker(quantile_sketch_max_rows=0)


def test_exact_quantile_fit_reports_methods_and_persists(tmp_path) -> None:
    x, y = _continuous_fixture()
    model = GBMRegressor(n_estimators=2, max_depth=2).fit(x, y)

    assert model.feature_quantile_cut_methods_ == ["exact", "exact"]

    path = tmp_path / "exact-quantile.agbm"
    model.save_model(path)
    restored = GBMRegressor.load_model(path)
    assert restored.quantile_sketch_max_rows is None
    assert restored.feature_quantile_cut_methods_ == ["exact", "exact"]
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


def test_classifier_and_independent_multilabel_expose_exact_methods() -> None:
    x, y = _continuous_fixture()
    classifier = GBMClassifier(n_estimators=2, max_depth=2).fit(
        x, (y > np.median(y)).astype(np.int32)
    )
    assert classifier.feature_quantile_cut_methods_ == ["exact", "exact"]

    multilabel = MultiLabelGBMRanker(
        ranking_objective="queryrmse",
        n_estimators=2,
        max_depth=2,
    ).fit(
        x,
        np.column_stack((y, y[::-1])),
        group=np.repeat(np.arange(6), 4),
    )
    assert multilabel.feature_quantile_cut_methods_ == ["exact", "exact"]


def test_sketch_activation_is_deterministic_and_persists(tmp_path) -> None:
    x, y = _continuous_fixture()
    params = {
        "n_estimators": 2,
        "max_depth": 2,
        "quantile_sketch_max_rows": 8,
    }
    first = GBMRegressor(**params).fit(x, y)
    second = GBMRegressor(**params).fit(x, y)

    assert first.feature_quantile_cut_methods_ == ["sketch", "sketch"]
    assert first._continuous_feature_quantile_cuts == second._continuous_feature_quantile_cuts
    assert all(len(cuts) <= 7 for cuts in first._continuous_feature_quantile_cuts)

    path = tmp_path / "sketched-quantile.agbm"
    first.save_model(path)
    restored = GBMRegressor.load_model(path)
    assert restored.quantile_sketch_max_rows == 8
    assert restored.feature_quantile_cut_methods_ == ["sketch", "sketch"]
    assert (
        restored._continuous_feature_quantile_cuts
        == first._continuous_feature_quantile_cuts
    )
    np.testing.assert_array_equal(restored.predict(x), first.predict(x))


def test_sketch_limit_at_row_count_keeps_exact_cuts() -> None:
    x, y = _continuous_fixture()
    exact = GBMRegressor(n_estimators=2).fit(x, y)
    bounded = GBMRegressor(
        n_estimators=2, quantile_sketch_max_rows=len(x)
    ).fit(x, y)

    assert bounded.feature_quantile_cut_methods_ == ["exact", "exact"]
    assert bounded._continuous_feature_quantile_cuts == exact._continuous_feature_quantile_cuts


def test_changing_sketch_limit_after_fit_requires_refit() -> None:
    x, y = _continuous_fixture()
    model = GBMRegressor(n_estimators=2).fit(x, y)

    model.set_params(quantile_sketch_max_rows=8)

    with pytest.raises(NotFittedError):
        model.predict(x)


def test_python_quantile_cuts_reserve_the_missing_bin_slot() -> None:
    """The Python mirror must budget bins exactly like the Rust path.

    255 data bins are delimited by 254 cuts. Emitting 255 would address 256
    intervals, and quantization then clamps the top one into its neighbour --
    merging the two highest quantiles.
    """
    from alloygbm import GBMRegressor

    values = [float(v) for v in range(1, 100_001)]
    cuts = GBMRegressor._single_feature_quantile_cuts_from_sorted_values(values, 255)
    assert len(cuts) == 254

    flat = [float(v) for v in range(1, 20_001)]
    derived = GBMRegressor._derive_dense_feature_quantile_cuts(flat, 20_000, 1, 256)
    assert len(derived) == 1
    assert len(derived[0]) <= 254, (
        f"{len(derived[0])} cuts for max_bins=256 leaves no free slot for the "
        "missing-value bin"
    )


def test_quantile_binning_top_bin_is_not_double_width() -> None:
    """End-to-end signature of the bin-budget defect.

    With the budget off by one the highest data bin absorbs two quantile
    intervals, so it holds roughly twice as many rows as its neighbours. This
    reads the borders the model actually stored rather than recomputing them,
    so it fails if either the Rust or the Python path regresses.
    """
    import numpy as np
    from alloygbm import GBMRegressor

    rng = np.random.default_rng(20260906)
    rows = 60_000
    X = rng.normal(size=(rows, 3)).astype(np.float32)
    y = X[:, 0].astype(np.float32)

    model = GBMRegressor(n_estimators=5, max_depth=4, seed=1, n_jobs=1).fit(X, y)
    cuts = model._continuous_feature_quantile_cuts
    assert cuts is not None, "quantile binning must record its borders"

    for feature_index, feature_cuts in enumerate(cuts):
        assert len(feature_cuts) <= 254, (
            f"feature {feature_index} stored {len(feature_cuts)} cuts; only 254 "
            "fit once the missing-value slot is reserved"
        )
        counts = np.bincount(
            np.minimum(
                np.searchsorted(
                    np.asarray(feature_cuts), X[:, feature_index], side="right"
                ),
                254,
            ),
            minlength=255,
        )
        assert counts[254] <= counts[253] * 2, (
            f"feature {feature_index}: top bin holds {counts[254]} rows against "
            f"{counts[253]} in its neighbour, so two quantile intervals merged"
        )


def test_weighted_quantile_binning_reserves_the_missing_bin_slot() -> None:
    """The weighted cut path is a separate branch and needs its own guard."""
    import numpy as np
    from alloygbm import GBMRegressor

    rng = np.random.default_rng(20260906)
    rows = 40_000
    X = rng.normal(size=(rows, 3)).astype(np.float32)
    y = X[:, 0].astype(np.float32)
    weights = rng.integers(1, 4, size=rows).astype(np.float64)

    model = GBMRegressor(n_estimators=5, max_depth=4, seed=1, n_jobs=1).fit(
        X, y, sample_weight=weights
    )
    cuts = model._continuous_feature_quantile_cuts
    assert cuts is not None
    for feature_index, feature_cuts in enumerate(cuts):
        assert len(feature_cuts) <= 254, (
            f"weighted path stored {len(feature_cuts)} cuts for feature "
            f"{feature_index}; only 254 fit once the missing slot is reserved"
        )


@pytest.mark.parametrize("max_bins", [3, 4, 256, 257, 512])
def test_row_and_dense_quantile_cuts_parity(max_bins: int) -> None:
    """Row fallback and dense helper must derive identical cuts across bin budgets."""
    import numpy as np
    from alloygbm import GBMRegressor

    rng = np.random.default_rng(20260906)
    rows_count = 4096
    cols_count = 3
    # Use nonintegral continuous values so pre-binned integer detection is bypassed
    raw_data = rng.normal(size=(rows_count, cols_count)).astype(np.float32) + 0.25
    # Introduce repeated values in column 0
    raw_data[:, 0] = np.round(raw_data[:, 0] * 5.0) / 5.0 + 0.25
    # Introduce NaNs in column 1
    nan_mask = rng.uniform(size=rows_count) < 0.1
    raw_data[nan_mask, 1] = np.nan

    flat_values = raw_data.flatten().tolist()
    dense_cuts = GBMRegressor._derive_dense_feature_quantile_cuts(
        flat_values, rows_count, cols_count, max_bins
    )

    rows = [[float(v) for v in row] for row in raw_data]
    row_cuts = GBMRegressor._derive_continuous_feature_quantile_cuts(
        rows, max_bins
    )

    assert len(row_cuts) == len(dense_cuts) == cols_count
    data_bin_count = max_bins - 1
    for fi in range(cols_count):
        assert row_cuts[fi] == dense_cuts[fi], (
            f"max_bins={max_bins}, feature={fi}: row cuts ({len(row_cuts[fi])}) != "
            f"dense cuts ({len(dense_cuts[fi])})"
        )
        assert len(row_cuts[fi]) <= max(data_bin_count - 1, 0)


def test_legacy_row_bridge_quantization_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit-level test forcing legacy row bridge to verify row quantile fallback."""
    import numpy as np
    from alloygbm import GBMRegressor
    from alloygbm._regressor import _base

    def _fail_load():
        raise RuntimeError("simulated missing native summary bridge")

    monkeypatch.setattr(_base, "_load_native_train_regression_artifact_dense_with_summary", _fail_load)
    monkeypatch.setattr(_base, "_load_native_train_regression_artifact_with_summary", _fail_load)

    rng = np.random.default_rng(20260906)
    rows_count = 2000
    X = (rng.normal(size=(rows_count, 3)).astype(np.float32) + 0.25).tolist()
    y = [float(r[0] * 2.0) for r in X]

    model = GBMRegressor(
        n_estimators=3,
        max_depth=3,
        continuous_binning_strategy="quantile",
        continuous_binning_max_bins=256,
        seed=42,
    ).fit(X, y)

    cuts = model._continuous_feature_quantile_cuts
    assert cuts is not None
    assert len(cuts) == 3
    for fi, fcuts in enumerate(cuts):
        assert len(fcuts) <= 254, f"legacy row bridge stored {len(fcuts)} cuts for feature {fi}"

    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert np.all(np.isfinite(preds))


def test_zero_addition_continuation_preserves_pre_fix_cut_space() -> None:
    """Continuation with zero additions must preserve prior model's quantile cut space and predictions."""
    import copy
    import numpy as np
    from alloygbm import GBMRegressor

    rng = np.random.default_rng(42)
    rows = 4096
    cols = 3
    X = rng.normal(size=(rows, cols)).astype(np.float32)
    y = (2.0 * X[:, 0] + X[:, 1] * X[:, 2]).astype(np.float32)

    # Train a base model
    base_model = GBMRegressor(
        n_estimators=12,
        max_depth=4,
        continuous_binning_strategy="quantile",
        continuous_binning_max_bins=256,
        seed=42,
    ).fit(X, y)

    # To simulate a pre-fix saved model that had 255 cuts:
    # derive 255 cuts (from data_bin_count=256 intervals, the pre-fix behavior)
    legacy_255_cuts = [
        GBMRegressor._single_feature_quantile_cuts_from_sorted_values(
            sorted(X[:, fi].tolist()), 256
        )
        for fi in range(cols)
    ]
    for fcuts in legacy_255_cuts:
        assert len(fcuts) == 255, f"expected 255 cuts to simulate pre-fix model, got {len(fcuts)}"

    # Attach the 255 cuts to base_model as if loaded from pre-fix wheel
    base_model._continuous_feature_quantile_cuts = copy.deepcopy(legacy_255_cuts)
    base_model._native_predictor_handle = base_model._build_native_predictor_handle(
        base_model.artifact_bytes, required=True
    )
    base_model._convert_predictor_thresholds_to_float()
    base_preds = base_model.predict(X)

    # 1. Continue via init_model with min_split_gain=1e30 so 0 new trees are accepted
    continued_init = GBMRegressor(
        n_estimators=1,
        training_policy="manual",
        min_split_gain=1e30,
        continuous_binning_strategy="quantile",
        continuous_binning_max_bins=256,
        seed=42,
    ).fit(X, y, init_model=base_model)

    assert bytes(continued_init.artifact_bytes) == bytes(base_model.artifact_bytes)
    assert continued_init._continuous_feature_quantile_cuts == legacy_255_cuts
    init_preds = continued_init.predict(X)
    assert np.array_equal(init_preds, base_preds), (
        f"init_model continuation altered predictions on {np.count_nonzero(init_preds != base_preds)} rows; "
        f"max delta {np.max(np.abs(init_preds - base_preds))}"
    )

    # 2. Continue via warm_start=True
    base_copy = copy.deepcopy(base_model)
    base_copy.warm_start = True
    base_copy.n_estimators = 13
    base_copy.training_policy = "manual"
    base_copy.min_split_gain = 1e30
    continued_warm = base_copy.fit(X, y)

    assert bytes(continued_warm.artifact_bytes) == bytes(base_model.artifact_bytes)
    assert continued_warm._continuous_feature_quantile_cuts == legacy_255_cuts
    warm_preds = continued_warm.predict(X)
    assert np.array_equal(warm_preds, base_preds), (
        f"warm_start continuation altered predictions on {np.count_nonzero(warm_preds != base_preds)} rows; "
        f"max delta {np.max(np.abs(warm_preds - base_preds))}"
    )


@pytest.mark.parametrize(
    ("max_bins", "sketch_rows"),
    [
        (256, None),  # exact U8 path
        (512, None),  # U16 wide path
        (256, 2000),  # active sketch path
    ],
)
def test_quantile_binning_stays_deterministic_across_thread_counts(
    max_bins: int, sketch_rows: int | None
) -> None:
    """The bin-budget fix must maintain exact equality on artifact bytes, cuts, and predictions."""
    import copy
    import hashlib
    import numpy as np
    from alloygbm import GBMRegressor

    rng = np.random.default_rng(20260906)
    rows = 60_000
    cols = 8
    X = rng.normal(size=(rows, cols)).astype(np.float32)
    scale = np.where(np.arange(rows) % 997 == 0, 1.0e6, 1.0).astype(np.float32)
    y = ((3.0 * X[:, 0] - 2.0 * X[:, 1]) * scale).astype(np.float32)

    # Construct probe containing normal values, extreme tails (+-1e9), and NaNs
    probe = rng.normal(size=(200, cols)).astype(np.float32)
    probe[0, :] = 1.0e9
    probe[1, :] = -1.0e9
    probe[2, 0] = np.nan
    probe[3, 1] = np.nan

    digests = {}
    cuts_by_job = {}
    preds_by_job = {}

    for n_jobs in (1, 2, 4):
        model = GBMRegressor(
            n_estimators=6,
            max_depth=5,
            seed=20260906,
            deterministic=True,
            continuous_binning_strategy="quantile",
            continuous_binning_max_bins=max_bins,
            quantile_sketch_max_rows=sketch_rows,
            n_jobs=n_jobs,
        ).fit(X, y)
        digests[n_jobs] = hashlib.sha256(bytes(model.artifact_bytes)).hexdigest()
        cuts_by_job[n_jobs] = copy.deepcopy(model._continuous_feature_quantile_cuts)
        preds_by_job[n_jobs] = model.predict(probe)

    ref_digest = digests[1]
    ref_cuts = cuts_by_job[1]
    ref_preds = preds_by_job[1]

    for n_jobs in (2, 4):
        assert digests[n_jobs] == ref_digest, (
            f"max_bins={max_bins}, sketch={sketch_rows}: artifact from n_jobs={n_jobs} "
            "differs from n_jobs=1"
        )
        assert cuts_by_job[n_jobs] == ref_cuts, (
            f"max_bins={max_bins}, sketch={sketch_rows}: quantile cuts from n_jobs={n_jobs} "
            "differ from n_jobs=1"
        )
        assert np.array_equal(preds_by_job[n_jobs], ref_preds, equal_nan=True), (
            f"max_bins={max_bins}, sketch={sketch_rows}: predictions from n_jobs={n_jobs} "
            "differ from n_jobs=1"
        )



