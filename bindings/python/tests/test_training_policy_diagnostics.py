"""Public fitted diagnostics for resolved training policy."""

from __future__ import annotations

import json
import sys
import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor
from alloygbm._regressor._validation import (
    _resolved_training_policy_from_metadata,
)


POLICY_KEYS = {
    "requested_mode",
    "requested_rounds",
    "effective_round_cap",
    "min_rows_per_leaf",
    "min_split_gain",
    "row_subsample",
    "col_subsample",
    "auto_split_l2_applied",
    "effective_split_l2",
    "split_noise_strength",
}


def valid_policy_metadata() -> dict[str, object]:
    return {
        "requested_mode": "auto",
        "requested_rounds": 40,
        "effective_round_cap": 32,
        "min_rows_per_leaf": 12,
        "min_split_gain": 0.0,
        "row_subsample": 0.8,
        "col_subsample": 0.5,
        "auto_split_l2_applied": False,
        "effective_split_l2": 2.0,
        "split_noise_strength": 1.0,
    }


def make_dense_fixture(
    *, rows: int, features: int, classes: int | None
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(73)
    X = rng.standard_normal((rows, features)).astype("float32")
    if classes is None:
        y = (1.5 * X[:, 0] - 0.25 * X[:, 1]).astype("float32")
    else:
        y = (np.arange(rows) % classes).astype("int32")
    return X, y


def test_auto_regressor_exposes_resolved_policy() -> None:
    X, y = make_dense_fixture(rows=2_048, features=32, classes=None)
    model = GBMRegressor(n_estimators=5, training_policy="auto", seed=7).fit(X, y)

    policy = model.resolved_training_policy_

    assert policy is not None
    assert set(policy) == POLICY_KEYS
    assert policy["requested_mode"] == "auto"
    assert policy["requested_rounds"] == 5
    assert policy["effective_round_cap"] == 5
    assert policy["row_subsample"] == pytest.approx(0.9)
    assert policy["col_subsample"] == pytest.approx(0.8)


def test_manual_regressor_preserves_explicit_policy_values() -> None:
    X, y = make_dense_fixture(rows=128, features=4, classes=None)
    model = GBMRegressor(
        n_estimators=4,
        training_policy="manual",
        min_data_in_leaf=7,
        min_split_gain=0.35,
        row_subsample=0.65,
        col_subsample=0.75,
        lambda_l2=1.25,
        seed=7,
    ).fit(X, y)

    policy = model.resolved_training_policy_

    assert policy is not None
    assert policy["requested_mode"] == "manual"
    assert policy["requested_rounds"] == 4
    assert policy["effective_round_cap"] == 4
    assert policy["min_rows_per_leaf"] == 7
    assert policy["min_split_gain"] == pytest.approx(0.35)
    assert policy["row_subsample"] == pytest.approx(0.65)
    assert policy["col_subsample"] == pytest.approx(0.75)
    assert policy["auto_split_l2_applied"] is False
    assert policy["effective_split_l2"] == pytest.approx(1.25)


def test_binary_classifier_assigns_resolved_policy() -> None:
    X, y = make_dense_fixture(rows=160, features=5, classes=2)
    model = GBMClassifier(n_estimators=3, training_policy="manual", seed=7).fit(X, y)

    policy = model.resolved_training_policy_

    assert policy is not None
    assert set(policy) == POLICY_KEYS
    assert policy["requested_mode"] == "manual"
    assert policy["requested_rounds"] == 3


def test_multiclass_auto_policy_reports_classifier_l2() -> None:
    # Auto mode gives multiclass softmax leaf L2 = 3 (objective-aware policy);
    # the regression-only small-wide L2 rule does not apply to classifiers.
    X, _ = make_dense_fixture(rows=128, features=32, classes=6)
    y = np.asarray(([0, 5] * 62) + [1, 2, 3, 4], dtype="int32")
    model = GBMClassifier(n_estimators=3, training_policy="auto", seed=7).fit(X, y)

    policy = model.resolved_training_policy_

    assert policy is not None
    assert policy["requested_mode"] == "auto"
    assert policy["auto_split_l2_applied"] is True
    assert policy["effective_split_l2"] == pytest.approx(3.0)

    manual = GBMClassifier(n_estimators=3, training_policy="manual", seed=7).fit(X, y)
    assert manual.resolved_training_policy_["auto_split_l2_applied"] is False
    assert manual.resolved_training_policy_["effective_split_l2"] == pytest.approx(0.0)


def test_policy_reports_split_noise_strength() -> None:
    X, y = make_dense_fixture(rows=160, features=5, classes=3)

    auto = GBMClassifier(n_estimators=3, training_policy="auto", seed=7).fit(X, y)
    manual = GBMClassifier(n_estimators=3, training_policy="manual", seed=7).fit(X, y)
    binary = GBMClassifier(n_estimators=3, training_policy="auto", seed=7).fit(
        X, (y > 0).astype("int32")
    )

    assert auto.resolved_training_policy_["split_noise_strength"] == pytest.approx(1.0)
    assert manual.resolved_training_policy_["split_noise_strength"] == 0.0
    assert binary.resolved_training_policy_["split_noise_strength"] == 0.0


@pytest.mark.parametrize("policy", ["auto", "manual"])
def test_random_strength_overrides_the_policy(policy: str) -> None:
    X, y = make_dense_fixture(rows=160, features=5, classes=3)

    off = GBMClassifier(
        n_estimators=3, training_policy=policy, random_strength=0.0, seed=7
    ).fit(X, y)
    on = GBMClassifier(
        n_estimators=3, training_policy=policy, random_strength=0.5, seed=7
    ).fit(X, y)
    regressor = GBMRegressor(
        n_estimators=3, training_policy=policy, random_strength=2.0, seed=7
    ).fit(X, y.astype("float32"))

    assert off.resolved_training_policy_["split_noise_strength"] == 0.0
    assert on.resolved_training_policy_["split_noise_strength"] == pytest.approx(0.5)
    assert regressor.resolved_training_policy_["split_noise_strength"] == pytest.approx(
        2.0
    )


def test_random_strength_changes_splits_deterministically() -> None:
    X, y = make_dense_fixture(rows=400, features=8, classes=None)
    base = GBMRegressor(n_estimators=10, training_policy="manual", seed=7)
    noisy = GBMRegressor(
        n_estimators=10, training_policy="manual", random_strength=1.0, seed=7
    )

    plain = base.fit(X, y).predict(X)
    first = noisy.fit(X, y).predict(X)
    second = noisy.fit(X, y).predict(X)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(plain, first)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "high"])
def test_random_strength_rejects_invalid_values(value: object) -> None:
    X, y = make_dense_fixture(rows=64, features=3, classes=3)
    model = GBMClassifier(n_estimators=2).set_params(random_strength=value)

    with pytest.raises(ValueError, match="random_strength"):
        model.fit(X, y)


def test_random_strength_is_a_public_param() -> None:
    model = GBMClassifier(random_strength=0.25)

    assert model.get_params()["random_strength"] == 0.25
    assert "random_strength=0.25" in repr(model)
    assert model.set_params(random_strength=None).random_strength is None


def test_auto_ranker_has_no_implicit_split_gain_floor() -> None:
    X, y = make_dense_fixture(rows=120, features=6, classes=None)
    group = np.repeat(np.arange(20), 6)
    model = GBMRanker(n_estimators=3, training_policy="auto", seed=7).fit(X, y, group=group)

    policy = model.resolved_training_policy_

    assert policy is not None
    assert policy["requested_mode"] == "auto"
    assert policy["min_split_gain"] == pytest.approx(0.0)


def test_resolved_policy_is_not_a_constructor_parameter() -> None:
    model = GBMRegressor()

    assert "resolved_training_policy_" not in model.get_params()


def test_reset_fitted_state_clears_resolved_policy() -> None:
    model = GBMRegressor()
    model.resolved_training_policy_ = {"requested_mode": "auto"}

    model._reset_fitted_state()

    assert not hasattr(model, "resolved_training_policy_")


def test_save_load_model_round_trips_resolved_policy(tmp_path) -> None:
    X, y = make_dense_fixture(rows=128, features=4, classes=None)
    model = GBMRegressor(
        n_estimators=4,
        training_policy="manual",
        min_data_in_leaf=7,
        min_split_gain=0.35,
        row_subsample=0.65,
        col_subsample=0.75,
        lambda_l2=1.25,
        seed=7,
    ).fit(X, y)
    path = tmp_path / "policy.agbm"

    model.save_model(str(path))

    payload = path.read_bytes()
    metadata_len = int.from_bytes(payload[4:8], "little")
    metadata = json.loads(payload[8 : 8 + metadata_len])
    restored = GBMRegressor.load_model(str(path))

    assert metadata["resolved_training_policy"] == model.resolved_training_policy_
    assert restored.resolved_training_policy_ == model.resolved_training_policy_


@pytest.mark.parametrize(
    ("estimator", "fit_kwargs"),
    [
        (GBMClassifier(n_estimators=3, training_policy="manual", seed=7), {}),
        (
            GBMRanker(n_estimators=3, training_policy="auto", seed=7),
            {"group": np.repeat(np.arange(20), 6)},
        ),
    ],
)
def test_classifier_and_ranker_save_load_round_trip_resolved_policy(
    estimator, fit_kwargs: dict[str, object], tmp_path
) -> None:
    if isinstance(estimator, GBMClassifier):
        X, y = make_dense_fixture(rows=120, features=6, classes=2)
    else:
        X, y = make_dense_fixture(rows=120, features=6, classes=None)
    estimator.fit(X, y, **fit_kwargs)
    path = tmp_path / f"{type(estimator).__name__}.agbm"

    estimator.save_model(str(path))
    restored = type(estimator).load_model(str(path))

    assert restored.resolved_training_policy_ == estimator.resolved_training_policy_


def test_load_model_without_policy_metadata_returns_none(tmp_path) -> None:
    X, y = make_dense_fixture(rows=128, features=4, classes=None)
    model = GBMRegressor(n_estimators=4, training_policy="auto", seed=7).fit(X, y)
    path = tmp_path / "policy.agbm"

    model.save_model(str(path))

    payload = path.read_bytes()
    metadata_len = int.from_bytes(payload[4:8], "little")
    metadata = json.loads(payload[8 : 8 + metadata_len])
    artifact = payload[8 + metadata_len :]
    metadata.pop("resolved_training_policy", None)
    legacy_metadata = json.dumps(metadata).encode("utf-8")
    path.write_bytes(
        b"AGBP"
        + len(legacy_metadata).to_bytes(4, "little")
        + legacy_metadata
        + artifact
    )

    restored = GBMRegressor.load_model(str(path))

    assert restored.resolved_training_policy_ is None


def test_policy_metadata_normalizer_returns_a_fresh_normalized_dictionary() -> None:
    metadata = valid_policy_metadata()

    normalized = _resolved_training_policy_from_metadata(metadata)

    assert normalized == metadata
    assert normalized is not metadata


@pytest.mark.parametrize(
    "value",
    [[], {}, 1, True, None],
)
def test_policy_metadata_normalizer_rejects_non_string_mode(value: object) -> None:
    metadata = valid_policy_metadata()
    metadata["requested_mode"] = value

    assert _resolved_training_policy_from_metadata(metadata) is None


@pytest.mark.parametrize(
    "field",
    ["min_split_gain", "row_subsample", "col_subsample", "effective_split_l2"],
)
@pytest.mark.parametrize("value", [True, "0.5"])
def test_policy_metadata_normalizer_rejects_non_numeric_float_categories(
    field: str, value: object
) -> None:
    metadata = valid_policy_metadata()
    metadata[field] = value

    assert _resolved_training_policy_from_metadata(metadata) is None


@pytest.mark.parametrize(
    "field",
    ["requested_rounds", "effective_round_cap", "min_rows_per_leaf"],
)
@pytest.mark.parametrize("value", [40.0, True, "40"])
def test_policy_metadata_normalizer_rejects_non_integer_categories(
    field: str, value: object
) -> None:
    metadata = valid_policy_metadata()
    metadata[field] = value

    assert _resolved_training_policy_from_metadata(metadata) is None


@pytest.mark.parametrize("value", [0, 1, "false", [], {}])
def test_policy_metadata_normalizer_rejects_non_boolean_provenance(
    value: object,
) -> None:
    metadata = valid_policy_metadata()
    metadata["auto_split_l2_applied"] = value

    assert _resolved_training_policy_from_metadata(metadata) is None


def test_policy_metadata_without_split_noise_strength_still_loads() -> None:
    metadata = valid_policy_metadata()
    metadata.pop("split_noise_strength")

    policy = _resolved_training_policy_from_metadata(metadata)

    assert policy is not None
    assert policy["split_noise_strength"] is None


@pytest.mark.parametrize("value", [-1.0, float("nan"), "1.0", True])
def test_policy_metadata_rejects_invalid_split_noise_strength(value: object) -> None:
    metadata = valid_policy_metadata()
    metadata["split_noise_strength"] = value

    assert _resolved_training_policy_from_metadata(metadata) is None


def test_policy_metadata_normalizer_requires_exact_stable_keys() -> None:
    missing_key = valid_policy_metadata()
    missing_key.pop("effective_split_l2")
    extra_key = valid_policy_metadata()
    extra_key["unknown_future_field"] = "unexpected"

    assert _resolved_training_policy_from_metadata(missing_key) is None
    assert _resolved_training_policy_from_metadata(extra_key) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field",
    ["min_split_gain", "row_subsample", "col_subsample", "effective_split_l2"],
)
def test_policy_metadata_normalizer_rejects_non_finite_float_values(
    field: str, value: float
) -> None:
    metadata = valid_policy_metadata()
    metadata[field] = value

    assert _resolved_training_policy_from_metadata(metadata) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_rounds", 0),
        ("effective_round_cap", 0),
        ("min_rows_per_leaf", 0),
        ("min_split_gain", -0.1),
        ("row_subsample", 0.0),
        ("row_subsample", 1.1),
        ("col_subsample", 0.0),
        ("col_subsample", 1.1),
        ("effective_split_l2", -0.1),
    ],
)
def test_policy_metadata_normalizer_rejects_invalid_ranges(
    field: str, value: object
) -> None:
    metadata = valid_policy_metadata()
    metadata[field] = value

    assert _resolved_training_policy_from_metadata(metadata) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_rounds", 4_097),
        ("effective_round_cap", 4_097),
        ("requested_rounds", sys.maxsize + 1),
        ("effective_round_cap", sys.maxsize + 1),
        ("min_rows_per_leaf", 1 << 32),
        ("min_rows_per_leaf", sys.maxsize + 1),
    ],
)
def test_policy_metadata_normalizer_rejects_unsupported_integer_bounds(
    field: str, value: int
) -> None:
    metadata = valid_policy_metadata()
    metadata[field] = value

    assert _resolved_training_policy_from_metadata(metadata) is None


def test_policy_metadata_normalizer_rejects_round_cap_above_request() -> None:
    metadata = valid_policy_metadata()
    metadata["requested_rounds"] = 31
    metadata["effective_round_cap"] = 32

    assert _resolved_training_policy_from_metadata(metadata) is None


def test_policy_metadata_normalizer_allows_large_supported_min_rows() -> None:
    metadata = valid_policy_metadata()
    metadata["min_rows_per_leaf"] = (1 << 32) - 1

    normalized = _resolved_training_policy_from_metadata(metadata)

    assert normalized is not None
    assert normalized["min_rows_per_leaf"] == (1 << 32) - 1


@pytest.mark.parametrize("value", [None, [], (), 0, "policy"])
def test_policy_metadata_normalizer_is_total_over_arbitrary_metadata(value: object) -> None:
    assert _resolved_training_policy_from_metadata(value) is None


@pytest.mark.parametrize(
    "field",
    ["min_split_gain", "row_subsample", "col_subsample", "effective_split_l2"],
)
def test_policy_metadata_normalizer_and_wrapper_load_reject_huge_float_ints(
    field: str,
    tmp_path,
) -> None:
    metadata_policy = valid_policy_metadata()
    metadata_policy[field] = int("9" * 4_000)

    assert _resolved_training_policy_from_metadata(metadata_policy) is None

    X, y = make_dense_fixture(rows=128, features=4, classes=None)
    model = GBMRegressor(n_estimators=4, training_policy="auto", seed=7).fit(X, y)
    path = tmp_path / f"huge-{field}.agbm"
    model.save_model(str(path))

    payload = path.read_bytes()
    metadata_len = int.from_bytes(payload[4:8], "little")
    metadata = json.loads(payload[8 : 8 + metadata_len])
    artifact = payload[8 + metadata_len :]
    metadata["resolved_training_policy"] = metadata_policy
    serialized_metadata = json.dumps(metadata).encode("utf-8")
    path.write_bytes(
        b"AGBP"
        + len(serialized_metadata).to_bytes(4, "little")
        + serialized_metadata
        + artifact
    )

    restored = GBMRegressor.load_model(str(path))

    assert restored.resolved_training_policy_ is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_rounds", 4_097),
        ("effective_round_cap", 4_097),
        ("min_rows_per_leaf", 1 << 32),
        ("min_rows_per_leaf", sys.maxsize + 1),
    ],
)
def test_wrapper_load_rejects_unsupported_policy_integer_bounds(
    field: str, value: int, tmp_path
) -> None:
    X, y = make_dense_fixture(rows=128, features=4, classes=None)
    model = GBMRegressor(n_estimators=4, training_policy="auto", seed=7).fit(X, y)
    path = tmp_path / f"invalid-{field}.agbm"
    model.save_model(str(path))

    payload = path.read_bytes()
    metadata_len = int.from_bytes(payload[4:8], "little")
    metadata = json.loads(payload[8 : 8 + metadata_len])
    artifact = payload[8 + metadata_len :]
    metadata["resolved_training_policy"][field] = value
    serialized_metadata = json.dumps(metadata).encode("utf-8")
    path.write_bytes(
        b"AGBP"
        + len(serialized_metadata).to_bytes(4, "little")
        + serialized_metadata
        + artifact
    )

    restored = GBMRegressor.load_model(str(path))

    assert restored.resolved_training_policy_ is None


def test_wrapper_load_accepts_native_min_rows_upper_bound(tmp_path) -> None:
    X, y = make_dense_fixture(rows=128, features=4, classes=None)
    model = GBMRegressor(n_estimators=4, training_policy="auto", seed=7).fit(X, y)
    path = tmp_path / "max-supported-min-rows.agbm"
    model.save_model(str(path))

    payload = path.read_bytes()
    metadata_len = int.from_bytes(payload[4:8], "little")
    metadata = json.loads(payload[8 : 8 + metadata_len])
    artifact = payload[8 + metadata_len :]
    metadata["resolved_training_policy"]["min_rows_per_leaf"] = (1 << 32) - 1
    serialized_metadata = json.dumps(metadata).encode("utf-8")
    path.write_bytes(
        b"AGBP"
        + len(serialized_metadata).to_bytes(4, "little")
        + serialized_metadata
        + artifact
    )

    restored = GBMRegressor.load_model(str(path))

    assert restored.resolved_training_policy_ is not None
    assert restored.resolved_training_policy_["min_rows_per_leaf"] == (1 << 32) - 1


def test_wrapper_load_rejects_round_cap_above_request(tmp_path) -> None:
    X, y = make_dense_fixture(rows=128, features=4, classes=None)
    model = GBMRegressor(n_estimators=4, training_policy="auto", seed=7).fit(X, y)
    path = tmp_path / "invalid-round-relationship.agbm"
    model.save_model(str(path))

    payload = path.read_bytes()
    metadata_len = int.from_bytes(payload[4:8], "little")
    metadata = json.loads(payload[8 : 8 + metadata_len])
    artifact = payload[8 + metadata_len :]
    metadata["resolved_training_policy"]["requested_rounds"] = 31
    metadata["resolved_training_policy"]["effective_round_cap"] = 32
    serialized_metadata = json.dumps(metadata).encode("utf-8")
    path.write_bytes(
        b"AGBP"
        + len(serialized_metadata).to_bytes(4, "little")
        + serialized_metadata
        + artifact
    )

    restored = GBMRegressor.load_model(str(path))

    assert restored.resolved_training_policy_ is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"training_mode": "morph"},
        {"leaf_model": "linear", "pl_split_candidates": 4},
    ],
)
def test_split_noise_reports_zero_where_the_split_search_ignores_it(
    kwargs: dict[str, object],
) -> None:
    X, y = make_dense_fixture(rows=200, features=5, classes=None)
    model = GBMRegressor(
        n_estimators=3, training_policy="manual", random_strength=1.0, seed=7, **kwargs
    ).fit(X, y)

    assert model.resolved_training_policy_["split_noise_strength"] == 0.0


@pytest.mark.parametrize("estimator_type", [GBMRegressor, GBMClassifier, GBMRanker])
def test_legacy_pickle_state_defaults_random_strength(estimator_type) -> None:
    state = estimator_type(random_strength=0.5).__getstate__()
    state.pop("random_strength")

    restored = estimator_type.__new__(estimator_type)
    restored.__setstate__(state)

    assert restored.random_strength is None
    assert restored.get_params()["random_strength"] is None
    assert "random_strength=None" in repr(restored)
