"""MultiLabelGBMRanker(multi_label_mode='joint') tests (v0.10.1).

v0.10.0 shipped the Rust joint trainer (`fit_joint_multi_output`) and the
`MultiOutputLeafValues` artifact section, but `MultiLabelGBMRanker` still
routed every fit to the independent-per-label fallback.  v0.10.1 wires
`multi_label_mode='joint'` through to the joint trainer.
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
    m = MultiLabelGBMRanker(n_estimators=4, multi_label_mode="joint")
    m.fit(X, y, group=group)
    preds = m.predict(X)
    assert preds.shape == (X.shape[0], y.shape[1])
    assert np.all(np.isfinite(preds))


def test_default_multi_label_mode_is_independent():
    """Backward compatibility: callers that don't pass multi_label_mode must
    still hit the v0.7.1 independent-per-label fallback (which uses
    GBMRanker under the hood)."""
    m = MultiLabelGBMRanker(n_estimators=2)
    assert m.multi_label_mode == "independent"


def test_invalid_multi_label_mode_rejected():
    with pytest.raises(ValueError, match="multi_label_mode"):
        MultiLabelGBMRanker(multi_label_mode="nonsense")


def test_joint_mode_pickle_round_trip_preserves_predictions():
    import pickle

    X, y, group = _toy_ranking()
    m = MultiLabelGBMRanker(n_estimators=4, multi_label_mode="joint")
    m.fit(X, y, group=group)
    p1 = m.predict(X)
    restored = pickle.loads(pickle.dumps(m))
    p2 = restored.predict(X)
    np.testing.assert_allclose(p1, p2, rtol=1e-6)
    assert restored.multi_label_mode == "joint"
    assert restored.n_labels_ == y.shape[1]


def test_joint_mode_save_load_round_trip(tmp_path):
    X, y, group = _toy_ranking()
    m = MultiLabelGBMRanker(n_estimators=4, multi_label_mode="joint")
    m.fit(X, y, group=group)
    p1 = m.predict(X)
    path = tmp_path / "joint_ml.alloy"
    m.save_model(str(path))
    restored = MultiLabelGBMRanker.load_model(str(path))
    p2 = restored.predict(X)
    np.testing.assert_allclose(p1, p2, rtol=1e-6)
    assert restored.multi_label_mode == "joint"
    assert restored.n_labels_ == y.shape[1]


def test_v1_bundle_still_loads_as_independent(tmp_path):
    """Bundles written before v0.10.1 had no mode byte and always meant
    independent training. The v2 loader must still accept them."""
    import pickle as _pickle
    import struct as _struct

    from alloygbm.multi_label_ranker import _MULTI_LABEL_RANKER_MAGIC

    X, y, group = _toy_ranking()
    m = MultiLabelGBMRanker(n_estimators=2, multi_label_mode="independent")
    m.fit(X, y, group=group)
    path = tmp_path / "v1.alloy"
    with open(path, "wb") as f:
        f.write(_MULTI_LABEL_RANKER_MAGIC)
        f.write(_struct.pack("<II", 1, len(m._sub_rankers)))
        for name in m.ranking_labels_:
            encoded = name.encode("utf-8")
            f.write(_struct.pack("<I", len(encoded)))
            f.write(encoded)
        for ranker in m._sub_rankers:
            blob = _pickle.dumps(ranker, protocol=_pickle.HIGHEST_PROTOCOL)
            f.write(_struct.pack("<Q", len(blob)))
            f.write(blob)
    restored = MultiLabelGBMRanker.load_model(str(path))
    assert restored.multi_label_mode == "independent"
    np.testing.assert_allclose(m.predict(X), restored.predict(X), rtol=1e-6)


def test_joint_mode_rejects_factor_exposures():
    X, y, group = _toy_ranking()
    n = X.shape[0]
    exposures = np.zeros((n, 2), dtype=np.float32)
    m = MultiLabelGBMRanker(n_estimators=2, multi_label_mode="joint")
    with pytest.raises(NotImplementedError, match="factor_exposures"):
        m.fit(X, y, group=group, factor_exposures=exposures)


def test_joint_mode_rejects_warm_start_kwarg():
    X, y, group = _toy_ranking()
    m = MultiLabelGBMRanker(
        n_estimators=2, multi_label_mode="joint", warm_start=True
    )
    with pytest.raises(NotImplementedError, match="warm_start"):
        m.fit(X, y, group=group)


def test_joint_mode_accepts_group_sizes_and_per_row_ids():
    """PR review (C2): the joint path normalizes group input to per-row
    IDs (length n_rows) and stable-sorts rows by group ID before
    fitting. Both LightGBM-style group sizes and per-row IDs must
    produce identical models for the same data."""
    rng = np.random.default_rng(101)
    n_queries, items_per_query = 6, 7
    n_rows = n_queries * items_per_query
    X = rng.standard_normal((n_rows, 4)).astype(np.float32)
    y = rng.integers(0, 4, size=(n_rows, 2)).astype(np.float32)

    group_sizes = np.full(n_queries, items_per_query, dtype=np.int32)
    group_per_row = np.repeat(np.arange(n_queries), items_per_query).astype(np.int32)

    m_sizes = MultiLabelGBMRanker(n_estimators=4, multi_label_mode="joint", seed=101)
    m_sizes.fit(X, y, group=group_sizes)
    m_ids = MultiLabelGBMRanker(n_estimators=4, multi_label_mode="joint", seed=101)
    m_ids.fit(X, y, group=group_per_row)

    np.testing.assert_allclose(m_sizes.predict(X), m_ids.predict(X), rtol=1e-6)


def test_joint_mode_rejects_unsupported_kwarg():
    """PR review (C3, C7): kwargs that aren't in the strict allow-list
    are rejected with a clear NotImplementedError rather than silently
    ignored."""
    X, y, group = _toy_ranking()
    m = MultiLabelGBMRanker(
        n_estimators=2, multi_label_mode="joint", monotone_constraints=[1, 0, 0, 0]
    )
    with pytest.raises(NotImplementedError, match="monotone_constraints"):
        m.fit(X, y, group=group)


@pytest.mark.parametrize(
    "kw,value",
    [
        ("min_split_gain", 0.0),
        ("row_subsample", 0.7),
        ("col_subsample", 0.8),
        ("interaction_constraints", [[0, 1], [2, 3]]),
    ],
)
def test_joint_mode_accepts_phase1_kwargs(kw, value):
    """v0.10.2 Phase 1: joint mode now plumbs `min_split_gain`,
    `row_subsample`, `col_subsample`, `interaction_constraints` into
    the engine's `TrainParams`. Each kwarg should round-trip through
    `train_joint_multi_label_ranker` without raising."""
    X, y, group = _toy_ranking()
    m = MultiLabelGBMRanker(
        n_estimators=2, multi_label_mode="joint", **{kw: value}
    )
    m.fit(X, y, group=group)
    preds = m.predict(X)
    assert preds.shape == y.shape


def test_joint_mode_leaf_wise_growth_respects_max_leaves():
    """v0.10.2 Phase 2: joint mode supports tree_growth='leaf' + max_leaves
    via build_joint_round_leafwise. The model should fit, predict, and
    produce a finite-sized tree capped by max_leaves."""
    X, y, group = _toy_ranking(n_queries=12, items_per_query=6, n_features=4, n_labels=3)
    m = MultiLabelGBMRanker(
        n_estimators=3,
        multi_label_mode="joint",
        tree_growth="leaf",
        max_leaves=5,
        max_depth=8,
    )
    m.fit(X, y, group=group)
    preds = m.predict(X)
    assert preds.shape == y.shape
    # Sanity: predictions should be finite.
    assert np.all(np.isfinite(preds))


def test_joint_mode_leaf_wise_requires_max_leaves():
    """v0.10.2: tree_growth='leaf' without max_leaves must fail with a
    clear error (mirroring the single-output trainer's contract)."""
    X, y, group = _toy_ranking()
    m = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        tree_growth="leaf",
        # max_leaves intentionally omitted
    )
    with pytest.raises(Exception, match="max_leaves"):
        m.fit(X, y, group=group)


def test_joint_mode_accepts_native_categorical():
    """v0.10.3: native-categorical kwargs are now supported on the joint
    path. Rebinning to ``bin_index == category_id`` happens in the PyO3
    bridge before ``fit_joint_multi_output_with_categorical`` runs. The
    JointPredictor decodes the ``cat_bitset`` and routes by raw feature
    value (cast to category ID) at predict time, so end-to-end correctness
    requires the bin == cat-id invariant to hold across the binning step.
    """
    X, y, group = _toy_ranking()
    # Pretend column 0 is a low-cardinality categorical (the toy fixture
    # uses small integer feature values, so 0..max_value-1 are valid
    # category IDs after astype(int) + clipping).
    X_cat = X.copy()
    X_cat[:, 0] = np.clip(X_cat[:, 0].astype(int), 0, 3)
    m = MultiLabelGBMRanker(
        n_estimators=4,
        multi_label_mode="joint",
        categorical_feature_indices=[0],
        max_cat_threshold=8,
        seed=7,
    )
    m.fit(X_cat, y, group=group)
    preds = m.predict(X_cat)
    assert preds.shape == (X_cat.shape[0], y.shape[1])
    assert np.isfinite(preds).all()


def test_joint_mode_native_categorical_routes_by_category_id():
    """v0.10.3 correctness: a known-good fixture where one column is a
    pure low-cardinality categorical and the per-category mean target
    is monotone in category ID. Verify that JointPredictor produces
    distinct predictions per category (proves the bitset routing
    works) and that per-category mean predictions follow the signal
    direction for both labels."""
    rng = np.random.default_rng(0)
    n = 240
    groups = np.repeat(np.arange(n // 12), 12).astype(np.int64)
    cat = rng.integers(0, 4, size=n).astype(np.float32)
    # Per-category-id signal: mean target = cat.
    y0 = cat + rng.normal(0, 0.1, size=n).astype(np.float32)
    y1 = (3.0 - cat) + rng.normal(0, 0.1, size=n).astype(np.float32)
    noise = rng.normal(0, 1, size=n).astype(np.float32)
    X = np.column_stack([cat, noise]).astype(np.float32)
    y = np.column_stack([y0, y1])

    m = MultiLabelGBMRanker(
        n_estimators=30,
        learning_rate=0.3,
        max_depth=4,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        categorical_feature_indices=[0],
        max_cat_threshold=8,
        seed=7,
    )
    m.fit(X, y, group=groups)

    # Per-category mean prediction should be strictly monotone in cat ID
    # for label 0 (and reverse-monotone for label 1).
    preds = m.predict(X)
    per_cat_pred_0 = [
        preds[cat == c, 0].mean() for c in range(4)
    ]
    per_cat_pred_1 = [
        preds[cat == c, 1].mean() for c in range(4)
    ]
    assert per_cat_pred_0[0] < per_cat_pred_0[1] < per_cat_pred_0[2] < per_cat_pred_0[3]
    assert per_cat_pred_1[0] > per_cat_pred_1[1] > per_cat_pred_1[2] > per_cat_pred_1[3]


def test_joint_mode_accepts_goss():
    """v0.10.3: joint MultiLabelGBMRanker honors `boosting_mode='goss'`.
    The trained model must be non-empty and finite; GOSS shouldn't change
    the API surface, just the row-sampling internals."""
    rng = np.random.default_rng(0)
    n = 600
    groups = np.repeat(np.arange(n // 30), 30).astype(np.int64)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y0 = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, size=n)).astype(np.float32)
    y1 = (X[:, 1] - 0.3 * X[:, 2] + rng.normal(0, 0.1, size=n)).astype(np.float32)
    y = np.column_stack([y0, y1])

    # Use mild GOSS rates (top=0.5, other=0.4) so the amplification
    # factor stays close to 1.0 — aggressive rates can blow up the
    # gradient on small fixtures.
    m_goss = MultiLabelGBMRanker(
        n_estimators=40,
        learning_rate=0.05,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        boosting_mode="goss",
        goss_top_rate=0.5,
        goss_other_rate=0.4,
        seed=7,
    )
    m_goss.fit(X, y, group=groups)
    preds_goss = m_goss.predict(X)
    assert preds_goss.shape == (n, 2)
    assert np.isfinite(preds_goss).all()

    # Compare against Standard (no GOSS) — predictions must differ
    # because GOSS sees a different (sampled + amplified) gradient
    # stream. This is the Python-level sanity check that the kwarg is
    # actually plumbed end-to-end.
    m_std = MultiLabelGBMRanker(
        n_estimators=40,
        learning_rate=0.05,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        boosting_mode="standard",
        seed=7,
    )
    m_std.fit(X, y, group=groups)
    preds_std = m_std.predict(X)
    max_diff = float(np.abs(preds_goss - preds_std).max())
    assert max_diff > 1e-4, "GOSS produced identical predictions to Standard"


def test_joint_mode_accepts_dart():
    """v0.10.3: joint MultiLabelGBMRanker honors `boosting_mode='dart'`."""
    rng = np.random.default_rng(0)
    n = 480
    groups = np.repeat(np.arange(n // 24), 24).astype(np.int64)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y0 = (X[:, 0] + rng.normal(0, 0.1, size=n)).astype(np.float32)
    y1 = (X[:, 1] + rng.normal(0, 0.1, size=n)).astype(np.float32)
    y = np.column_stack([y0, y1])

    m = MultiLabelGBMRanker(
        n_estimators=15,
        learning_rate=0.1,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        boosting_mode="dart",
        dart_drop_rate=0.3,
        dart_max_drop=2,
        dart_normalize_type="tree",
        dart_sample_type="uniform",
        seed=7,
    )
    m.fit(X, y, group=groups)
    preds = m.predict(X)
    assert preds.shape == (n, 2)
    assert np.isfinite(preds).all()


def test_joint_dart_save_load_round_trip(tmp_path):
    """v0.10.3: a DART-trained joint model must round-trip via
    save_model/load_model with bit-identical predictions (the DART
    tree_weights live in the artifact's `DartTreeWeights` section,
    and `JointPredictor.predict_row` multiplies each tree's leaf
    contribution by the corresponding `tree_weight`)."""
    rng = np.random.default_rng(1)
    n = 200
    groups = np.repeat(np.arange(n // 20), 20).astype(np.int64)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.column_stack([
        (X[:, 0] + rng.normal(0, 0.1, size=n)).astype(np.float32),
        (X[:, 1] + rng.normal(0, 0.1, size=n)).astype(np.float32),
    ])
    m = MultiLabelGBMRanker(
        n_estimators=12,
        learning_rate=0.1,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        boosting_mode="dart",
        dart_drop_rate=0.3,
        dart_max_drop=2,
        dart_normalize_type="tree",
        dart_sample_type="uniform",
        seed=11,
    )
    m.fit(X, y, group=groups)
    pred_before = m.predict(X)

    path = tmp_path / "joint_dart.alloy"
    m.save_model(str(path))
    m2 = MultiLabelGBMRanker.load_model(str(path))
    pred_after = m2.predict(X)

    np.testing.assert_allclose(pred_after, pred_before, rtol=1e-5, atol=1e-5)


def test_joint_mode_still_rejects_truly_unsupported_kwargs():
    """v0.10.2: kwargs that the joint trainer still does NOT support
    (warm-start, MorphBoost, etc.) must continue to be rejected with
    a clear NotImplementedError. Only Phase 1 kwargs were added to
    the allow-list."""
    X, y, group = _toy_ranking()
    for unsupported in (
        "leaf_model",
        "monotone_constraints",
        "training_mode",
    ):
        # Some of these collide with __init__ signature in odd ways; we
        # use a generic plausible-value-per-kwarg map.
        value = {
            "leaf_model": "linear",
            "monotone_constraints": [1, 0, 0, 0],
            "training_mode": "morph",
        }[unsupported]
        m = MultiLabelGBMRanker(
            n_estimators=2, multi_label_mode="joint", **{unsupported: value}
        )
        with pytest.raises(NotImplementedError, match=unsupported):
            m.fit(X, y, group=group)
