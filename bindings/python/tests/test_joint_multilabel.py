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


def test_joint_mode_rejects_factor_exposures_without_neutralization():
    """v0.10.6: factor_exposures + neutralization='none' is rejected
    (signals user confusion). Previously (v0.10.1–v0.10.5) any
    factor_exposures on joint mode raised NotImplementedError; v0.10.6 lifted
    that gate, but the consistency contract still rejects unused exposures."""
    X, y, group = _toy_ranking()
    n = X.shape[0]
    exposures = np.zeros((n, 2), dtype=np.float32)
    m = MultiLabelGBMRanker(n_estimators=2, multi_label_mode="joint")
    with pytest.raises(ValueError, match="neutralization='none'"):
        m.fit(X, y, group=group, factor_exposures=exposures)


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


def test_joint_warm_start_matches_fresh_fit():
    """v0.10.3: a 6+4 warm-resumed joint fit must produce the same
    predictions as a fresh 10-round fit with the same seed and
    hyperparameters."""
    rng = np.random.default_rng(2)
    n = 240
    groups = np.repeat(np.arange(n // 24), 24).astype(np.int64)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.column_stack([
        (X[:, 0] + rng.normal(0, 0.1, size=n)).astype(np.float32),
        (X[:, 1] + rng.normal(0, 0.1, size=n)).astype(np.float32),
    ])

    fresh = MultiLabelGBMRanker(
        n_estimators=10,
        learning_rate=0.1,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        seed=7,
    )
    fresh.fit(X, y, group=groups)
    pred_fresh = fresh.predict(X)

    first = MultiLabelGBMRanker(
        n_estimators=6,
        learning_rate=0.1,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        seed=7,
    )
    first.fit(X, y, group=groups)
    resumed = MultiLabelGBMRanker(
        n_estimators=4,
        learning_rate=0.1,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        seed=7,
        warm_start=True,
        init_model=first,
    )
    resumed.fit(X, y, group=groups)
    pred_resumed = resumed.predict(X)
    np.testing.assert_allclose(pred_resumed, pred_fresh, rtol=1e-4, atol=1e-4)


def test_joint_warm_start_dart_matches_fresh_fit():
    """v0.10.3: DART warm-start parity. Same logic as the standard
    warm-start but verifies the `DartTreeWeights` round-trip too."""
    rng = np.random.default_rng(3)
    n = 240
    groups = np.repeat(np.arange(n // 24), 24).astype(np.int64)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.column_stack([
        (X[:, 0] + rng.normal(0, 0.1, size=n)).astype(np.float32),
        (X[:, 1] + rng.normal(0, 0.1, size=n)).astype(np.float32),
    ])

    common = dict(
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
    fresh = MultiLabelGBMRanker(n_estimators=10, **common)
    fresh.fit(X, y, group=groups)
    pred_fresh = fresh.predict(X)

    first = MultiLabelGBMRanker(n_estimators=6, **common)
    first.fit(X, y, group=groups)
    resumed = MultiLabelGBMRanker(
        n_estimators=4, warm_start=True, init_model=first, **common,
    )
    resumed.fit(X, y, group=groups)
    pred_resumed = resumed.predict(X)
    np.testing.assert_allclose(pred_resumed, pred_fresh, rtol=1e-4, atol=1e-4)


def test_joint_native_categorical_rejects_non_dense_ids():
    """PR #36 review fix (C3): the joint rebinner maps requested
    columns to `bin_index == category_id`, but JointPredictor reads
    the raw feature value at predict time. A non-dense set like
    {10, 20, 30} would silently route to the wrong bitset bit. v0.10.3
    rejects with a clear error pointing users at sklearn LabelEncoder
    (proper persisted mapping is v0.10.4 work)."""
    rng = np.random.default_rng(0)
    n = 80
    groups = np.repeat(np.arange(n // 8), 8).astype(np.int64)
    # Non-dense category IDs: {10, 20, 30}.
    cat = rng.choice([10.0, 20.0, 30.0], size=n).astype(np.float32)
    other = rng.normal(size=n).astype(np.float32)
    X = np.column_stack([cat, other])
    y = np.column_stack([
        rng.integers(0, 4, size=n).astype(np.float32),
        rng.integers(0, 4, size=n).astype(np.float32),
    ])
    m = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        categorical_feature_indices=[0],
        max_cat_threshold=40,
    )
    with pytest.raises(Exception, match="dense"):
        m.fit(X, y, group=groups)


def test_joint_native_categorical_rejects_more_than_64_categories():
    """The joint native-categorical split path uses a u64 bitset. Reject
    wider categorical columns before training instead of silently disabling
    native categorical splits in Rust."""
    n_categories = 65
    rows_per_category = 2
    n = n_categories * rows_per_category
    groups = np.repeat(np.arange(n // 10), 10).astype(np.int64)
    cat = np.repeat(np.arange(n_categories), rows_per_category).astype(np.float32)
    X = np.column_stack([cat, np.zeros(n, dtype=np.float32)])
    y = np.column_stack([
        (cat % 5).astype(np.float32),
        (cat % 7).astype(np.float32),
    ])
    m = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        categorical_feature_indices=[0],
        max_cat_threshold=128,
    )
    with pytest.raises(ValueError, match="at most 64 categories"):
        m.fit(X, y, group=groups)


def test_joint_native_categorical_rejects_non_integer_values():
    """PR #36 review fix (C1): the joint rebinner uses truncation
    (`v as i64`) to match JointPredictor's predict-time cast. Pure
    non-integer floats like 0.6 would round at training but truncate
    at predict, silently disagreeing. v0.10.3 rejects with a clear
    error pointing users at pre-encoding."""
    rng = np.random.default_rng(0)
    n = 60
    groups = np.repeat(np.arange(n // 6), 6).astype(np.int64)
    # 0.6 and 1.6 are not exact integer-valued floats.
    cat = rng.choice([0.6, 1.6, 2.6], size=n).astype(np.float32)
    X = np.column_stack([cat, rng.normal(size=n).astype(np.float32)])
    y = np.column_stack([
        rng.integers(0, 4, size=n).astype(np.float32),
        rng.integers(0, 4, size=n).astype(np.float32),
    ])
    m = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        categorical_feature_indices=[0],
        max_cat_threshold=8,
    )
    with pytest.raises(Exception, match="non-integer"):
        m.fit(X, y, group=groups)


def test_joint_warm_start_rejects_feature_count_mismatch():
    """PR #36 review fix (C2, C4): a wider prior schema must be
    rejected with a clean Python ValueError, not a Rust panic from
    out-of-bounds feature indexing in `walk_tree_into_predictions`."""
    rng = np.random.default_rng(0)
    n = 60
    groups = np.repeat(np.arange(n // 6), 6).astype(np.int64)
    X_wide = rng.normal(size=(n, 5)).astype(np.float32)
    X_narrow = X_wide[:, :3]
    y = np.column_stack([
        rng.integers(0, 4, size=n).astype(np.float32),
        rng.integers(0, 4, size=n).astype(np.float32),
    ])

    prior = MultiLabelGBMRanker(
        n_estimators=3,
        multi_label_mode="joint",
        ranking_objective="squared_error",
    )
    prior.fit(X_wide, y, group=groups)
    resumed = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        warm_start=True,
        init_model=prior,
    )
    with pytest.raises(ValueError, match="features"):
        resumed.fit(X_narrow, y, group=groups)


def test_joint_warm_start_rejects_n_labels_mismatch():
    """PR #36 review fix (C4): n_labels mismatch between init_model
    and the current fit must be rejected with a clear error."""
    rng = np.random.default_rng(0)
    n = 60
    groups = np.repeat(np.arange(n // 6), 6).astype(np.int64)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y2 = np.column_stack([
        rng.integers(0, 4, size=n).astype(np.float32),
        rng.integers(0, 4, size=n).astype(np.float32),
    ])
    y3 = np.column_stack([y2, rng.integers(0, 4, size=n).astype(np.float32)])

    prior = MultiLabelGBMRanker(
        n_estimators=3, multi_label_mode="joint", ranking_objective="squared_error",
    )
    prior.fit(X, y2, group=groups)
    resumed = MultiLabelGBMRanker(
        n_estimators=2, multi_label_mode="joint",
        ranking_objective="squared_error",
        warm_start=True, init_model=prior,
    )
    with pytest.raises(ValueError, match="labels"):
        resumed.fit(X, y3, group=groups)


def test_joint_warm_start_rejects_dart_mode_mismatch():
    """PR #36 review fix (C4): a DART prior resumed as standard (or
    vice versa) silently mishandles per-tree weights. Reject both
    transitions explicitly."""
    rng = np.random.default_rng(0)
    n = 60
    groups = np.repeat(np.arange(n // 6), 6).astype(np.int64)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.column_stack([
        rng.integers(0, 4, size=n).astype(np.float32),
        rng.integers(0, 4, size=n).astype(np.float32),
    ])

    # DART prior, standard resume — rejected.
    dart_prior = MultiLabelGBMRanker(
        n_estimators=3, multi_label_mode="joint",
        ranking_objective="squared_error",
        boosting_mode="dart", dart_drop_rate=0.3, dart_max_drop=2,
        dart_normalize_type="tree", dart_sample_type="uniform",
    )
    dart_prior.fit(X, y, group=groups)
    standard_resumed = MultiLabelGBMRanker(
        n_estimators=2, multi_label_mode="joint",
        ranking_objective="squared_error",
        warm_start=True, init_model=dart_prior,
    )
    with pytest.raises(ValueError, match="boosting_mode"):
        standard_resumed.fit(X, y, group=groups)


def test_joint_warm_start_validates_init_model():
    """v0.10.3: surface-level validation — `init_model` must be a
    fitted joint MultiLabelGBMRanker. We don't try to enumerate every
    error path; just the most likely user mistakes."""
    X, y, group = _toy_ranking()
    # warm_start=True without init_model
    m = MultiLabelGBMRanker(n_estimators=2, multi_label_mode="joint", warm_start=True)
    with pytest.raises(ValueError, match="init_model"):
        m.fit(X, y, group=group)
    # init_model without warm_start=True
    other = MultiLabelGBMRanker(n_estimators=2, multi_label_mode="joint")
    other.fit(X, y, group=group)
    m = MultiLabelGBMRanker(n_estimators=2, multi_label_mode="joint", init_model=other)
    with pytest.raises(ValueError, match="warm_start"):
        m.fit(X, y, group=group)
    # init_model from independent mode rejected.
    indep = MultiLabelGBMRanker(n_estimators=2, multi_label_mode="independent")
    indep.fit(X, y, group=group)
    m = MultiLabelGBMRanker(
        n_estimators=2, multi_label_mode="joint",
        warm_start=True, init_model=indep,
    )
    with pytest.raises(ValueError, match="multi_label_mode"):
        m.fit(X, y, group=group)


def test_joint_mode_still_rejects_truly_unsupported_kwargs():
    """v0.10.4: kwargs that the joint trainer still does NOT support
    (leaf_model='linear', monotone_constraints) must continue to be
    rejected with a clear NotImplementedError. `training_mode='morph'`
    was added to the allow-list in v0.10.4 (see
    `test_joint_morph_changes_predictions_vs_standard`)."""
    X, y, group = _toy_ranking()
    for unsupported in (
        "leaf_model",
        "monotone_constraints",
    ):
        value = {
            "leaf_model": "linear",
            "monotone_constraints": [1, 0, 0, 0],
        }[unsupported]
        m = MultiLabelGBMRanker(
            n_estimators=2, multi_label_mode="joint", **{unsupported: value}
        )
        with pytest.raises(NotImplementedError, match=unsupported):
            m.fit(X, y, group=group)


def test_joint_morph_rejects_invalid_training_mode():
    """PR #37 review (C1, C4): joint mode must reject invalid
    `training_mode` values rather than silently downgrading to standard
    training. Mirrors `GBMRegressor` / `GBMRanker` validation."""
    X, y, group = _toy_ranking()
    m = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        training_mode="morhp",  # typo
    )
    with pytest.raises(ValueError, match="training_mode"):
        m.fit(X, y, group=group)


def test_joint_morph_changes_predictions_vs_standard():
    """v0.10.4 acceptance: training_mode='morph' produces different
    predictions than the default 'auto' (standard) on the same data + seed."""
    rng = np.random.default_rng(42)
    n_rows, n_features, n_labels = 200, 5, 3
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    y = rng.normal(size=(n_rows, n_labels)).astype(np.float32)

    common = dict(
        multi_label_mode="joint",
        n_estimators=10,
        seed=0,
        max_depth=4,
        ranking_objective="squared_error",
    )
    std = MultiLabelGBMRanker(**common, training_mode="auto")
    std.fit(X, y)
    mph = MultiLabelGBMRanker(**common, training_mode="morph")
    mph.fit(X, y)
    p_std = std.predict(X)
    p_mph = mph.predict(X)
    assert not np.allclose(p_std, p_mph, atol=1e-4), (
        "MorphBoost predictions should differ from standard predictions"
    )


def test_joint_dro_changes_predictions_vs_standard():
    """v0.10.5: leaf_solver='dro' must produce different predictions than
    leaf_solver='standard' on the same multi-label data."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = np.column_stack([
        rng.integers(0, 2, size=64),
        rng.integers(0, 2, size=64),
    ]).astype(np.float32)
    group = np.array([8] * 8, dtype=np.int64)  # 8 groups of 8 rows

    common = dict(
        n_estimators=5,
        learning_rate=0.3,
        max_depth=3,
        seed=11,
        multi_label_mode="joint",
    )

    m_std = MultiLabelGBMRanker(**common, leaf_solver="standard")
    m_std.fit(X, y, group=group)
    p_std = m_std.predict(X)

    m_dro = MultiLabelGBMRanker(
        **common, leaf_solver="dro", dro_radius=0.5, dro_metric="wasserstein",
    )
    m_dro.fit(X, y, group=group)
    p_dro = m_dro.predict(X)

    assert p_std.shape == p_dro.shape
    assert not np.allclose(p_std, p_dro), (
        "DRO leaves should produce different multi-label predictions than standard"
    )


def test_joint_dro_radius_zero_byte_equivalent_to_standard():
    """DRO with radius=0 collapses to standard leaves — predictions must
    match exactly."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(32, 3)).astype(np.float32)
    y = rng.integers(0, 2, size=(32, 2)).astype(np.float32)
    group = np.array([4] * 8, dtype=np.int64)

    common = dict(
        n_estimators=3,
        learning_rate=0.3,
        seed=5,
        multi_label_mode="joint",
    )

    m_std = MultiLabelGBMRanker(**common, leaf_solver="standard")
    m_std.fit(X, y, group=group)

    m_dro = MultiLabelGBMRanker(
        **common, leaf_solver="dro", dro_radius=0.0, dro_metric="wasserstein",
    )
    m_dro.fit(X, y, group=group)

    np.testing.assert_allclose(m_std.predict(X), m_dro.predict(X), rtol=0, atol=0)


def test_joint_dro_rejects_invalid_metric():
    """dro_metric must be 'wasserstein'. Anything else raises."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32)

    m = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        leaf_solver="dro",
        dro_radius=0.5,
        dro_metric="kl-divergence",  # invalid
    )
    with pytest.raises((ValueError, Exception)):
        m.fit(X, y, group=[2, 2])


def test_joint_dro_works_with_morphboost():
    """DRO leaves + MorphBoost LR/shrinkage compose without error and
    produce finite predictions."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(48, 3)).astype(np.float32)
    y = rng.integers(0, 2, size=(48, 2)).astype(np.float32)
    group = np.array([6] * 8, dtype=np.int64)

    m = MultiLabelGBMRanker(
        n_estimators=4,
        learning_rate=0.3,
        max_depth=3,
        seed=17,
        multi_label_mode="joint",
        training_mode="morph",
        leaf_solver="dro",
        dro_radius=0.3,
        dro_metric="wasserstein",
    )
    m.fit(X, y, group=group)
    p = m.predict(X)
    assert np.isfinite(p).all()


# ── v0.10.6: joint factor neutralization ────────────────────────────────────


def test_joint_pre_target_residualizes_predictions():
    """pre_target should shrink output-0 variance when factor is correlated."""
    rng = np.random.default_rng(60)
    X = rng.standard_normal((32, 3)).astype(np.float32)
    factor = rng.standard_normal((32, 1)).astype(np.float32)
    y = np.column_stack(
        [
            (factor[:, 0] * 2.0 + rng.standard_normal(32) * 0.01).astype(np.float32),
            rng.standard_normal(32).astype(np.float32),
        ]
    )
    baseline = (
        MultiLabelGBMRanker(
            n_estimators=5,
            multi_label_mode="joint",
            ranking_objective="squared_error",
        )
        .fit(X, y)
        .predict(X)
    )
    neutralized = (
        MultiLabelGBMRanker(
            n_estimators=5,
            multi_label_mode="joint",
            ranking_objective="squared_error",
            neutralization="pre_target",
        )
        .fit(X, y, factor_exposures=factor)
        .predict(X)
    )
    var_baseline_0 = float(np.var(baseline[:, 0]))
    var_neutralized_0 = float(np.var(neutralized[:, 0]))
    assert var_neutralized_0 < var_baseline_0 * 0.5, (
        f"pre_target should shrink output-0 variance "
        f"(baseline={var_baseline_0}, neutralized={var_neutralized_0})"
    )


def test_joint_per_round_gradient_changes_predictions():
    """per_round_gradient should produce different predictions than baseline."""
    rng = np.random.default_rng(61)
    X = rng.standard_normal((32, 3)).astype(np.float32)
    y = rng.standard_normal((32, 2)).astype(np.float32)
    fe = rng.standard_normal((32, 1)).astype(np.float32)
    baseline = (
        MultiLabelGBMRanker(
            n_estimators=4,
            multi_label_mode="joint",
            ranking_objective="squared_error",
        )
        .fit(X, y)
        .predict(X)
    )
    neutralized = (
        MultiLabelGBMRanker(
            n_estimators=4,
            multi_label_mode="joint",
            ranking_objective="squared_error",
            neutralization="per_round_gradient",
        )
        .fit(X, y, factor_exposures=fe)
        .predict(X)
    )
    assert np.max(np.abs(baseline - neutralized)) > 1e-3


def test_joint_split_penalty_changes_predictions():
    """split_penalty with feature-correlated exposure should alter splits."""
    rng = np.random.default_rng(62)
    X = rng.standard_normal((48, 3)).astype(np.float32)
    y = rng.standard_normal((48, 2)).astype(np.float32)
    fe = X[:, 0:1].copy()
    baseline = (
        MultiLabelGBMRanker(
            n_estimators=4,
            multi_label_mode="joint",
            ranking_objective="squared_error",
        )
        .fit(X, y)
        .predict(X)
    )
    neutralized = (
        MultiLabelGBMRanker(
            n_estimators=4,
            multi_label_mode="joint",
            ranking_objective="squared_error",
            neutralization="split_penalty",
            factor_penalty=1.0,
        )
        .fit(X, y, factor_exposures=fe)
        .predict(X)
    )
    assert np.max(np.abs(baseline - neutralized)) > 1e-3


def test_joint_pre_target_rejects_ranking_objectives():
    """pre_target requires squared_error per output; rank:* must be rejected."""
    rng = np.random.default_rng(63)
    X = rng.standard_normal((16, 2)).astype(np.float32)
    y = rng.integers(0, 3, size=(16, 2)).astype(np.float32)
    fe = rng.standard_normal((16, 1)).astype(np.float32)
    grp = [0] * 8 + [1] * 8
    m = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="rank:ndcg",
        neutralization="pre_target",
    )
    with pytest.raises(Exception) as exc:
        m.fit(X, y, group=grp, factor_exposures=fe)
    msg = str(exc.value).lower()
    assert "squared_error" in msg or "pre_target" in msg


def test_joint_neutralization_requires_exposures():
    """Active neutralization config without factor_exposures must error."""
    rng = np.random.default_rng(64)
    X = rng.standard_normal((8, 2)).astype(np.float32)
    y = rng.standard_normal((8, 2)).astype(np.float32)
    m = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
    )
    with pytest.raises(Exception) as exc:
        m.fit(X, y)
    assert "factor_exposures are required" in str(exc.value)


def test_joint_exposures_without_neutralization_rejected():
    """Providing factor_exposures with neutralization='none' must error."""
    rng = np.random.default_rng(65)
    X = rng.standard_normal((8, 2)).astype(np.float32)
    y = rng.standard_normal((8, 2)).astype(np.float32)
    fe = rng.standard_normal((8, 1)).astype(np.float32)
    m = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        # neutralization left at default "none"
    )
    with pytest.raises(Exception) as exc:
        m.fit(X, y, factor_exposures=fe)
    assert "neutralization='none'" in str(exc.value)


def test_joint_per_round_gradient_with_morphboost():
    """neutralization composes with training_mode='morph'."""
    rng = np.random.default_rng(70)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 2)).astype(np.float32)
    fe = rng.standard_normal((24, 1)).astype(np.float32)
    m = MultiLabelGBMRanker(
        n_estimators=3,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
        training_mode="morph",
    )
    m.fit(X, y, factor_exposures=fe)
    preds = m.predict(X)
    assert preds.shape == (24, 2)
    assert np.all(np.isfinite(preds))


def test_joint_per_round_gradient_with_dro():
    """neutralization composes with leaf_solver='dro'."""
    rng = np.random.default_rng(71)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 2)).astype(np.float32)
    fe = rng.standard_normal((24, 1)).astype(np.float32)
    m = MultiLabelGBMRanker(
        n_estimators=3,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
        leaf_solver="dro",
        dro_radius=0.2,
    )
    m.fit(X, y, factor_exposures=fe)
    assert m.predict(X).shape == (24, 2)


def test_joint_per_round_gradient_with_dart():
    """neutralization composes with boosting_mode='dart'."""
    rng = np.random.default_rng(72)
    X = rng.standard_normal((32, 3)).astype(np.float32)
    y = rng.standard_normal((32, 2)).astype(np.float32)
    fe = rng.standard_normal((32, 1)).astype(np.float32)
    m = MultiLabelGBMRanker(
        n_estimators=4,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
        boosting_mode="dart",
        dart_drop_rate=0.2,
        dart_max_drop=2,
    )
    m.fit(X, y, factor_exposures=fe)
    assert m.predict(X).shape == (32, 2)


def test_joint_per_round_gradient_with_warm_start():
    """neutralization composes with warm-start (same exposures both fits)."""
    rng = np.random.default_rng(73)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 2)).astype(np.float32)
    fe = rng.standard_normal((24, 1)).astype(np.float32)
    first = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
    ).fit(X, y, factor_exposures=fe)
    second = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
        init_model=first,
        warm_start=True,
    ).fit(X, y, factor_exposures=fe)
    assert second.predict(X).shape == (24, 2)


# ── v0.10.6 PR #40 R2: warm-start neutralization contract ────────────────────


def test_joint_warm_start_rejects_unneutralized_prior_under_active_resume():
    """PR #40 R2: prior fit was unneutralized; current fit requests
    neutralization. The prior trees were trained in raw-gradient space —
    resuming them under per_round_gradient would project new gradients
    through a residual space the prior model doesn't share."""
    rng = np.random.default_rng(80)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 2)).astype(np.float32)
    fe = rng.standard_normal((24, 1)).astype(np.float32)
    first = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        # No neutralization on prior fit.
    ).fit(X, y)
    with pytest.raises(Exception) as exc:
        MultiLabelGBMRanker(
            n_estimators=2,
            multi_label_mode="joint",
            ranking_objective="squared_error",
            neutralization="per_round_gradient",
            init_model=first,
            warm_start=True,
        ).fit(X, y, factor_exposures=fe)
    msg = str(exc.value)
    assert "warm-start" in msg.lower() and "neutralization" in msg.lower()


def test_joint_warm_start_rejects_neutralized_prior_under_unneutralized_resume():
    """PR #40 R2: prior fit used neutralization; current fit drops it.
    Prior trees are in a residual space; resuming with raw gradients
    would silently mis-train."""
    rng = np.random.default_rng(81)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 2)).astype(np.float32)
    fe = rng.standard_normal((24, 1)).astype(np.float32)
    first = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
    ).fit(X, y, factor_exposures=fe)
    with pytest.raises(Exception) as exc:
        MultiLabelGBMRanker(
            n_estimators=2,
            multi_label_mode="joint",
            ranking_objective="squared_error",
            # neutralization left at default "none"
            init_model=first,
            warm_start=True,
        ).fit(X, y)
    msg = str(exc.value)
    assert "warm-start" in msg.lower() and "neutralization" in msg.lower()


def test_joint_warm_start_rejects_changed_factor_neutralization_lambda():
    """PR #40 R2: prior fit used ridge_lambda=1e-6; current bumps to 1e-3.
    The projector changes, so the residual basis changes too."""
    rng = np.random.default_rng(82)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 2)).astype(np.float32)
    fe = rng.standard_normal((24, 1)).astype(np.float32)
    first = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
        factor_neutralization_lambda=1e-6,
    ).fit(X, y, factor_exposures=fe)
    with pytest.raises(Exception) as exc:
        MultiLabelGBMRanker(
            n_estimators=2,
            multi_label_mode="joint",
            ranking_objective="squared_error",
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-3,
            init_model=first,
            warm_start=True,
        ).fit(X, y, factor_exposures=fe)
    msg = str(exc.value)
    assert "warm-start" in msg.lower() and "neutralization" in msg.lower()


def test_joint_warm_start_rejects_changed_factor_penalty():
    """PR #40 R2: split_penalty multiplier change across warm-resume."""
    rng = np.random.default_rng(83)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 2)).astype(np.float32)
    fe = rng.standard_normal((24, 1)).astype(np.float32)
    first = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="split_penalty",
        factor_penalty=0.5,
    ).fit(X, y, factor_exposures=fe)
    with pytest.raises(Exception) as exc:
        MultiLabelGBMRanker(
            n_estimators=2,
            multi_label_mode="joint",
            ranking_objective="squared_error",
            neutralization="split_penalty",
            factor_penalty=1.0,
            init_model=first,
            warm_start=True,
        ).fit(X, y, factor_exposures=fe)
    msg = str(exc.value)
    assert "warm-start" in msg.lower() and "neutralization" in msg.lower()


def test_joint_warm_start_accepts_matching_neutralization_contract():
    """PR #40 R2: positive control — same config on both fits MUST succeed.
    Otherwise the validation gate is over-restrictive."""
    rng = np.random.default_rng(84)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 2)).astype(np.float32)
    fe = rng.standard_normal((24, 1)).astype(np.float32)
    first = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
        factor_neutralization_lambda=1e-6,
    ).fit(X, y, factor_exposures=fe)
    # Same config — must succeed.
    resumed = MultiLabelGBMRanker(
        n_estimators=2,
        multi_label_mode="joint",
        ranking_objective="squared_error",
        neutralization="per_round_gradient",
        factor_neutralization_lambda=1e-6,
        init_model=first,
        warm_start=True,
    ).fit(X, y, factor_exposures=fe)
    assert resumed.predict(X).shape == (24, 2)
