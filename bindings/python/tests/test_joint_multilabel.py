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
