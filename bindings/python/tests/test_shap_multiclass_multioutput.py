"""Tests for multiclass and multi-output (joint ranking) SHAP and SHAP interactions."""

from __future__ import annotations

import numpy as np
import pytest

from alloygbm import GBMClassifier, MultiLabelGBMRanker


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def test_multiclass_shap_additivity() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4)).astype(np.float32)
    logits_true = X[:, 0] * 1.5 - X[:, 1] * 0.8
    p = softmax(np.stack([logits_true, -logits_true, np.zeros_like(logits_true)], axis=1))
    y = np.array([rng.choice(3, p=p_row) for p_row in p], dtype=np.int32)

    model = GBMClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        seed=42,
        deterministic=True,
    )
    model.fit(X, y)

    # Test include_expected_value=True
    expected, shap_vals = model.shap_values(X, include_expected_value=True)
    assert len(expected) == 3
    assert len(shap_vals) == 3
    shap_vals = [np.array(v) for v in shap_vals]
    assert shap_vals[0].shape == (100, 4)

    # Reconstruct logits and verify softmax against predict_proba
    recon_logits = np.stack(
        [np.sum(shap_vals[c], axis=1) + expected[c] for c in range(3)], axis=1
    )
    prob_recon = softmax(recon_logits)
    prob_pred = model.predict_proba(X)
    np.testing.assert_allclose(prob_recon, prob_pred, atol=1e-5, rtol=1e-4)

    # Test include_expected_value=False
    shap_vals_only = model.shap_values(X)
    assert len(shap_vals_only) == 3
    for c in range(3):
        np.testing.assert_allclose(shap_vals_only[c], shap_vals[c])


def test_multiclass_linear_leaves_shap_additivity() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 4)).astype(np.float32)
    logits_true = X[:, 0] * 1.2 - X[:, 1] * 0.6
    p = softmax(np.stack([logits_true, -logits_true, np.zeros_like(logits_true)], axis=1))
    y = np.array([rng.choice(3, p=p_row) for p_row in p], dtype=np.int32)

    model = GBMClassifier(
        leaf_model="linear",
        n_estimators=8,
        learning_rate=0.08,
        max_depth=3,
        seed=42,
        deterministic=True,
    )
    model.fit(X, y)

    expected, shap_vals = model.shap_values(X, include_expected_value=True)
    assert len(expected) == 3
    assert len(shap_vals) == 3
    shap_vals = [np.array(v) for v in shap_vals]

    recon_logits = np.stack(
        [np.sum(shap_vals[c], axis=1) + expected[c] for c in range(3)], axis=1
    )
    prob_recon = softmax(recon_logits)
    prob_pred = model.predict_proba(X)
    np.testing.assert_allclose(prob_recon, prob_pred, atol=1e-5, rtol=1e-4)


def test_joint_ranker_shap_additivity() -> None:
    rng = np.random.default_rng(42)
    n_rows = 120
    X = rng.normal(size=(n_rows, 5)).astype(np.float32)
    y = np.stack([
        X[:, 0] * 1.0 - X[:, 1] * 0.5 + rng.normal(scale=0.1, size=n_rows),
        X[:, 2] * 0.8 + X[:, 3] * 0.4 + rng.normal(scale=0.1, size=n_rows),
        X[:, 4] * 1.2 + rng.normal(scale=0.1, size=n_rows),
    ], axis=1).astype(np.float32)
    group = np.repeat(np.arange(12, dtype=np.int32), 10)

    model = MultiLabelGBMRanker(
        multi_label_mode="joint",
        n_estimators=15,
        learning_rate=0.1,
        max_depth=3,
        seed=42,
    )
    model.fit(X, y, group=group)

    expected, shap_vals = model.shap_values(X, include_expected_value=True)
    assert len(expected) == 3
    assert len(shap_vals) == 3
    shap_vals = [np.array(v) for v in shap_vals]

    # Additivity to predict
    predicted = model.predict(X)
    for c in range(3):
        reconstructed = np.sum(shap_vals[c], axis=1) + expected[c]
        np.testing.assert_allclose(reconstructed, predicted[:, c], atol=1e-4, rtol=1e-4)


def test_joint_ranker_shap_interactions() -> None:
    rng = np.random.default_rng(42)
    n_rows = 80
    X = rng.normal(size=(n_rows, 3)).astype(np.float32)
    y = np.stack([
        X[:, 0] * 1.0 - X[:, 1] * 0.5 + rng.normal(scale=0.1, size=n_rows),
        X[:, 2] * 0.8 + rng.normal(scale=0.1, size=n_rows),
    ], axis=1).astype(np.float32)
    group = np.repeat(np.arange(8, dtype=np.int32), 10)

    model = MultiLabelGBMRanker(
        multi_label_mode="joint",
        n_estimators=10,
        learning_rate=0.08,
        max_depth=3,
        seed=42,
    )
    model.fit(X, y, group=group)

    expected, shap_vals = model.shap_values(X, include_expected_value=True)
    shap_vals = [np.array(v) for v in shap_vals]

    predicted = model.predict(X)

    # SHAP interactions additivity, symmetry, and row-marginal
    expected_int, interaction_vals = model.shap_interaction_values(X, include_expected_value=True)
    assert len(expected_int) == 2
    assert len(interaction_vals) == 2
    interaction_vals = [np.array(v) for v in interaction_vals]
    assert interaction_vals[0].shape == (80, 3, 3)

    for c in range(2):
        # Expected values should match
        np.testing.assert_allclose(expected_int[c], expected[c])

        # Interaction additivity to prediction
        recon_int = np.sum(interaction_vals[c], axis=(1, 2)) + expected_int[c]
        np.testing.assert_allclose(recon_int, predicted[:, c], atol=1e-4, rtol=1e-4)

        # Symmetry: interaction[c][r][i][j] == interaction[c][r][j][i]
        for r in range(n_rows):
            np.testing.assert_allclose(interaction_vals[c][r], interaction_vals[c][r].T, atol=1e-5)

        # Row marginal: sum(interaction[c], axis=2) == shap_values[c]
        row_marginal = np.sum(interaction_vals[c], axis=2)
        np.testing.assert_allclose(row_marginal, shap_vals[c], atol=1e-4, rtol=1e-4)
