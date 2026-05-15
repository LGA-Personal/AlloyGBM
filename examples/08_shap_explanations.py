"""SHAP explanations: per-row attributions + global feature importance.

AlloyGBM ships a native TreeSHAP implementation that computes exact
Shapley values in polynomial time.  Two entry points:

- ``model.shap_values(X)`` — per-row, per-feature attributions
  satisfying ``sum(values) + expected_value == predict(x)`` (exact for
  ``leaf_model="constant"``; best-effort for ``leaf_model="linear"``).
- ``model.feature_importances(X)`` — global SHAP-based importance,
  aggregating per-row attributions across the dataset.
"""

from __future__ import annotations

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from alloygbm import GBMRegressor


def main() -> None:
    # `as_frame=True` returns pandas DataFrames; AlloyGBM captures the
    # column names at fit time so SHAP output is labeled with the
    # original feature names.
    data = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=7
    )

    model = GBMRegressor(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        training_policy="manual",
        deterministic=True,
        seed=7,
    )
    model.fit(X_train, y_train)

    # Per-row attributions for the first 3 test rows.
    # shap_values returns a list of per-row attribution lists; convert
    # to a numpy array for convenient row math.
    import numpy as np

    expected_value, raw_values = model.shap_values(
        X_test[:3], include_expected_value=True
    )
    values = np.asarray(raw_values, dtype=float)

    print(f"expected_value (model baseline output): {float(expected_value):.4f}")
    print(f"shap_values shape:                      {values.shape}")

    # Additivity: sum(values_row) + expected_value ≈ predict(row).
    preds = model.predict(X_test[:3])
    for i, p in enumerate(preds):
        reconstructed = float(values[i].sum() + expected_value)
        print(
            f"row {i}: predict={float(p):.4f}  reconstructed={reconstructed:.4f}  "
            f"diff={abs(float(p) - reconstructed):.2e}"
        )

    # Global importance — sorted descending so the top features
    # surface first.  Note: the SHAP additivity check has a tight
    # absolute tolerance (1e-5) that can be exceeded by f32 round-off
    # on large samples (see docs/limitations.md), so we keep the
    # explain set to a representative 300-row subsample.
    importance = model.feature_importances(X_test[:200])
    importance_sorted = sorted(importance, key=lambda pair: pair[1], reverse=True)

    print("\nTop 5 features by global SHAP importance:")
    for name, score in importance_sorted[:5]:
        print(f"  {name:20s}  {score:.4f}")


if __name__ == "__main__":
    main()
