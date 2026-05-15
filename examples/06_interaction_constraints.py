"""Interaction constraints: restrict root-to-leaf paths to feature groups.

LightGBM-compatible feature interaction constraints.  Each inner list
defines a group of features; any root-to-leaf path may only split on
features from a single still-active group.  Features outside all groups
are unrestricted.

Useful when you have domain knowledge that certain feature groups
should not interact (e.g. demographic features and product features in
a recommender, regulatory feature isolation, interpretability
constraints).
"""

from __future__ import annotations

import numpy as np

from alloygbm import GBMRegressor, rmse


def _make_synthetic(seed: int = 7):
    rng = np.random.RandomState(seed)
    n_rows, n_features = 1500, 10
    X = rng.randn(n_rows, n_features).astype(np.float32)
    y = (
        X[:, 0] * 2.0
        + X[:, 1]
        + X[:, 5] * 1.5
        - X[:, 6]
        + rng.randn(n_rows).astype(np.float32) * 0.2
    )
    return X, y


def main() -> None:
    X, y = _make_synthetic()
    split = int(len(y) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Define two non-overlapping feature groups.  Within any single
    # tree, every split on a root-to-leaf path must come from the same
    # group (or from features 8/9 which are in no group and therefore
    # unrestricted).
    groups = [[0, 1, 2, 3, 4], [5, 6, 7]]

    unrestricted = GBMRegressor(
        learning_rate=0.05, max_depth=6, n_estimators=300, seed=7, deterministic=True
    )
    unrestricted.fit(X_train, y_train)

    constrained = GBMRegressor(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=300,
        interaction_constraints=groups,
        seed=7,
        deterministic=True,
    )
    constrained.fit(X_train, y_train)

    print(f"unrestricted test RMSE:    {rmse(y_test, unrestricted.predict(X_test)):.4f}")
    print(f"constrained test RMSE:     {rmse(y_test, constrained.predict(X_test)):.4f}")
    print(f"interaction groups:        {groups}")


if __name__ == "__main__":
    main()
