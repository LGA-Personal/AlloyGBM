"""Warm-start: resume training from a saved model.

Two common patterns:

1.  ``warm_start=True`` — fit the same estimator instance again to add
    more rounds to the existing ensemble.
2.  ``init_model=<previous>`` — pass a previously-fitted model into a
    new estimator's constructor to continue training from those trees.

Both produce a model whose ensemble is the concatenation of the
original trees and the newly trained trees.  This example uses
``training_policy="manual"`` so the round counts are deterministic
(``auto`` early-stops, which makes the "added N rounds" headline
fuzzy).
"""

from __future__ import annotations

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from alloygbm import GBMRegressor, rmse


def main() -> None:
    data = fetch_california_housing(as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=7
    )

    # Phase 1: train for exactly 100 rounds.  `training_policy="manual"`
    # disables auto early-stopping so we get exactly the round count we
    # asked for — this makes the warm-start delta easy to read.
    model = GBMRegressor(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=100,
        training_policy="manual",
        warm_start=True,
        deterministic=True,
        seed=7,
    )
    model.fit(X_train, y_train)
    phase1_rmse = rmse(y_test, model.predict(X_test))
    phase1_trees = model.n_estimators_

    # Phase 2: continue.  Bumping n_estimators to 200 means AlloyGBM
    # trains 100 *additional* rounds on top of the existing ensemble.
    model.set_params(n_estimators=200)
    model.fit(X_train, y_train)
    phase2_rmse = rmse(y_test, model.predict(X_test))
    phase2_trees = model.n_estimators_

    print(f"phase 1: trees={phase1_trees:>3}, test RMSE={phase1_rmse:.4f}")
    print(f"phase 2: trees={phase2_trees:>3}, test RMSE={phase2_rmse:.4f}")
    print(f"phase 2 added {phase2_trees - phase1_trees} trees to the ensemble.")


if __name__ == "__main__":
    main()
