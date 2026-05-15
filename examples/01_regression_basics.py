"""GBMRegressor: train, predict, evaluate on California Housing.

Demonstrates the most common AlloyGBM workflow: train a regressor on a
real tabular dataset, hold out a validation set, fit with early
stopping, and evaluate with RMSE.
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
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=7
    )

    # `training_policy="auto"` (the default) applies dataset-aware
    # heuristics for min_split_gain / min_data_in_leaf / regularization
    # — exactly what you want for a first training run.
    model = GBMRegressor(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=2000,
        early_stopping_rounds=50,
        deterministic=True,
        seed=7,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    preds = model.predict(X_test)
    test_rmse = rmse(y_test, preds)

    print(f"rounds trained:   {model.n_estimators_}")
    print(f"best iteration:   {model.best_iteration_}")
    print(f"best validation:  {model.best_score_:.4f}")
    print(f"test RMSE:        {test_rmse:.4f}")


if __name__ == "__main__":
    main()
