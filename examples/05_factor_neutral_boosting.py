"""Factor-neutral boosting: project gradients away from nuisance factors.

A common pattern in quantitative finance: you know certain common
factors (sector, beta, momentum) explain a lot of return variance and
you don't want your model to spend tree capacity learning those.
``neutralization="per_round_gradient"`` projects each boosting round's
gradients onto the orthogonal complement of the factor exposure
matrix.

This example synthesizes a dataset where two known factors contribute
to the target, then trains two regressors — one without and one with
neutralization — and reports the factor-effectiveness diagnostic that
quantifies how much projected signal was removed each round.
"""

from __future__ import annotations

import numpy as np

from alloygbm import GBMRegressor, rmse


def _make_factor_dataset(seed: int = 7):
    rng = np.random.RandomState(seed)
    n_rows, n_features, n_factors = 2000, 10, 2

    X = rng.randn(n_rows, n_features).astype(np.float32)
    # Two factor exposures (sector dummies, say).
    F = rng.randn(n_rows, n_factors).astype(np.float32)

    # Target depends on both X and F.
    y = (
        0.5 * X[:, 0]
        + 0.3 * X[:, 1]
        + 1.2 * F[:, 0]
        + 0.8 * F[:, 1]
        + rng.randn(n_rows).astype(np.float32) * 0.2
    )
    return X, y, F


def main() -> None:
    X, y, F = _make_factor_dataset()
    split = int(len(y) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    F_train = F[:split]

    # Plain training — model learns whatever variance the features
    # explain, including factor-correlated signal.
    plain = GBMRegressor(
        learning_rate=0.05, max_depth=4, n_estimators=300, seed=7, deterministic=True
    )
    plain.fit(X_train, y_train)
    plain_rmse = rmse(y_test, plain.predict(X_test))

    # Factor-neutral training — gradients are projected away from F at
    # every round.  The model can only learn signal orthogonal to F.
    neutral = GBMRegressor(
        learning_rate=0.05,
        max_depth=4,
        n_estimators=300,
        neutralization="per_round_gradient",
        factor_neutralization_lambda=1e-6,
        seed=7,
        deterministic=True,
    )
    neutral.fit(X_train, y_train, factor_exposures=F_train)
    neutral_rmse = rmse(y_test, neutral.predict(X_test))

    # diagnostics_per_round_ surfaces the per-round effectiveness:
    # how much of the gradient norm was projected away.  ~0 means no
    # factor signal; ~1 means the gradients were almost entirely in
    # the factor subspace.
    effectiveness = [
        d["neutralization_effectiveness"] for d in neutral.diagnostics_per_round_
    ]

    print(f"plain test RMSE:           {plain_rmse:.4f}")
    print(f"neutralized test RMSE:     {neutral_rmse:.4f}")
    print(
        "neutralization effectiveness (mean / max / final):  "
        f"{np.mean(effectiveness):.3f} / {max(effectiveness):.3f} / "
        f"{effectiveness[-1]:.3f}"
    )
    print(
        "Note: the neutralized model's RMSE is *expected* to be higher "
        "on this synthetic target because the factors actually drive most "
        "of the variance.  The point is the model is factor-orthogonal."
    )


if __name__ == "__main__":
    main()
