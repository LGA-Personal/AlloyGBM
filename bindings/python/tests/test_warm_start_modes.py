"""Warm-start coverage matrix for the v0.7.1 contract.

This file documents (and verifies) what `init_model=` / `warm_start=True`
supports across every leaf model / training mode combination introduced in
v0.6 and v0.7:

* ``leaf_model="linear"`` — piecewise-linear leaves
* ``leaf_solver="dro"`` — distributionally-robust scalar leaves
* ``training_mode="morph"`` — MorphBoost adaptive criterion
* ``neutralization=*`` — factor-neutral boosting

For each mode the test fits a base model, resumes it (or a fresh estimator
seeded from it via ``init_model``), and asserts that:

* the resumed fit runs to completion without raising;
* the resumed model adds the requested rounds on top of the base.

Where the mode has an end-to-end equivalence contract ("fit N then M more is
the same as fit N+M from scratch") this is documented inline.  MorphBoost's
EMA is *not* restored across warm-starts in v0.7.1, so resumed training
restarts the EMA cold — equivalence to a fresh fit is approximate and is not
enforced numerically here.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRegressor


def _data(seed: int = 0, n: int = 120):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3)).astype("float32")
    y = (X[:, 0] * 1.2 - X[:, 1] * 0.4).astype("float32")
    return X, y


class TestWarmStartPLLeaves:
    def test_resume_adds_rounds(self) -> None:
        X, y = _data(seed=11)
        m1 = GBMRegressor(n_estimators=3, leaf_model="linear", max_depth=2).fit(X, y)
        m2 = GBMRegressor(n_estimators=3, leaf_model="linear", max_depth=2).fit(
            X, y, init_model=m1
        )
        assert m1.rounds_completed_ == 3
        assert m2.rounds_completed_ == 3
        # Predictions must still be finite and differ from the base — the
        # resumed model added trees so it should produce a different result
        # on at least some rows.
        p1 = np.asarray(m1.predict(X[:5]))
        p2 = np.asarray(m2.predict(X[:5]))
        assert np.all(np.isfinite(p2))
        assert not np.allclose(p1, p2)


class TestWarmStartDROLeaves:
    def test_resume_adds_rounds(self) -> None:
        X, y = _data(seed=13)
        m1 = GBMRegressor(n_estimators=3, leaf_solver="dro", max_depth=2).fit(X, y)
        m2 = GBMRegressor(n_estimators=3, leaf_solver="dro", max_depth=2).fit(
            X, y, init_model=m1
        )
        assert m2.rounds_completed_ == 3
        p2 = np.asarray(m2.predict(X[:5]))
        assert np.all(np.isfinite(p2))


class TestWarmStartMorphBoost:
    def test_resume_runs_without_error(self) -> None:
        # MorphBoost EMA is not restored across warm-starts in v0.7.1; the
        # resumed model still runs, but its EMA-driven gain shaping starts
        # cold for the new rounds.  See `docs/limitations.md` for the caveat.
        X, y = _data(seed=17)
        m1 = GBMRegressor(n_estimators=3, training_mode="morph", max_depth=2).fit(X, y)
        m2 = GBMRegressor(n_estimators=3, training_mode="morph", max_depth=2).fit(
            X, y, init_model=m1
        )
        assert m2.rounds_completed_ == 3
        p2 = np.asarray(m2.predict(X[:5]))
        assert np.all(np.isfinite(p2))


class TestWarmStartFactorNeutralization:
    @pytest.mark.parametrize(
        "kind",
        ["pre_target", "per_round_gradient", "split_penalty"],
    )
    def test_resume_succeeds_when_exposures_supplied(self, kind: str) -> None:
        X, y = _data(seed=19)
        # Neutralize against features that are unrelated to the target so
        # the resumed fit still has signal to learn from.
        exposures = X[:, 2:3].copy()
        kwargs = dict(
            n_estimators=3,
            neutralization=kind,
            factor_neutralization_lambda=1e-4,
            max_depth=2,
        )
        if kind == "split_penalty":
            kwargs["factor_penalty"] = 1e-2
        m1 = GBMRegressor(**kwargs).fit(X, y, factor_exposures=exposures)
        m2 = GBMRegressor(**kwargs).fit(X, y, init_model=m1, factor_exposures=exposures)
        assert m1.rounds_completed_ > 0
        assert m2.rounds_completed_ > 0

    def test_resume_without_exposures_is_rejected(self) -> None:
        X, y = _data(seed=23)
        exposures = X[:, 2:3].copy()
        m1 = GBMRegressor(
            n_estimators=3,
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
        ).fit(X, y, factor_exposures=exposures)
        m2 = GBMRegressor(
            n_estimators=3,
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
        )
        with pytest.raises(ValueError, match="requires factor_exposures"):
            m2.fit(X, y, init_model=m1)

    @pytest.mark.parametrize(
        "init_mode,new_mode",
        [
            # Same exposures + a different mode silently switched the
            # boosting path before v0.7.1; verify every cross-mode pair
            # is now rejected up front.
            ("per_round_gradient", "pre_target"),
            ("per_round_gradient", "split_penalty"),
            ("split_penalty", "per_round_gradient"),
            ("pre_target", "per_round_gradient"),
            ("none", "per_round_gradient"),
            ("per_round_gradient", "none"),
        ],
    )
    def test_resume_with_mismatched_mode_is_rejected(
        self, init_mode: str, new_mode: str
    ) -> None:
        X, y = _data(seed=37)
        exposures = X[:, 2:3].copy()
        init_kwargs: dict[str, object] = {"n_estimators": 2}
        if init_mode != "none":
            init_kwargs["neutralization"] = init_mode
            init_kwargs["factor_neutralization_lambda"] = 1e-4
            if init_mode == "split_penalty":
                init_kwargs["factor_penalty"] = 1e-2
        m1_kwargs = dict(init_kwargs)
        fit_exposures = exposures if init_mode != "none" else None
        m1 = GBMRegressor(**m1_kwargs).fit(
            X, y, factor_exposures=fit_exposures
        )
        new_kwargs: dict[str, object] = {"n_estimators": 2}
        if new_mode != "none":
            new_kwargs["neutralization"] = new_mode
            new_kwargs["factor_neutralization_lambda"] = 1e-4
            if new_mode == "split_penalty":
                new_kwargs["factor_penalty"] = 1e-2
        m2 = GBMRegressor(**new_kwargs)
        with pytest.raises(ValueError, match="does not match"):
            m2.fit(
                X,
                y,
                init_model=m1,
                factor_exposures=exposures if new_mode != "none" else None,
            )

    def test_resume_with_mismatched_lambda_is_rejected(self) -> None:
        X, y = _data(seed=41)
        exposures = X[:, 2:3].copy()
        m1 = GBMRegressor(
            n_estimators=2,
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
        ).fit(X, y, factor_exposures=exposures)
        m2 = GBMRegressor(
            n_estimators=2,
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-3,  # different λ
        )
        with pytest.raises(ValueError, match="factor_neutralization_lambda"):
            m2.fit(X, y, init_model=m1, factor_exposures=exposures)

    def test_resume_with_mismatched_split_penalty_is_rejected(self) -> None:
        X, y = _data(seed=43)
        exposures = X[:, 2:3].copy()
        m1 = GBMRegressor(
            n_estimators=2,
            neutralization="split_penalty",
            factor_neutralization_lambda=1e-4,
            factor_penalty=1e-2,
        ).fit(X, y, factor_exposures=exposures)
        m2 = GBMRegressor(
            n_estimators=2,
            neutralization="split_penalty",
            factor_neutralization_lambda=1e-4,
            factor_penalty=5e-2,  # different penalty
        )
        with pytest.raises(ValueError, match="factor_penalty"):
            m2.fit(X, y, init_model=m1, factor_exposures=exposures)

    def test_resume_with_warm_start_flag(self) -> None:
        # The other entry point: warm_start=True reuses the estimator's own
        # state across successive fit calls.  Same exposures contract.
        X, y = _data(seed=29)
        exposures = X[:, 2:3].copy()
        m = GBMRegressor(
            n_estimators=2,
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
            warm_start=True,
        )
        m.fit(X, y, factor_exposures=exposures)
        first_rounds = m.rounds_completed_
        m.n_estimators = 3
        m.fit(X, y, factor_exposures=exposures)
        assert m.rounds_completed_ >= first_rounds


class TestWarmStartClassifierNeutralization:
    def test_per_round_neutralization_warm_start(self) -> None:
        # Classifier inherits warm-start behavior from GBMRegressor; verify
        # the neutralization branch is exercised end-to-end.
        X, y_reg = _data(seed=31)
        y = (y_reg > 0).astype("int32")
        exposures = X[:, 2:3].copy()
        m1 = GBMClassifier(
            n_estimators=3,
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
        ).fit(X, y, factor_exposures=exposures)
        m2 = GBMClassifier(
            n_estimators=3,
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
        ).fit(X, y, init_model=m1, factor_exposures=exposures)
        assert m1.rounds_completed_ > 0
        assert m2.rounds_completed_ > 0
