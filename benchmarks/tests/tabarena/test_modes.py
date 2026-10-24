import json

import numpy as np
import pytest


def test_modes_search_preserves_prefix_and_covers_only_legal_combinations():
    from tabarena.models.alloygbm.hpo import (
        sample_alloy_configs,
        sample_alloy_modes_configs,
    )

    configs = sample_alloy_modes_configs(200)
    assert configs[:25] == sample_alloy_configs(25)
    assert sample_alloy_modes_configs(31) == configs[:31]
    assert sample_alloy_modes_configs(200) == configs
    assert len(configs) == 200
    modes = {
        (c["training_mode"], c["leaf_model"], c.get("leaf_solver", "standard"))
        for c in configs[25:]
    }
    assert modes == {
        ("manual", "constant", "standard"),
        ("morph", "constant", "standard"),
        ("manual", "constant", "dro"),
        ("morph", "constant", "dro"),
        ("manual", "linear", "standard"),
        ("morph", "linear", "standard"),
    }
    assert all(
        c.get("leaf_solver") != "dro" or c["leaf_model"] == "constant" for c in configs
    )
    json.dumps(configs, allow_nan=False)


@pytest.mark.parametrize("problem", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("training_mode", ["manual", "morph"])
@pytest.mark.parametrize(
    "leaf_model,leaf_solver",
    [("constant", "standard"), ("constant", "dro"), ("linear", "standard")],
)
def test_modes_fit_actual_adapter(
    problem, training_mode, leaf_model, leaf_solver, tmp_path
):
    import pandas as pd
    from tabarena.models.alloygbm.model import AlloyGBMModel

    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(96, 4)).astype("float32"))
    y = (
        rng.normal(size=96)
        if problem == "regression"
        else np.arange(96) % (2 if problem == "binary" else 3)
    )
    model = AlloyGBMModel(
        path=str(tmp_path),
        name="mode",
        problem_type=problem,
        eval_metric="rmse" if problem == "regression" else "log_loss",
        hyperparameters={
            "n_estimators": 4,
            "early_stopping_rounds": None,
            "max_depth": 2,
            "training_policy": "manual",
            "training_mode": training_mode,
            "leaf_model": leaf_model,
            "leaf_solver": leaf_solver,
            "lambda_l2": 1.0,
        },
    )
    model.fit(X=X.iloc[:72], y=y[:72], X_val=X.iloc[72:], y_val=y[72:], num_cpus=1)
    predictions = model.predict_proba(X.iloc[72:])
    assert len(predictions) == 24 and np.isfinite(predictions).all()
    info = model.get_info()["alloygbm"]["parameters"]
    assert (info["training_mode"], info["leaf_model"], info["leaf_solver"]) == (
        training_mode,
        leaf_model,
        leaf_solver,
    )


def test_versioned_search_uses_official_full_grid_without_mutating_v1():
    from benchmarks.tabarena.protocol import RunSpec, resolve_run

    extended = resolve_run(
        RunSpec(profile="full", num_cpus=16, search_space="modes-v2")
    )
    assert len(extended.jobs) == 164016 and len(extended.tasks) == 816
    assert len({j.task.dataset for j in extended.jobs}) == 51
    assert extended.context.validation_protocol.key() == "8x1"
    original = resolve_run(RunSpec(profile="lite-hpo", num_cpus=16))
    full_keys = {json.dumps(j.to_dict(), sort_keys=True) for j in extended.jobs}
    assert all(
        json.dumps(j.to_dict(), sort_keys=True) in full_keys for j in original.jobs
    )
    with pytest.raises(ValueError):
        RunSpec(search_space="unknown")


def test_lite_200_matches_published_candidate_budget_and_official_lite_scope():
    from benchmarks.tabarena.protocol import RunSpec, resolve_run

    run = resolve_run(RunSpec(profile="lite-200", num_cpus=16, search_space="modes-v2"))
    assert len(run.jobs) == 10251 and len(run.tasks) == 51
    assert len(run.experiments) == 201
    assert {j.task.as_triple()[1:] for j in run.jobs} == {(0, 0)}
