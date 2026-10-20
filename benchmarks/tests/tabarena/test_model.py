import numpy as np
import pandas as pd
import pytest
from tabarena.models.alloygbm.model import AlloyGBMModel


@pytest.mark.parametrize("problem_type", ["regression", "binary", "multiclass"])
def test_registered_model_fit_save_load(tmp_path, problem_type):
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    x["category"] = pd.Series(np.arange(100) % 4).astype("category")
    x.loc[0, "a"] = np.nan
    y = (
        pd.Series(x.b.fillna(0))
        if problem_type == "regression"
        else pd.Series(np.arange(100) % (2 if problem_type == "binary" else 3))
    )
    model = AlloyGBMModel(
        path=str(tmp_path / "model"),
        name="test",
        problem_type=problem_type,
        eval_metric="root_mean_squared_error"
        if problem_type == "regression"
        else "log_loss",
        hyperparameters={"n_estimators": 3, "learning_rate": 0.1},
    )
    model.fit(
        X=x[:80],
        y=y[:80],
        X_val=x[80:],
        y_val=y[80:],
        num_cpus=1,
        num_gpus=0,
        time_limit=30,
    )
    p = model.predict_proba(x[80:])
    assert p.shape == ((20, 3) if problem_type == "multiclass" else (20,))
    assert np.isfinite(p).all()
    model.save()
    restored = AlloyGBMModel.load(model.path)
    np.testing.assert_array_equal(p, restored.predict_proba(x[80:]))
    assert model.model.n_jobs == 1
    assert model.get_info()["alloygbm"]["rounds"] == model.model.n_estimators_


def test_auto_default_never_splits_its_own_holdout(tmp_path):
    x = pd.DataFrame({"a": np.arange(50, dtype=float)})
    m = AlloyGBMModel(
        path=str(tmp_path), name="no-validation", problem_type="regression"
    )
    with pytest.raises(ValueError, match="validation"):
        m.fit(X=x, y=x.a, num_cpus=1)


def test_sampled_config_warmup_has_no_hidden_validation_split():
    from tabarena.models.alloygbm.hpo import sample_alloy_configs
    from tabarena.models.warmup import warmup_model_cls

    report = warmup_model_cls(
        AlloyGBMModel,
        problem_type="regression",
        num_cpus=1,
        num_gpus=0,
        hyperparameters=sample_alloy_configs(1)[0],
    )
    assert not report.failed_steps
    assert report.dummy_fit["ran"]


def test_native_timeout_becomes_upstream_time_limit(tmp_path, monkeypatch):
    from alloygbm import GBMRegressor
    from autogluon.core.utils.exceptions import TimeLimitExceeded

    def exhausted(*args, **kwargs):
        raise TimeoutError("no completed round")

    monkeypatch.setattr(GBMRegressor, "fit", exhausted)
    x = pd.DataFrame({"a": np.arange(50, dtype=float)})
    m = AlloyGBMModel(
        path=str(tmp_path),
        name="timeout",
        problem_type="regression",
        hyperparameters={"n_estimators": 3},
    )
    with pytest.raises(TimeLimitExceeded):
        m.fit(
            X=x[:40],
            y=x.a[:40],
            X_val=x[40:],
            y_val=x.a[40:],
            num_cpus=1,
            num_gpus=0,
            time_limit=30,
        )
    assert m.model is None


@pytest.mark.parametrize(
    "problem_type,metric",
    [
        ("binary", "roc_auc"),
        ("binary", "log_loss"),
        ("regression", "root_mean_squared_error"),
    ],
)
def test_stopping_metric_matches_upstream_on_raw_margins(problem_type, metric):
    from autogluon.core.metrics import get_metric
    from scipy.special import expit
    from tabarena.models.alloygbm.model import stopping_callback

    scorer = get_metric(metric, problem_type=problem_type)
    callback = stopping_callback(problem_type, scorer)
    y = np.array([0, 1, 1, 0])
    raw = np.array([-3.0, 0.2, 2.0, 0.5])
    prediction = expit(raw) if problem_type == "binary" else raw
    name, error, higher = callback(y, raw)
    assert name == scorer.name
    assert error == pytest.approx(scorer.error(y, prediction))
    assert not higher
    assert np.isfinite(callback(np.zeros(2), np.zeros(2))[1])


def test_multiclass_uses_native_log_loss_and_rejects_other_metrics():
    from autogluon.core.metrics import get_metric
    from tabarena.models.alloygbm.model import stopping_callback

    assert stopping_callback("multiclass", get_metric("log_loss")) is None
    with pytest.raises(ValueError, match="stopping metric"):
        stopping_callback("multiclass", get_metric("accuracy"))


def test_cooperative_overrun_retains_completed_model(tmp_path, monkeypatch):
    import tabarena.models.alloygbm.model as module
    from alloygbm import GBMRegressor

    clock = [0.0]
    fit = GBMRegressor.fit

    def finish(*args, **kwargs):
        result = fit(*args, **kwargs)
        clock[0] = 31.0  # A completed native round may cross the soft deadline.
        return result

    monkeypatch.setattr(GBMRegressor, "fit", finish)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    x = pd.DataFrame({"a": np.arange(50, dtype=float)})
    m = AlloyGBMModel(
        path=str(tmp_path),
        name="overrun",
        problem_type="regression",
        hyperparameters={"n_estimators": 3},
    )
    m.fit(
        X=x[:40],
        y=x.a[:40],
        X_val=x[40:],
        y_val=x.a[40:],
        num_cpus=1,
        num_gpus=0,
        time_limit=30,
    )
    assert m.model.n_estimators_ > 0
    assert np.isfinite(m.predict_proba(x[40:])).all()


def test_binary_fit_early_stops_on_task_auc(tmp_path):
    x = pd.DataFrame({"a": np.arange(80, dtype=float)})
    y = pd.Series(np.arange(80) % 2)
    m = AlloyGBMModel(
        path=str(tmp_path),
        name="auc",
        problem_type="binary",
        eval_metric="roc_auc",
        hyperparameters={
            "n_estimators": 20,
            "early_stopping_rounds": 2,
            "ag_args_fit": {"stopping_metric": "roc_auc"},
        },
    )
    m.fit(X=x[:60], y=y[:60], X_val=x[60:], y_val=y[60:], num_cpus=1, num_gpus=0)
    assert "roc_auc" in m.model.evals_result_["validation"]


def test_worker_serialization_imports_only_registered_payload(tmp_path):
    import subprocess
    import sys

    import cloudpickle

    target = tmp_path / "class.pkl"
    target.write_bytes(cloudpickle.dumps(AlloyGBMModel))
    subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            (
                "import cloudpickle,sys; from tabarena.models.alloygbm.model import AlloyGBMModel; "
                'assert cloudpickle.load(open(sys.argv[1],"rb")) is AlloyGBMModel; '
                'assert "benchmarks" not in sys.modules'
            ),
            str(target),
        ],
        cwd=tmp_path,
        check=True,
    )


def test_unknown_and_missing_categories_survive_serialization(tmp_path):
    train = pd.DataFrame(
        {
            "category": pd.Categorical(["a", "b"] * 30),
            "number": np.arange(60, dtype=float),
        }
    )
    valid = pd.DataFrame(
        {
            "category": pd.Categorical(["unseen", None, "a"]),
            "number": [np.nan, 15.0, 42.0],
        }
    )
    m = AlloyGBMModel(
        path=str(tmp_path / "model"),
        name="unknown-category",
        problem_type="regression",
        hyperparameters={"n_estimators": 3},
    )
    m.fit(
        X=train,
        y=train.number,
        X_val=valid,
        y_val=pd.Series([0.0, 15.0, 42.0]),
        num_cpus=1,
        num_gpus=0,
    )
    before = m.predict_proba(valid)
    assert np.isfinite(before).all()
    m.save()
    restored = AlloyGBMModel.load(m.path)
    np.testing.assert_array_equal(before, restored.predict_proba(valid))


def test_missing_global_class_fails_instead_of_silently_reordering(tmp_path):
    train = pd.DataFrame({"number": np.arange(60, dtype=float)})
    m = AlloyGBMModel(
        path=str(tmp_path),
        name="missing-class",
        problem_type="multiclass",
        eval_metric="log_loss",
        hyperparameters={"n_estimators": 3},
    )
    with pytest.raises(ValueError, match="missing an expected target class"):
        m.fit(
            X=train,
            y=pd.Series(np.arange(60) % 2),
            X_val=train[:6],
            y_val=pd.Series(np.arange(6) % 2),
            num_classes=3,
            num_cpus=1,
            num_gpus=0,
        )
