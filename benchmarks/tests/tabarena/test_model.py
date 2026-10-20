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
