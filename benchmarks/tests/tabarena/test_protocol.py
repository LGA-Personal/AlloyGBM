import json

import pytest
from tabarena.models.alloygbm.hpo import gen_alloygbm, sample_alloy_configs

from benchmarks.tabarena.protocol import RunSpec, resolve_run


def test_search_space_stability_and_public_default():
    assert gen_alloygbm.manual_configs == [{}]
    a = sample_alloy_configs(5)
    assert a == sample_alloy_configs(25)[:5]
    assert len(a) == 5
    assert all(
        c["n_estimators"] == 10000 and c["early_stopping_rounds"] == 50 for c in a
    )
    assert all(c["training_policy"] == "manual" for c in a)
    json.dumps(a)


def test_official_smoke_grid_and_default_passthrough():
    run = resolve_run(RunSpec())
    assert len(run.jobs) == 27  # Four boosters plus the default RF calibrator.
    assert len(run.tasks) == 3
    assert run.context.validation_protocol.key() == "8x1"
    assert run.context.enforces_validation_protocol
    for job in run.jobs:
        assert job.experiment is not None


def test_unknown_profile_and_invalid_resources():
    with pytest.raises(ValueError):
        RunSpec(profile="my-custom-split")
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            RunSpec(num_cpus=invalid)


def test_job_roundtrip_preserves_upstream_experiment_and_protocol():
    from tabarena.benchmark.experiment.job import Job

    run = resolve_run(RunSpec(models=("AlloyGBM",)))
    original = run.jobs[0]
    restored = Job.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.to_dict() == original.to_dict()
    assert restored.task.as_triple() == original.task.as_triple()


def test_official_lite_scope_and_author_default_identity():
    from tabarena.models import get_model_registry

    defaults = resolve_run(RunSpec(profile="lite-defaults"))
    tuned = resolve_run(RunSpec(profile="lite-hpo"))
    assert len(defaults.tasks) == len(tuned.tasks) == 51
    assert len(defaults.jobs) == 255
    assert len(tuned.jobs) == 5355
    assert sum("RandomForest" in j.experiment.name for j in tuned.jobs) == 51
    assert RunSpec().execution == "upstream"
    metadata = get_model_registry()["AlloyGBM"].method_metadata
    default = next(e for e in defaults.experiments if e.name == metadata.config_default)
    params = default.to_yaml_dict()["model_hyperparameters"]
    assert set(params) == {"ag_args", "ag_args_ensemble"}
    assert metadata.model_key == "ALLOYGBM"
