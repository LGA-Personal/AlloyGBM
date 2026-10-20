"""Read upstream caches and fail closed on missing or nonstandard fit evidence."""

from pathlib import Path


def result_cacher(run_dir, job, metadata):
    from tabarena.benchmark.experiment.experiment_runner_api import (
        _build_results_cacher,
        task_cache_key_from_task_id_str,
    )
    from tabarena.utils.cache import CacheFunctionPickle

    # Use the pinned writer's own constructor, including compression and task IDs.
    return _build_results_cacher(
        cache_cls=CacheFunctionPickle,
        cache_cls_kwargs=None,
        base_cache_path=str(run_dir),
        method_name=job.experiment.name,
        cache_task_key=task_cache_key_from_task_id_str(metadata.task_id_str),
        fold=job.task.fold,
        repeat=job.task.repeat,
    )


def validate_result(result):
    import numpy as np

    protocol = result.get("validation_protocol", {})
    expected = {
        "key": "8x1",
        "num_bag_folds_resolved": 8,
        "num_bag_sets_resolved": 1,
        "num_bag_folds_fitted": 8,
        "num_bag_sets_fitted": 1,
        "num_child_models": 8,
        "custom_splits": False,
    }
    if any(protocol.get(k) != v for k, v in expected.items()) or not protocol.get(
        "protocol", {}
    ).get("enforced"):
        raise ValueError("result lacks enforced, fully fitted TabArena 8x1 evidence")
    artifacts = result.get("simulation_artifacts")
    if not artifacts:
        raise ValueError("result lacks validation/test predictions")
    val_indices, test_indices = artifacts["y_val_idx"], artifacts["y_test_idx"]
    if set(val_indices) & set(test_indices):
        raise ValueError("outer test rows overlap training/OOF rows")
    for split in ("val", "test"):
        targets = artifacts["y_" + split]
        predictions = artifacts["pred_proba_dict_" + split]
        if not predictions or any(
            len(values) != len(targets) or not np.isfinite(values).all()
            for values in predictions.values()
        ):
            raise ValueError(
                "missing, nonfinite or misaligned " + split + " predictions"
            )
    if not np.isfinite(result["metric_error"]):
        raise ValueError("nonfinite test metric")
    return expected


def completed_result_paths(run_dir, resolved):
    from benchmarks.tabarena.provenance import file_hash, fingerprint
    from benchmarks.tabarena.run import load_latest

    latest = load_latest(run_dir)
    paths = []
    for job, metadata in zip(
        resolved.jobs, resolved.context.metadata_for_jobs(resolved.jobs)
    ):
        if latest.get(fingerprint(job.to_dict()), {}).get("status") != "completed":
            continue
        cacher = result_cacher(run_dir, job, metadata)
        if not cacher.exists:
            raise ValueError("completed job has lost its upstream result cache")
        expected_hash = latest[fingerprint(job.to_dict())].get("result_sha256")
        if file_hash(cacher.cache_file) != expected_hash:
            raise ValueError("completed result checksum differs")
        validate_result(cacher.load_cache())
        paths.append(Path(cacher.cache_file))
    if not paths:
        raise ValueError("no completed, conformant results to report; see jobs.jsonl")
    return paths
