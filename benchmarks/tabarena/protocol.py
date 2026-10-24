"""Choose run scope; all statistical protocol comes from pinned TabArena."""

from __future__ import annotations

from dataclasses import dataclass

SMOKE = ("blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal")
MODELS = ("AlloyGBM", "LightGBM", "XGBoost", "CatBoost", "RandomForest")
DEFAULT_MODELS = ("AlloyGBM",)
COUNTS = {"smoke": 1, "lite-defaults": 0, "lite-hpo": 25, "full": "all"}


@dataclass(frozen=True)
class RunSpec:
    profile: str = "smoke"
    num_cpus: int | None = None
    dataset_names: tuple[str, ...] | None = None
    models: tuple[str, ...] = DEFAULT_MODELS
    execution: str = "upstream"

    def __post_init__(self):
        if self.execution not in ("local", "upstream"):
            raise ValueError("unknown execution mode")
        if self.profile not in COUNTS:
            raise ValueError("unknown benchmark profile")
        if self.num_cpus is not None and (
            isinstance(self.num_cpus, bool)
            or not isinstance(self.num_cpus, int)
            or self.num_cpus < 1
        ):
            raise ValueError("num_cpus must be positive or None")
        if not self.models or set(self.models) - set(MODELS):
            raise ValueError("unknown or empty model list")
        if len(set(self.models)) != len(self.models):
            raise ValueError("duplicate model")


@dataclass
class ResolvedRun:
    spec: RunSpec
    context: object
    experiments: list
    jobs: list
    tasks: list[dict]
    configs: list[dict]


def resolve_run(spec: RunSpec) -> ResolvedRun:
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.contexts import TabArenaContext

    # No custom splitting, preprocessing, fold-fitting, stopping, or scoring settings.
    context = TabArenaContext(backend="native" if spec.execution == "local" else "ray")
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[
            (name, 0 if name == "RandomForest" else COUNTS[spec.profile])
            for name in spec.models
        ]
    )
    experiments = bundle.build_experiments(num_cpus=spec.num_cpus, num_gpus=0)
    names = spec.dataset_names or (SMOKE if spec.profile == "smoke" else None)
    jobs = context.build_jobs(
        experiments,
        subset=None if spec.profile == "full" else "lite",
        dataset_names=names,
    )
    metadata = context.metadata_for_jobs(jobs)
    tasks = {job.task.as_triple(): meta.to_dict() for job, meta in zip(jobs, metadata)}
    return ResolvedRun(
        spec,
        context,
        experiments,
        jobs,
        list(tasks.values()),
        [exp.to_yaml_dict() for exp in experiments],
    )
