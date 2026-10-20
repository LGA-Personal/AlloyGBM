"""Regenerate standard TabArena reports from trusted local upstream caches."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path


def validate_local_coverage(manifest, latest):
    if "RandomForest" in manifest["spec"]["models"]:
        rf = [
            r
            for r in latest.values()
            if r.get("experiment", "").startswith("RandomForest_")
        ]
        if len(rf) != len(manifest["tasks"]) or any(
            r["status"] != "completed" for r in rf
        ):
            raise ValueError(
                "local RF calibration must complete every scoped task before ranking; see coverage.csv"
            )
        return True
    if len(latest) != manifest["expected_jobs"] or any(
        r["status"] != "completed" for r in latest.values()
    ):
        raise ValueError(
            "local ranking without RF requires complete coverage; see coverage.csv"
        )
    return False


def comparison_context(methods, backend, *, published=False, calibrated=False):
    from tabarena.contexts import TabArenaContext

    if published:
        return TabArenaContext(
            extra_methods=[m for m in methods if m.method == "[Local] AlloyGBM"],
            backend=backend,
            only_valid_tasks=True,
        )
    calibration = "[Local] RF (default)" if calibrated else None
    return TabArenaContext(
        methods=[],
        extra_methods=methods,
        backend=backend,
        fillna_method=calibration,
        calibration_method=calibration,
        only_valid_tasks=True,
    )


def write_coverage(run_dir):
    import pandas as pd

    from benchmarks.tabarena.run import load_latest

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    rows = list(load_latest(run_dir).values())
    rows.extend(
        {"key": None, "status": "pending"}
        for _ in range(max(0, manifest["expected_jobs"] - len(rows)))
    )
    output = run_dir / "coverage.csv"
    pd.DataFrame(
        rows,
        columns=[
            "key",
            "status",
            "experiment",
            "task",
            "elapsed_seconds",
            "error",
            "cache_hit",
        ],
    ).to_csv(output, index=False)
    return output


def write_selection_weights(method, run_dir):
    """Replay upstream selection to expose weights omitted by its summary frames."""
    import pandas as pd

    repo = method.method_metadata.load_processed(download=False)
    for variant in ("tuned", "tuned_ensemble"):
        expected = method.hpo_results[method.hpo_results.method_subtype.eq(variant)]
        matrices = []
        for row in expected.itertuples():
            settings = row.method_metadata
            configs = repo.configs(
                tasks=[(row.dataset, row.fold)], config_types=[settings["config_type"]]
            )
            errors, weights = repo.evaluate_ensembles(
                datasets=[row.dataset],
                folds=[row.fold],
                configs=configs,
                ensemble_size=settings["n_iterations"],
                fit_order=settings["fit_order"],
                seed=0,
                time_limit=settings["time_limit"],
                backend="native",
            )
            # Diagnostics must reproduce the standard scores; they never replace them.
            for metric in ("metric_error", "metric_error_val"):
                pd.testing.assert_series_equal(
                    pd.Series([getattr(row, metric)]),
                    errors[metric].reset_index(drop=True),
                    check_names=False,
                    check_dtype=False,
                )
            matrices.append(weights.reset_index())
        if matrices:
            pd.concat(matrices, ignore_index=True).to_csv(
                run_dir
                / (method.method_metadata.method + "-" + variant + "-weights.csv"),
                index=False,
            )


def process_standard_results(run_dir):
    from benchmarks.tabarena.artifacts import completed_result_paths
    from benchmarks.tabarena.protocol import RunSpec, resolve_run
    from benchmarks.tabarena.run import ROOT
    from tabarena.caching import CacheConfig
    from tabarena.end_to_end import EndToEnd, EndToEndResults
    from tabarena.models import get_model_registry

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    spec = RunSpec(**manifest["spec"])
    resolved = resolve_run(spec)
    CacheConfig.from_root(ROOT / ".cache/tabarena/cache").apply()
    paths = completed_result_paths(run_dir, resolved)
    collected = []
    for name in spec.models:
        info = get_model_registry()[name]
        prefix = info.model_cls.ag_name + "_"
        method_paths = [p for p in paths if p.parts[-4].startswith(prefix)]
        if not method_paths:
            continue  # Coverage still retains every failed/pending entrant.
        defaults = [
            e.name
            for e in resolved.experiments
            if e.name.startswith(prefix) and "_c1_" in e.name
        ]
        if len(defaults) != 1:
            raise ValueError("expected exactly one declared default: " + name)
        metadata = replace(
            info.method_metadata,
            suite="local-" + manifest["fingerprint"],
            cache_type="local",
            cache_kwargs={},
            cache_root=ROOT / ".cache/tabarena/cache/tabarena",
            artifact_dir=None,
            config_default=defaults[0],
            verified=False,
        )
        # Per-method metadata preserves declared model identity and default names;
        # the standard pipeline owns selection, ensembling and metric semantics.
        processed = EndToEnd.from_path_raw(
            run_dir / "data",
            method_metadata=metadata,
            task_metadata=resolved.context.task_metadata_collection.subset_to_jobs(
                resolved.jobs
            ),
            file_paths=method_paths,
            cache=True,
            cache_processed=True,
            backend="native" if spec.execution == "local" else "ray",
            num_cpus=spec.num_cpus or os.cpu_count() or 1,
        )
        collected.extend(processed.method_results_lst)
    results = EndToEndResults(method_results_lst=collected)
    for method in results.method_results_lst:
        if (
            method.method_metadata.method == "RandomForest"
            and method.hpo_results is not None
        ):
            method.hpo_results = method.hpo_results[
                method.hpo_results.method_subtype.eq("default")
            ].copy()
        method.model_results.to_csv(
            run_dir / (method.method_metadata.method + "-configs.csv"), index=False
        )
        if method.hpo_results is not None:
            expected = (
                {"default"}
                if method.method_metadata.method == "RandomForest"
                else {"default", "tuned", "tuned_ensemble"}
            )
            if set(method.hpo_results.method_subtype) != expected:
                raise ValueError(
                    "an entire variant has no successful result; refusing incomplete ranking"
                )
            write_selection_weights(method, run_dir)
            method.hpo_results.to_csv(
                run_dir / (method.method_metadata.method + "-variants.csv"), index=False
            )
        method.method_metadata.to_yaml()
    return results.to_method_metadata_lst(new_result_prefix="[Local] ")


def _write_report(run_dir, *, published_context=False):
    import pandas as pd

    from benchmarks.tabarena.protocol import COUNTS
    from benchmarks.tabarena.provenance import file_hash
    from benchmarks.tabarena.run import load_latest

    run_dir = Path(run_dir).resolve()
    write_coverage(run_dir)
    methods = process_standard_results(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    backend = "native" if manifest["spec"]["execution"] == "local" else "ray"
    calibrated = validate_local_coverage(manifest, load_latest(run_dir))
    missing = {"[Local] " + name for name in manifest["spec"]["models"]} - {
        m.method for m in methods
    }
    if missing:
        raise ValueError(
            "no successful configurations for entrants; refusing incomplete ranking: "
            + str(sorted(missing))
        )
    context = comparison_context(methods, backend, calibrated=calibrated)
    output = run_dir / "report-local"
    leaderboard = context.compare(output_dir=output)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    coverage = pd.read_csv(run_dir / "coverage.csv")
    summary = {
        "run_fingerprint": manifest["fingerprint"],
        "report_code_sha256": file_hash(Path(__file__)),
        "ledger_sha256": file_hash(run_dir / "jobs.jsonl"),
        "profile": manifest["spec"]["profile"],
        "sampled_candidates_per_booster": COUNTS[manifest["spec"]["profile"]],
        "expected_jobs": manifest["expected_jobs"],
        "outcomes": coverage.status.value_counts().to_dict(),
        "partial": not coverage.status.eq("completed").all(),
        "protocol": "TabArena-v0.1 8x1",
        "execution": manifest["spec"]["execution"],
        "local_pool_calibration": context.calibration_method,
        "public_defaults": "current constructor defaults with supplied validation",
        "source_commit": manifest["provenance"]["source_commit"],
        "upstream_revision": manifest["upstream"],
        "official_submission_verified": False,
        "published_context": published_context,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "summary.md").write_text(
        "# Local standard TabArena results\n\n"
        + "Protocol: official TabArena-v0.1, 8 folds × 1 set. Local entrant pool; calibration/fallback recorded below.\n\n"
        + "Scope and missing outcomes must be read alongside ranks. These are not official leaderboard placements.\n\n"
        + "```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n"
    )
    if published_context:
        published = comparison_context(methods, backend, published=True)
        published.compare(output_dir=run_dir / "report-published")
        sources = [
            {
                "metadata": method.to_info_dict(),
                "artifact_sha256": {
                    str(path): file_hash(path)
                    for path in sorted(method.path_results.glob("*.parquet"))
                },
            }
            for method in published.method_metadata_collection.method_metadata_lst
        ]
        (run_dir / "report-published" / "sources.json").write_text(
            json.dumps(
                {
                    "upstream_revision": manifest["upstream"],
                    "methods": sources,
                    "run_fingerprint": manifest["fingerprint"],
                    "ledger_sha256": file_hash(run_dir / "jobs.jsonl"),
                    "report_code_sha256": file_hash(Path(__file__)),
                    "local_entrants": [
                        m.method for m in methods if m.method == "[Local] AlloyGBM"
                    ],
                    "sampled_candidates_per_booster": COUNTS[
                        manifest["spec"]["profile"]
                    ],
                    "candidate_budget_note": "See recorded local candidate count (additional to default); published baselines generally use 200. Full profile uses the pinned upstream all budget. See each method's source metadata.",
                },
                indent=2,
                default=str,
                allow_nan=False,
            )
            + "\n"
        )
        (run_dir / "report-published" / "SOURCE_NOTE.md").write_text(
            "Published results use their source hardware/library versions and candidate budgets; local results use this run manifest. "
            "The published Pareto/timing outputs are mixed-hardware context, not local speedup evidence. "
            "See report-local for the matched local entrant pool.\n"
            "Only AlloyGBM is added to the published pool. "
            f"Local sampled candidates per booster: {COUNTS[manifest['spec']['profile']]} (additional to default); published tuned baselines generally use 200.\n"
        )
    return output


def write_report(run_dir, *, published_context=False):
    from benchmarks.tabarena.run import exclusive_run

    run_dir = Path(run_dir).resolve()
    with exclusive_run(run_dir):
        return _write_report(run_dir, published_context=published_context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--published-context", action="store_true")
    args = parser.parse_args()
    print(write_report(args.run_dir, published_context=args.published_context))
