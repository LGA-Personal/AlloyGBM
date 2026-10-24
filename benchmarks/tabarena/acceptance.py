"""Audit complete Lite or full official scope, preserving failure explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.tabarena.provenance import file_hash, fingerprint
from benchmarks.tabarena.run import exclusive_run, load_latest


def validate_scope_and_outcomes(manifest, jobs, latest, explanations):
    spec = manifest["spec"]
    from benchmarks.tabarena.protocol import MODELS

    models = spec.get("models", [])
    full = spec["profile"] == "full"
    task_count = 816 if full else 51
    candidates = 201 if full or spec["profile"] == "lite-200" else 26
    if (
        spec["profile"] not in {"lite-hpo", "lite-200", "full"}
        or spec.get("dataset_names")
        or "AlloyGBM" not in models
        or set(models) - set(MODELS)
        or len(models) != len(set(models))
        or len(manifest["tasks"]) != task_count
        or len(jobs)
        != task_count * sum(1 if m == "RandomForest" else candidates for m in models)
    ):
        raise ValueError(
            "acceptance requires the complete 51-dataset AlloyGBM Lite HPO run (51 splits, default plus 25 samples) or Lite-200 (51 splits, default plus 200 samples) or full run (816 splits, default plus 200 samples); peers are optional"
        )
    from benchmarks.tabarena.resources import validate_acceptance_resources

    validate_acceptance_resources(spec, manifest["runtime"])
    keys = {fingerprint(job) for job in jobs}
    if set(latest) != keys or any(
        row["status"] not in {"completed", "failed"} for row in latest.values()
    ):
        raise ValueError(
            "every expected job must have a terminal outcome; unexpected keys are forbidden"
        )
    failures = {key for key, row in latest.items() if row["status"] == "failed"}
    if set(explanations) != failures or any(
        not isinstance(v, str) or not v.strip() for v in explanations.values()
    ):
        raise ValueError(
            "every failure needs an explicit, nonempty explanation (no extra keys)"
        )
    return failures


def validate_report_evidence(manifest, summary, sources, ledger_hash):
    if not summary.get("published_context"):
        raise ValueError(
            "acceptance requires --published-context with standard calibration/fallback"
        )
    if (
        sources.get("calibration_method") != "RF (default)"
        or sources.get("fillna_method") != "RF (default)"
    ):
        raise ValueError("acceptance requires published RF calibration/fallback")
    if sources.get("local_entrants") != ["[Local] AlloyGBM"]:
        raise ValueError(
            "published comparison must add only AlloyGBM to the upstream pool"
        )
    for record in (summary, sources):
        if (
            record.get("run_fingerprint") != manifest["fingerprint"]
            or record.get("ledger_sha256") != ledger_hash
        ):
            raise ValueError("regenerate both reports from the current terminal ledger")
    if sources.get("report_code_sha256") != summary.get("report_code_sha256"):
        raise ValueError(
            "local and published reports used different processor revisions"
        )


def audit_acceptance(run_dir, explanations_path=None):
    import pandas as pd

    from benchmarks.tabarena.artifacts import completed_result_paths
    from benchmarks.tabarena.protocol import RunSpec, resolve_run

    run_dir = Path(run_dir).resolve()
    with exclusive_run(run_dir):
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        jobs = json.loads((run_dir / "job_plan.json").read_text())
        explanations = (
            json.loads(Path(explanations_path).read_text()) if explanations_path else {}
        )
        failures = validate_scope_and_outcomes(
            manifest, jobs, load_latest(run_dir), explanations
        )
        resolved = resolve_run(RunSpec(**manifest["spec"]))
        if [job.to_dict() for job in resolved.jobs] != jobs:
            raise ValueError("recorded job plan differs from the pinned standard plan")
        paths = completed_result_paths(run_dir, resolved)
        report = run_dir / "report-published"
        summary = json.loads((report / "summary.json").read_text())
        sources = json.loads((run_dir / "report-published/sources.json").read_text())
        validate_report_evidence(
            manifest, summary, sources, file_hash(run_dir / "jobs.jsonl")
        )
        for method in manifest["spec"]["models"]:
            variants = pd.read_csv(run_dir / (method + "-variants.csv"))
            expected_variants = (
                {"default"}
                if method == "RandomForest"
                else {"default", "tuned", "tuned_ensemble"}
            )
            if set(variants.method_subtype) != expected_variants:
                raise ValueError("missing standard variants: " + method)
        result = {
            "run_fingerprint": manifest["fingerprint"],
            "expected_jobs": len(jobs),
            "completed_jobs": len(paths),
            "failed_jobs": len(failures),
            "profile": manifest["spec"]["profile"],
            "search_space": manifest["spec"].get("search_space", "v1"),
            "failure_explanations": explanations,
            "ledger_sha256": file_hash(run_dir / "jobs.jsonl"),
            "leaderboard_sha256": file_hash(report / "tabarena_leaderboard.csv"),
            "published_leaderboard_sha256": file_hash(
                run_dir / "report-published/tabarena_leaderboard.csv"
            ),
            "published_sources_sha256": file_hash(
                run_dir / "report-published/sources.json"
            ),
            "official_submission_verified": False,
            "note": "Mechanical evidence audit; explanations and benchmark claims still need author review.",
        }
        (run_dir / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--failure-explanations", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(audit_acceptance(args.run_dir, args.failure_explanations), indent=2)
    )
