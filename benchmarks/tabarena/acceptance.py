"""Audit full Lite evidence before merging the separate legacy-retirement PR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.tabarena.provenance import file_hash, fingerprint
from benchmarks.tabarena.run import exclusive_run, load_latest


def validate_scope_and_outcomes(manifest, jobs, latest, explanations):
    spec = manifest["spec"]
    if (
        spec["profile"] != "lite-hpo"
        or spec.get("dataset_names")
        or set(spec["models"]) != {"AlloyGBM", "LightGBM", "XGBoost", "CatBoost"}
        or len(manifest["tasks"]) != 51
        or len(jobs) != 5304
    ):
        raise ValueError(
            "acceptance requires the full four-method, 51-dataset Lite HPO run"
        )
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
            raise ValueError(
                "recorded job plan differs from the pinned standard Lite plan"
            )
        paths = completed_result_paths(run_dir, resolved)
        report = run_dir / "report-local"
        summary = json.loads((report / "summary.json").read_text())
        if summary.get("run_fingerprint") != manifest["fingerprint"] or summary.get(
            "ledger_sha256"
        ) != file_hash(run_dir / "jobs.jsonl"):
            raise ValueError(
                "regenerate reports from the current terminal ledger before acceptance"
            )
        for method in manifest["spec"]["models"]:
            variants = pd.read_csv(run_dir / (method + "-variants.csv"))
            if set(variants.method_subtype) != {"default", "tuned", "tuned_ensemble"}:
                raise ValueError("missing standard variants: " + method)
        result = {
            "run_fingerprint": manifest["fingerprint"],
            "expected_jobs": len(jobs),
            "completed_jobs": len(paths),
            "failed_jobs": len(failures),
            "failure_explanations": explanations,
            "ledger_sha256": file_hash(run_dir / "jobs.jsonl"),
            "leaderboard_sha256": file_hash(report / "leaderboard.csv"),
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
