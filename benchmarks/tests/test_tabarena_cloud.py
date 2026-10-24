import hashlib
import json
import subprocess

import pytest


def test_export_and_verify_survive_removal_of_server_copy(tmp_path):
    from benchmarks.tabarena.cloud import export_bundle, verify_bundle

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "source.py").write_text("# frozen source\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    run = root / "benchmarks/results/tabarena/run"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"fingerprint": "test"}))
    (run / "jobs.jsonl").write_text('{"status":"failed"}\n')
    cache = root / ".cache/tabarena/cache"
    cache.mkdir(parents=True)
    (cache / "predictions.pkl").write_bytes(b"trusted cache")
    lock = root / "benchmarks/tabarena/requirements.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("# lock\n")
    receipt = root / ".cache/tabarena/build-receipt.json"
    receipt.write_text("{}")
    bundle = export_bundle(root, run, tmp_path / "evidence.tar.gz")
    import shutil

    shutil.rmtree(root)
    assert verify_bundle(bundle)["run_fingerprint"] == "test"
    import tarfile

    recovered_bundle = tmp_path / "source.bundle"
    with tarfile.open(bundle) as archive:
        recovered_bundle.write_bytes(archive.extractfile("source.bundle").read())
        assert archive.extractfile("cache/predictions.pkl").read() == b"trusted cache"
    recovered = tmp_path / "recovered"
    subprocess.run(
        ["git", "clone", "-q", str(recovered_bundle), str(recovered)], check=True
    )
    assert (recovered / "source.py").read_text() == "# frozen source\n"
    bundle.write_bytes(bundle.read_bytes()[:-20])
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(bundle)


def test_export_refuses_live_run_and_symlinked_evidence(tmp_path):
    from benchmarks.tabarena.cloud import export_bundle
    from benchmarks.tabarena.run import exclusive_run

    root = tmp_path / "repo"
    run = root / "benchmarks/results/tabarena/run"
    with exclusive_run(run), pytest.raises(ValueError, match="driver"):
        export_bundle(root, run, tmp_path / "out.tar.gz")
    (run / "run_manifest.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="symlink"):
        export_bundle(root, run, tmp_path / "out.tar.gz")


def test_verify_requires_sidecar(tmp_path):
    from benchmarks.tabarena.cloud import verify_bundle

    archive = tmp_path / "missing.tar.gz"
    archive.write_bytes(b"incomplete transfer")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_bundle(archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.parametrize("fit_exit,report_exit", [(0, 0), (124, 1), (1, 0)])
@pytest.mark.parametrize(
    "stage",
    [
        "smoke",
        "large",
        "lite-defaults",
        "lite-hpo",
        "modes-smoke",
        "modes-large",
        "lite-200",
    ],
)
def test_session_exports_failures_and_preserves_exit_status(
    tmp_path, fit_exit, report_exit, stage
):
    """Exercise the real shell workflow with process boundaries replaced, no fits."""
    import os
    import shutil
    from pathlib import Path

    root = tmp_path / "repo"
    remote = root / "benchmarks/tabarena/remote"
    remote.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "tabarena/remote/run-session.sh"
    shutil.copyfile(source, remote / "run-session.sh")
    (root / ".cache/tabarena").mkdir(parents=True)
    fake_python = root / ".venv-tabarena/bin/python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("""#!/usr/bin/env bash
set -eu
case "$2:$3" in
  benchmarks.tabarena.cloud:preflight) echo '{}';;
  benchmarks.tabarena.run:--profile)
    while [[ "$1" != --output-root ]]; do shift; done
    mkdir -p "$2/run"
    echo '{}' > "$2/run/run_manifest.json"
    exit "$TASK_TEST_FIT_EXIT";;
  benchmarks.tabarena.report:*) exit "$TASK_TEST_REPORT_EXIT";;
  benchmarks.tabarena.cloud:export) echo "$*" > "$TASK_TEST_EXPORT_LOG";;
  *) exit 99;;
esac
""")
    fake_python.chmod(0o700)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout = fake_bin / "timeout"
    timeout.write_text(
        '#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$*" >> "$TASK_TEST_TIMEOUT_LOG"\nshift 3\nexec "$@"\n'
    )
    timeout.chmod(0o700)
    env = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "ALLOYGBM_BENCH_SESSION_ROOT": str(tmp_path),
        "TASK_TEST_FIT_EXIT": str(fit_exit),
        "TASK_TEST_REPORT_EXIT": str(report_exit),
        "TASK_TEST_EXPORT_LOG": str(tmp_path / "export.log"),
        "TASK_TEST_TIMEOUT_LOG": str(tmp_path / "timeout.log"),
    }
    result = subprocess.run(
        ["bash", str(remote / "run-session.sh"), stage, "2"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (fit_exit or report_exit), result.stderr
    assert "benchmarks.tabarena.cloud export" in (tmp_path / "export.log").read_text()
    bounds = (tmp_path / "timeout.log").read_text()
    assert "--models AlloyGBM" in bounds
    assert (
        "--profile lite-200"
        if stage in {"modes-smoke", "modes-large", "lite-200"}
        else "--profile lite-hpo"
    ) in bounds
    assert "--threads 16" in bounds
    assert "--published-only" in bounds
    if stage in {"smoke", "large"}:
        assert "--stage-candidates 1" in bounds
    if stage == "lite-defaults":
        assert "--stage-candidates 0" in bounds
    assert "--signal=INT --kill-after=120s 2h" in bounds
    assert "--signal=INT --kill-after=120s 1h" in bounds
    status = json.loads(
        next(
            root.glob("benchmarks/results/tabarena/*/run/session-status.json")
        ).read_text()
    )
    assert status == {"fit_exit": fit_exit, "report_exit": report_exit}
