"""Pull immutable checkpoints over SSH and acknowledge verified off-host copies."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from benchmarks.tabarena.checkpoints import atomic_json, verify_checkpoint


def verify_local_chain(directory, *, now=None):
    directory = Path(directory)
    index_path = directory / "verified-index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {"sequence": 0, "sha256": None, "run_fingerprint": None}
    )
    for path in sorted(directory.glob("*.tar.gz")):
        sidecar = Path(str(path) + ".json")
        if not sidecar.exists():
            continue  # A file may arrive before its publication receipt.
        receipt = json.loads(sidecar.read_text())
        if receipt["sequence"] <= index["sequence"]:
            continue  # Already verified immutable bytes; restore verifies all again.
        receipt = verify_checkpoint(path)
        if (
            receipt["sequence"] != index["sequence"] + 1
            or receipt["previous_sha256"] != index["sha256"]
            or index["run_fingerprint"] not in (None, receipt["run_fingerprint"])
        ):
            raise ValueError("backup checkpoint chain gap or identity mismatch")
        index = {key: receipt[key] for key in ("sequence", "sha256", "run_fingerprint")}
        atomic_json(index_path, index)
    if not index["sequence"]:
        raise ValueError("no verified checkpoint backup available")
    # Refresh acknowledgement only after verifying all currently published segments.
    return {**index, "verified_at_unix": time.time() if now is None else now}


def require_recent_backup(run_dir, *, maximum_age, now=None):
    if maximum_age <= 0:
        return
    run_dir = Path(run_dir)
    path = run_dir / "backup-ack.json"
    if not path.exists():
        raise ValueError("verified off-host backup acknowledgement is missing")
    receipt = json.loads(path.read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    age = (time.time() if now is None else now) - receipt["verified_at_unix"]
    if (
        receipt["run_fingerprint"] != manifest["fingerprint"]
        or age < 0
        or age > maximum_age
    ):
        raise ValueError("verified off-host backup is stale or belongs to another run")


def sync_checkpoints(host, remote_run, destination):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", host) or not re.fullmatch(
        r"/[A-Za-z0-9_./-]+", remote_run
    ):
        raise ValueError("use an SSH alias and an absolute simple remote run path")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # rsync publishes a transferred file via rename. Existing archives are immutable.
    subprocess.run(
        [
            "rsync",
            "-a",
            "--ignore-existing",
            "--include=*.tar.gz",
            "--include=*.tar.gz.json",
            "--exclude=*",
            f"{host}:{remote_run}/checkpoints/",
            str(destination) + "/",
        ],
        check=True,
    )
    receipt = verify_local_chain(destination)
    # Send the acknowledgement only after archive/member hashes and chain checks pass.
    script = (
        "import json,os,sys;from pathlib import Path;"
        'p=Path(sys.argv[1])/"backup-ack.json";t=p.with_suffix(".partial");'
        "t.write_text(sys.stdin.read());"
        'f=t.open("rb");os.fsync(f.fileno());f.close();t.replace(p)'
    )
    command = shlex.join(["python3", "-c", script, remote_run])
    subprocess.run(
        ["ssh", host, command], input=json.dumps(receipt) + "\n", text=True, check=True
    )
    atomic_json(destination / "last-sync.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("remote_run")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            sync_checkpoints(args.host, args.remote_run, args.destination), indent=2
        )
    )


if __name__ == "__main__":
    main()
