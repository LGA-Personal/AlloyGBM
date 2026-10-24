"""Incremental immutable result checkpoints; independent of report generation.

Call create_checkpoint only while owning exclusive_run. Verification/restoration
use the standard library and never load pickle files. Keep every chain segment.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

from benchmarks.tabarena.provenance import file_hash
from benchmarks.tabarena.run import exclusive_run, load_latest


def durable_file(path):
    """Sync result bytes before acknowledging them in the durable job ledger."""
    path = Path(path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    durable_file(temporary)
    temporary.replace(path)
    durable_file(path)


def safe_path(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe checkpoint member")
    return path


def create_checkpoint(run_dir):
    run_dir = Path(run_dir)
    directory = run_dir / "checkpoints"
    directory.mkdir(exist_ok=True)
    state_path = directory / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"sequence": 0, "ledger_offset": 0, "keys": [], "previous_sha256": None}
    )
    # Recover publication completed just before a crash writing the local index.
    recovered = False
    while True:
        following = directory / f"{state['sequence'] + 1:06d}.tar.gz"
        if not following.exists() or not Path(str(following) + ".json").exists():
            break
        receipt = verify_checkpoint(following)
        if receipt["previous_sha256"] != state["previous_sha256"]:
            raise ValueError("checkpoint recovery chain differs")
        state = {
            "sequence": receipt["sequence"],
            "ledger_offset": receipt["ledger_offset"],
            "keys": sorted(set(state["keys"]) | set(receipt["completed_keys"])),
            "previous_sha256": receipt["sha256"],
        }
        recovered = True
    if recovered:
        atomic_json(state_path, state)
    latest = load_latest(run_dir)
    saved_keys = set(state["keys"])
    new = {
        key: row
        for key, row in latest.items()
        if row["status"] == "completed" and key not in saved_keys
    }
    ledger = (run_dir / "jobs.jsonl").read_bytes()
    ledger = ledger[: ledger.rfind(b"\n") + 1]
    if state["sequence"] and len(ledger) == state["ledger_offset"]:
        return None
    touched = {
        json.loads(line)["key"]
        for line in ledger[state["ledger_offset"] :].splitlines()
    }
    paths = set()
    for key, row in new.items():
        relative = safe_path(row["cache_path"])
        cache = run_dir / relative
        if cache.is_symlink() or file_hash(cache) != row["result_sha256"]:
            raise ValueError("completed result checksum differs: " + key)
        paths.update(p for p in cache.parent.rglob("*") if p.is_file())
    # Failed/interrupted attempt diagnostics matter even without a result cache.
    for key in touched:
        for folder in ("job_logs", "job_specs"):
            path = (
                run_dir / folder / (key + (".log" if folder == "job_logs" else ".json"))
            )
            if path.exists():
                paths.add(path)
    if not state["sequence"]:
        paths.update(
            run_dir / name
            for name in (
                "run_manifest.json",
                "configs.json",
                "tasks.json",
                "job_plan.json",
            )
        )
    if not state["sequence"] and (run_dir / "origins").exists():
        paths.update(p for p in (run_dir / "origins").rglob("*") if p.is_file())
    if any(p.is_symlink() for p in paths):
        raise ValueError("symlinked checkpoint evidence")
    number = state["sequence"] + 1
    output = directory / f"{number:06d}.tar.gz"
    if output.exists() or Path(str(output) + ".json").exists():
        raise ValueError("uncommitted checkpoint exists; verify it before recovery")
    partial = output.with_suffix(".partial")
    try:
        with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
            for path in sorted(paths):
                archive.add(
                    path, arcname=str(path.relative_to(run_dir)), recursive=False
                )
            delta = ledger[state["ledger_offset"] :]
            info = tarfile.TarInfo("ledger.delta")
            info.size = len(delta)
            archive.addfile(info, io.BytesIO(delta))
        durable_file(partial)
        partial.replace(output)
        receipt = {
            "run_fingerprint": json.loads((run_dir / "run_manifest.json").read_text())[
                "fingerprint"
            ],
            "sequence": number,
            "previous_sha256": state["previous_sha256"],
            "sha256": file_hash(output),
            "completed_keys": sorted(new),
            "ledger_offset": len(ledger),
            "files": {str(p.relative_to(run_dir)): file_hash(p) for p in sorted(paths)},
        }
        atomic_json(Path(str(output) + ".json"), receipt)
        verify_checkpoint(output)
        atomic_json(
            state_path,
            {
                "sequence": number,
                "ledger_offset": len(ledger),
                "keys": sorted(set(state["keys"]) | set(new)),
                "previous_sha256": receipt["sha256"],
            },
        )
        return output
    finally:
        partial.unlink(missing_ok=True)


def verify_checkpoint(archive_path):
    import hashlib

    path = Path(archive_path)
    receipt = json.loads(Path(str(path) + ".json").read_text())
    if file_hash(path) != receipt["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    found = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            safe_path(member.name)
            if not member.isfile() or member.name in found:
                raise ValueError("unsafe or duplicate checkpoint member")
            found.add(member.name)
            stream = archive.extractfile(member)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if (
                member.name != "ledger.delta"
                and receipt["files"].get(member.name) != digest
            ):
                raise ValueError("checkpoint member checksum mismatch")
    if found != set(receipt["files"]) | {"ledger.delta"}:
        raise ValueError("checkpoint files missing")
    return receipt


def restore_checkpoints(directory, destination):
    """Reassemble a verified chain into a new directory; leave originals intact."""
    directory, destination = Path(directory), Path(destination)
    archives = sorted(directory.glob("*.tar.gz"))
    if not archives:
        raise ValueError("empty checkpoint chain")
    receipts = []
    previous, identity = None, None
    for number, path in enumerate(archives, 1):
        receipt = verify_checkpoint(path)
        identity = identity or receipt["run_fingerprint"]
        if (
            receipt["sequence"] != number
            or receipt["previous_sha256"] != previous
            or receipt["run_fingerprint"] != identity
        ):
            raise ValueError("checkpoint chain is incomplete or inconsistent")
        previous = receipt["sha256"]
        receipts.append(receipt)
    if destination.exists():
        raise ValueError("restore destination must be new")
    destination.mkdir(parents=True)
    for path in archives:
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                stream = archive.extractfile(member)
                target = destination / (
                    safe_path(member.name)
                    if member.name != "ledger.delta"
                    else "jobs.jsonl"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open(
                    "ab" if member.name == "ledger.delta" else "wb"
                ) as output:
                    shutil.copyfileobj(stream, output)
    # Preserve the checkpoint chain, including its local index, for further backups.
    shutil.copytree(directory, destination / "checkpoints")
    atomic_json(
        destination / "checkpoints/state.json",
        {
            "sequence": len(receipts),
            "ledger_offset": receipts[-1]["ledger_offset"],
            "keys": sorted({key for r in receipts for key in r["completed_keys"]}),
            "previous_sha256": previous,
        },
    )
    for row in load_latest(destination).values():
        if (
            row["status"] == "completed"
            and file_hash(destination / safe_path(row["cache_path"]))
            != row["result_sha256"]
        ):
            raise ValueError("restored result checksum differs")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "verify", "restore"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.action == "create":
        with exclusive_run(args.path):
            print(create_checkpoint(args.path))
    elif args.action == "verify":
        print(json.dumps(verify_checkpoint(args.path), indent=2))
    else:
        if args.destination is None:
            parser.error("--destination is required for restore")
        print(restore_checkpoints(args.path, args.destination))


if __name__ == "__main__":
    main()
