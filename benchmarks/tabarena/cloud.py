"""Check a disposable Linux host and preserve evidence before destroying it.

This module never provisions, deletes, or authenticates to a cloud provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(root):
    from benchmarks.tabarena.provenance import collect_provenance
    from benchmarks.tabarena.resources import validate_acceptance_resources

    provenance = collect_provenance(root, root / ".cache/tabarena/build-receipt.json")
    validate_acceptance_resources(
        {"execution": "upstream", "num_cpus": None}, provenance
    )
    if provenance["source_status"].strip():
        raise ValueError("cloud runs require a clean committed checkout")
    if shutil.disk_usage(root).free < 80 * 1024**3:
        raise ValueError(
            "cloud runs require at least 80 GiB free disk after setup; allow extra space for export"
        )
    return provenance


def export_bundle(root, run_dir, output):
    from benchmarks.tabarena.run import exclusive_run

    root, run_dir, output = (
        Path(root).resolve(),
        Path(run_dir).resolve(),
        Path(output).absolute(),
    )
    run_dir.relative_to(root / "benchmarks/results/tabarena")
    if output.exists() or Path(str(output) + ".sha256.json").exists():
        raise ValueError("refusing to overwrite an existing evidence bundle")
    if output.is_relative_to(root):
        raise ValueError(
            "export outside the checkout to avoid archiving the output itself"
        )
    with exclusive_run(run_dir):
        inputs = {
            "run": run_dir,
            "cache": root / ".cache/tabarena/cache",
            "requirements.lock": root / "benchmarks/tabarena/requirements.lock",
            "build-receipt.json": root / ".cache/tabarena/build-receipt.json",
        }
        for path in inputs.values():
            if path.is_symlink() or any(
                p.is_symlink() for p in path.rglob("*") if path.is_dir()
            ):
                raise ValueError("symlinked evidence must be resolved before export")
            if not path.exists():
                raise FileNotFoundError(path)
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        if partial.exists():
            raise ValueError(
                "partial export already exists; inspect it before retrying"
            )
        try:
            with tempfile.TemporaryDirectory(dir=output.parent) as temp:
                source = Path(temp) / "source.bundle"
                subprocess.run(
                    ["git", "-C", str(root), "bundle", "create", str(source), "HEAD"],
                    check=True,
                )
                patch = Path(temp) / "source.patch"
                patch.write_bytes(
                    subprocess.check_output(
                        ["git", "-C", str(root), "diff", "--binary", "HEAD"]
                    )
                )
                with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
                    for name, path in {
                        **inputs,
                        "source.bundle": source,
                        "source.patch": patch,
                    }.items():
                        archive.add(path, arcname=name)
            partial.rename(output)
            evidence = {
                "sha256": sha256(output),
                "run_fingerprint": manifest["fingerprint"],
                "archive": output.name,
            }
            Path(str(output) + ".sha256.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )
        finally:
            partial.unlink(missing_ok=True)
    verify_bundle(output)
    return output


def verify_bundle(path):
    path = Path(path)
    evidence = json.loads(Path(str(path) + ".sha256.json").read_text())
    if sha256(path) != evidence["sha256"]:
        raise ValueError(
            "evidence bundle checksum mismatch; do not delete the server copy"
        )
    required = {
        "run/run_manifest.json",
        "run/jobs.jsonl",
        "source.bundle",
        "requirements.lock",
        "build-receipt.json",
    }
    found = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = Path(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("unsafe evidence archive member")
            found.add(member.name)
            if member.name == "run/run_manifest.json":
                manifest = json.load(archive.extractfile(member))
                if manifest["fingerprint"] != evidence["run_fingerprint"]:
                    raise ValueError("bundle manifest differs from checksum receipt")
    if required - found or not any(name.startswith("cache/") for name in found):
        raise ValueError("evidence bundle is missing required recovery files")
    return evidence


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    export = commands.add_parser("export")
    export.add_argument("run_dir", type=Path)
    export.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(root), indent=2))
    elif args.command == "export":
        print(export_bundle(root, args.run_dir, args.output))
    else:
        print(json.dumps(verify_bundle(args.archive), indent=2))


if __name__ == "__main__":
    main()
