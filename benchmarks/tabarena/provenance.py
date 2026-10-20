"""Attest the actual native extension and source used by benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def fingerprint(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(root):
    paths = [
        *root.glob("crates/**/*.rs"),
        *root.glob("bindings/python/src/**/*.rs"),
        *root.glob("bindings/python/alloygbm/**/*.py"),
        root / "Cargo.toml",
        *root.glob("crates/*/Cargo.toml"),
        root / "bindings/python/Cargo.toml",
    ]
    paths = [p for p in paths if ".cache" not in p.parts and ".venv" not in p.parts]
    paths.extend(
        [root / "Cargo.lock", root / "rust-toolchain.toml", root / "pyproject.toml"]
    )
    return fingerprint(
        {str(p.relative_to(root)): file_hash(p) for p in sorted(set(paths))}
    )


def runtime_identity(root):
    import inspect

    import alloygbm
    from alloygbm import GBMRegressor, _alloygbm

    if (
        Path(alloygbm.__file__).resolve().parent
        != (root / "bindings/python/alloygbm").resolve()
    ):
        raise ValueError(
            "imported AlloyGBM package is not the selected source checkout"
        )
    if "time_limit" not in inspect.signature(GBMRegressor.fit).parameters:
        raise ValueError(
            "installed extension/API lacks required fit-time deadline support"
        )
    return {
        "source_hash": source_hash(root),
        "extension_path": str(Path(_alloygbm.__file__).resolve()),
        "extension_sha256": file_hash(_alloygbm.__file__),
        "python_package": alloygbm.__file__,
    }


def record_build_receipt(source_root: Path, output: Path):
    """Call immediately after the documented clean release build, not as a rebuild substitute."""
    root = source_root.resolve()
    receipt = runtime_identity(root)
    receipt["source_commit"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt["rustc"] = subprocess.check_output(
        ["rustc", "--version"], text=True
    ).strip()
    receipt["profile"] = "release"
    receipt["build_flags"] = {
        k: os.environ.get(k)
        for k in (
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_PROFILE_RELEASE_OPT_LEVEL",
        )
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def collect_provenance(source_root: Path, receipt_path: Path):
    root = source_root.resolve()
    forbidden = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("ALLOYGBM_EXPERIMENT_") or k == "ALLOYGBM_PROFILE"
    }
    if forbidden:
        raise ValueError(
            "experiment/profiling environment is active: " + str(sorted(forbidden))
        )
    receipt = json.loads(receipt_path.read_text())
    current = runtime_identity(root)
    for key in ("source_hash", "extension_sha256", "extension_path"):
        if current[key] != receipt[key]:
            raise ValueError(
                "stale build receipt: "
                + key
                + " differs; rebuild the release extension"
            )
    flags = {k: os.environ.get(k) for k in receipt["build_flags"]}
    if flags != receipt["build_flags"]:
        raise ValueError("build flags differ from receipt")
    from packaging.requirements import Requirement

    lock = root / "benchmarks/tabarena/requirements.lock"
    for line in lock.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line.strip())
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if requirement.url:
            continue  # The three source packages are attested separately.
        if not requirement.specifier.contains(
            importlib.metadata.version(requirement.name), prereleases=True
        ):
            raise ValueError(
                "installed dependency differs from lock: " + requirement.name
            )
    import bencheval
    import psutil

    import tabarena

    upstream = root / ".cache/tabarena/upstream"
    for name, module in (("tabarena", tabarena), ("bencheval", bencheval)):
        expected = upstream / "packages" / name / "src" / name
        if Path(module.__file__).resolve().parent != expected.resolve():
            raise ValueError("install the pinned prepared clone as editable: " + name)
    if sys.platform == "darwin":
        cpu_model = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    else:
        cpu_model = platform.processor()
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            cpu_model = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")
                ),
                cpu_model,
            )
    hardware = {
        "cpu_model": cpu_model,
        "physical_cpus": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "effective_cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
    }
    versions = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    result = {
        **receipt,
        "versions": dict(sorted(versions.items())),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "hardware": hardware,
        "command": sys.argv,
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    result["dependency_lock_sha256"] = (
        file_hash(root / "benchmarks/tabarena/requirements.lock")
        if (root / "benchmarks/tabarena/requirements.lock").exists()
        else None
    )
    result["source_status"] = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    result["dirty_diff_sha256"] = hashlib.sha256(
        subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"])
    ).hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["record-build", "check"])
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/tabarena/build-receipt.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(
        json.dumps(
            record_build_receipt(root, args.output)
            if args.action == "record-build"
            else collect_provenance(root, args.output),
            indent=2,
        )
    )
