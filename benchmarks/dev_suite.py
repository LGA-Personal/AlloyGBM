"""Development (calibration) dataset suite, disjoint from the OpenML test suites.

OpenML-CC18 (suite 99) and OpenML-CTR23 (suite 353) are AlloyGBM's held-out
test set: ``frontier_matrix.py --openml-suite cc18 ctr23`` is the benchmark
that decides whether a change ships. Defaults, auto-mode thresholds and
meta-feature rules must therefore be fitted on *other* data, or the test
numbers stop meaning anything. This module defines that other data.

The suite is drawn from OpenML suites that are not the test suites (the
Grinsztajn et al. 2022 tabular benchmark, 334-337, and the AutoML Benchmark,
269 and 271). Every dataset that also appears in CC18 or CTR23 is removed,
matched four ways because the same data is often re-uploaded under a new id
or name:

1. the same OpenML dataset id;
2. the same normalized name (lowercase, alphanumerics only);
3. a manual alias list of known re-uploads (``houses`` is California
   housing, ``Australian`` is the credit-approval data, ...);
4. the same row x column shape.

``freeze`` resolves the suites over the network and writes the surviving
tasks, and every exclusion with its reason, to ``dev_suite_manifest.json``
next to this file; that committed file is the suite's definition.
``export`` downloads each frozen task once, caps it at ``--max-rows`` rows
(a deterministic, stratified subsample) and saves one ``.npz`` per dataset,
so calibration can run where openml.org is unreachable. ``load`` turns that
cache into :class:`frontier_matrix.Dataset` objects with seeded 75/25
splits, which ``frontier_matrix.py run --dev-suite DIR`` scores like any
other cell.

Note: ``default_quality.py`` predates this split and includes scikit-learn's
``breast_cancer`` (CC18's wdbc) and ``digits`` (CC18's optdigits); do not
calibrate on those two.

Usage::

    python benchmarks/dev_suite.py freeze            # needs openml + network
    python benchmarks/dev_suite.py export --cache DIR [--max-rows 20000]
    python benchmarks/frontier_matrix.py run --dev-suite DIR --regimes defaults \\
        --arms alloy --output dev.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "dev_suite_manifest.json"

TEST_SUITES = {"cc18": 99, "ctr23": 353}
DEV_SUITES = {
    "grinsztajn-num-clf": 337,
    "grinsztajn-num-reg": 336,
    "grinsztajn-cat-clf": 334,
    "grinsztajn-cat-reg": 335,
    "amlb-clf": 271,
    "amlb-reg": 269,
}

# Normalized dev-side name -> normalized test-side names it duplicates. Only
# re-uploads whose names differ need listing; equal names are caught anyway.
ALIASES: dict[str, tuple[str, ...]] = {
    "houses": ("californiahousing",),  # Pace & Barry California census blocks
    "california": ("californiahousing",),
    "housesales": ("kingscounty",),
    "wine": ("whitewine", "redwine"),  # Grinsztajn's binarized wine quality
    "winequality": ("whitewine", "redwine"),
    "superconduct": ("superconductivity",),
    "cpuact": ("cpuactivity",),
    "miamihousing2016": ("miamihousing",),
    "australian": ("creditapproval",),  # Quinlan's credit screening data
    "satellite": ("satimage",),  # AMLB's anomaly version of satimage
    "bankmarketing": ("bankmarketing",),
    "brazilianhouses": ("brazilianhouses",),
}

# Same underlying data as a test task without sharing name, id or shape,
# found by reading the kept list (2026-10-13), plus one dev-internal duplicate.
SUSPECTED_OVERLAP: dict[str, str] = {
    "ada": "built from the census Adult data (CC18 adult)",
    "gina": "built from MNIST digits (CC18 mnist_784)",
    "madeline": "artificial AutoML-challenge set generated like madelon (CC18 madelon)",
    "sylvine": "derived from forest covertype, which the dev suite already has",
}

# Datasets kept out regardless of overlap: too large to be a quick dev
# signal even after subsampling, or images/text that CC18 already excludes.
SKIP_NAMES = {"mnist784", "fashionmnist", "cifar10", "devnagariscript"}

MAX_FEATURES = 2_000
DEFAULT_MAX_ROWS = 20_000
SPLITS = 3


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _openml():
    try:
        import openml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("dev_suite freeze/export need the 'openml' package") from exc
    return openml


def _suite_entries(openml, suite_id: int) -> list[dict]:
    """Task id, dataset id, name and shape for every task of a suite."""
    suite = openml.study.get_suite(suite_id)
    out = []
    for task_id in suite.tasks:
        task = openml.tasks.get_task(task_id, download_data=False, download_qualities=False)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False,
                                              download_qualities=True)
        q = dataset.qualities or {}
        kind = ("regression"
                if task.task_type_id == openml.tasks.TaskType.SUPERVISED_REGRESSION
                else "classification")
        out.append({
            "task_id": int(task_id),
            "dataset_id": int(task.dataset_id),
            "name": dataset.name,
            "kind": kind,
            "rows": int(q.get("NumberOfInstances", 0) or 0),
            "features": int(q.get("NumberOfFeatures", 0) or 0) - 1,
            "classes": int(q.get("NumberOfClasses", 0) or 0),
        })
    return out


def overlap_reason(entry: dict, test: Sequence[dict]) -> str | None:
    """Why ``entry`` duplicates a test-suite dataset, or None if it does not."""
    name = normalize(entry["name"])
    for t in test:
        tname = normalize(t["name"])
        if entry["dataset_id"] == t["dataset_id"]:
            return f"same dataset id as test task {t['task_id']} ({t['name']})"
        if name == tname:
            return f"same name as test task {t['task_id']} ({t['name']})"
        if tname in ALIASES.get(name, ()):
            return f"known re-upload of test task {t['task_id']} ({t['name']})"
        if entry["rows"] and (entry["rows"], entry["features"]) == (t["rows"], t["features"]):
            return f"same shape as test task {t['task_id']} ({t['name']})"
    return None


def freeze(path: Path = MANIFEST, log: Callable[[str], None] = print) -> dict:
    openml = _openml()
    test: list[dict] = []
    for label, suite_id in TEST_SUITES.items():
        test += [e | {"suite": label} for e in _suite_entries(openml, suite_id)]
    log(f"test suites: {len(test)} tasks")
    kept: list[dict] = []
    excluded: list[dict] = []
    seen_names: set[str] = set()
    for label, suite_id in DEV_SUITES.items():
        for e in _suite_entries(openml, suite_id):
            e = e | {"suite": label}
            name = normalize(e["name"])
            reason = overlap_reason(e, test)
            if reason is None and name in SUSPECTED_OVERLAP:
                reason = f"suspected overlap: {SUSPECTED_OVERLAP[name]}"
            if reason is None and name in SKIP_NAMES:
                reason = "image data, skipped as in the CC18 pilot"
            if reason is None and e["features"] > MAX_FEATURES:
                reason = f"{e['features']} features > {MAX_FEATURES}"
            if reason is None and name in seen_names:
                reason = "already in the dev suite from another suite"
            if reason is None and e["kind"] == "classification" and e["classes"] > 100:
                reason = f"{e['classes']} classes"
            if reason:
                excluded.append(e | {"reason": reason})
                continue
            seen_names.add(name)
            kept.append(e)
    manifest = {
        "description": "AlloyGBM dev/calibration suite; disjoint from OpenML CC18 and CTR23",
        "test_suites": TEST_SUITES,
        "dev_suites": DEV_SUITES,
        "aliases": {k: list(v) for k, v in ALIASES.items()},
        "max_features": MAX_FEATURES,
        "tasks": sorted(kept, key=lambda e: (e["kind"], e["name"].lower())),
        "excluded": sorted(excluded, key=lambda e: (e["suite"], e["name"].lower())),
        "test_tasks": sorted(({k: t[k] for k in ("task_id", "dataset_id", "name", "suite")}
                              for t in test), key=lambda t: t["task_id"]),
    }
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"dev suite: {len(kept)} tasks kept, {len(excluded)} excluded -> {path}")
    return manifest


def _subsample(n: int, max_rows: int, labels: np.ndarray | None, seed: int) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n)
    from sklearn.model_selection import train_test_split

    stratify = labels
    if labels is not None:
        _, counts = np.unique(labels, return_counts=True)
        if counts.min() < 2:
            stratify = None
    keep, _ = train_test_split(np.arange(n), train_size=max_rows, random_state=seed,
                               stratify=stratify)
    return np.sort(keep)


def export(cache: Path, max_rows: int = DEFAULT_MAX_ROWS, path: Path = MANIFEST,
           tasks: Iterable[int] | None = None, log: Callable[[str], None] = print) -> None:
    """Download each frozen task once and save it as ``<cache>/<task_id>.npz``."""
    import pandas as pd

    openml = _openml()
    manifest = json.loads(path.read_text())
    wanted = set(tasks) if tasks else None
    cache.mkdir(parents=True, exist_ok=True)
    for e in manifest["tasks"]:
        if wanted and e["task_id"] not in wanted:
            continue
        out = cache / f"{e['task_id']}.npz"
        if out.exists():
            continue
        try:
            task = openml.tasks.get_task(e["task_id"], download_data=False)
            dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True)
            X, y, is_cat, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
            if str(getattr(dataset, "format", "")).lower() == "sparse_arff":
                # Sparse ARFF leaves implicit zeros as missing.
                X = X.astype(float).fillna(0.0)
            labels = None if e["kind"] == "regression" else pd.Categorical(y).codes
            keep = _subsample(len(X), max_rows, labels, seed=e["task_id"])
            X, y = X.iloc[keep], y.iloc[keep]
            columns, categorical = [], []
            for j, (col, cat) in enumerate(zip(X.columns, is_cat)):
                values = X[col]
                if cat or values.dtype == object or str(values.dtype) == "category":
                    codes = pd.Categorical(values).codes.astype(float)
                    codes[codes < 0] = np.nan
                    columns.append(codes)
                    categorical.append(j)
                else:
                    columns.append(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))
            if e["kind"] == "regression":
                target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
            else:
                target = pd.Categorical(y).codes.astype(float)
                target[target < 0] = np.nan
            np.savez_compressed(out, X=np.column_stack(columns), y=target,
                                categorical=np.asarray(categorical, dtype=np.int64),
                                kind=e["kind"], name=e["name"], task_id=e["task_id"],
                                original_rows=e["rows"])
            log(f"  {e['task_id']:>7} {e['name']:<40} {len(keep):>7,} rows")
        except Exception as exc:  # keep going; a missing file is reported by load()
            log(f"  {e['task_id']:>7} {e['name']:<40} FAILED: {type(exc).__name__}: {exc}")


def load(cache: Path, splits: int = SPLITS, tasks: Iterable[int] | None = None,
         path: Path = MANIFEST, log: Callable[[str], None] = print) -> list:
    """Datasets from an exported cache, each with ``splits`` seeded 75/25 folds."""
    sys.path.insert(0, str(HERE))
    from sklearn.model_selection import train_test_split

    import frontier_matrix as fm

    manifest = json.loads(path.read_text())
    wanted = set(tasks) if tasks else None
    out = []
    for e in manifest["tasks"]:
        if wanted and e["task_id"] not in wanted:
            continue
        file = cache / f"{e['task_id']}.npz"
        if not file.exists():
            log(f"  dev suite: {e['task_id']} ({e['name']}) not in cache, skipped")
            continue
        z = np.load(file, allow_pickle=False)
        X, y = z["X"], z["y"].astype(float)
        if e["kind"] == "regression":
            task = fm.REGRESSION
        else:
            labels, counts = np.unique(y[np.isfinite(y)], return_counts=True)
            y[~np.isin(y, labels[counts >= 4])] = np.nan
            kept = np.isfinite(y)
            y[kept] = np.unique(y[kept], return_inverse=True)[1]
            task = fm.BINARY if len(np.unique(y[kept])) == 2 else fm.MULTICLASS
        finite = np.isfinite(y)
        X, y = X[finite], y[finite]
        classify = task != fm.REGRESSION
        folds = []
        for s in range(splits):
            tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=s,
                                      stratify=y if classify else None)
            folds.append((np.sort(tr), np.sort(te)))
        n_classes = int(y.max()) + 1 if classify else 2
        spec = fm.DataSpec(name=f"dev_{e['task_id']}_{e['name']}", task=task,
                           axis="real data", level="dev", tier="full", rows=len(y),
                           features=X.shape[1], n_classes=max(n_classes, 2),
                           cat_low=0, cat_high=0)
        out.append(fm.Dataset(spec, X, y, tuple(int(c) for c in z["categorical"]), None,
                              folds=tuple(folds)))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="resolve suites over the network and write the manifest")
    ex = sub.add_parser("export", help="download frozen tasks into an .npz cache")
    ex.add_argument("--cache", type=Path, required=True)
    ex.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    ex.add_argument("--tasks", type=int, nargs="+")
    sh = sub.add_parser("show", help="print the frozen suite")
    sh.add_argument("--excluded", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze()
    elif args.command == "export":
        export(args.cache, args.max_rows, tasks=args.tasks)
    else:
        manifest = json.loads(MANIFEST.read_text())
        for e in manifest["excluded" if args.excluded else "tasks"]:
            extra = f"  <- {e['reason']}" if args.excluded else ""
            print(f"{e['task_id']:>7} {e['suite']:<20} {e['kind']:<14} {e['rows']:>9,} x "
                  f"{e['features']:<5} {e['name']}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
