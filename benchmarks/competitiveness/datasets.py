"""Deterministic, download-free fixtures for competitiveness measurements."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

try:  # scipy is an existing benchmark dependency, but keep import optional.
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - exercised only in minimal installs
    sp = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DatasetCase:
    name: str
    task: str
    X_train: object
    y_train: np.ndarray
    X_test: object
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    group_train: np.ndarray | None
    group_test: np.ndarray | None
    categorical_feature_indices: tuple[int, ...]
    input_representation: str
    dataset_sha256: str
    rounds: int
    depth: int
    metric_name: str


def _split(rows: int, *, group_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    cut = rows * 4 // 5
    if group_size is not None:
        cut = (cut // group_size) * group_size
    if cut <= 0 or cut >= rows:
        raise ValueError("rows must provide nonempty train and test splits")
    return np.arange(cut, dtype=np.int64), np.arange(cut, rows, dtype=np.int64)


def _dense_regression(rng: np.random.Generator, rows: int, features: int) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((rows, features)).astype(np.float32)
    coefficients = np.array([1.25, -0.9, 0.55, 0.3, -0.2], dtype=np.float32)[:features]
    y = X[:, : len(coefficients)] @ coefficients + 0.75 * X[:, 0] * X[:, 1]
    y += rng.normal(0.0, 0.1, rows).astype(np.float32)
    return X, y.astype(np.float32)


def _published_deep_scaling_dense(
    rng: np.random.Generator, rows: int, features: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ``benchmarks/deep_scaling_comparison.py::make_dataset``."""

    X = rng.normal(size=(rows, features)).astype(np.float32)
    signal = X[:, :5] @ np.array([1.5, -2.0, 0.75, 1.0, -0.5], dtype=np.float32)
    interaction = 0.8 * X[:, 0] * X[:, 1]
    y = (signal + interaction + rng.normal(scale=0.1, size=rows)).astype(np.float32)
    return X, y


def _binary(rng: np.random.Generator, rows: int, features: int) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((rows, features)).astype(np.float32)
    coefficients = np.array([1.2, -1.0, 0.7, 0.45, -0.35], dtype=np.float32)[:features]
    logit = X[:, : len(coefficients)] @ coefficients + 0.8 * X[:, 0] * X[:, 1] - 0.2 * X[:, min(2, features - 1)] ** 2
    y = (logit > 0.0).astype(np.float32)
    return X, y


def _ranking(rng: np.random.Generator, rows: int, groups: int, features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rows % groups:
        raise ValueError("ranking rows must be divisible by groups")
    X = rng.standard_normal((rows, features)).astype(np.float32)
    group_size = rows // groups
    signal = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    grade = np.clip(np.floor(2.0 * signal + 2.0), 0, 4).astype(np.int32)
    ids = np.repeat(np.arange(groups, dtype=np.int64), group_size)
    return X, grade, ids


def _categorical(rng: np.random.Generator, rows: int, cardinalities: Sequence[int]) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], list[np.ndarray]]:
    numeric = rng.standard_normal((rows, 20)).astype(np.float32)
    categories: list[np.ndarray] = []
    for cardinality in cardinalities:
        if int(cardinality) <= 0:
            raise ValueError("categorical cardinalities must be positive")
        categories.append(rng.integers(0, int(cardinality), size=rows, dtype=np.int32))
    X = np.column_stack([numeric, *categories]).astype(np.float32)
    cat_indices = tuple(range(20, 20 + len(categories)))
    logit = 0.8 * numeric[:, 0] + (categories[0] == 1) * 1.2
    if len(categories) > 1:
        logit += (categories[1] % 7 == 0) * 0.8
    y = (logit > 0).astype(np.float32)
    return X, y, cat_indices, categories


def _csr(rng: np.random.Generator, rows: int, features: int, density: float):
    if sp is None:
        raise ImportError("scipy is required for csr_sparse fixtures")
    if not 0 < float(density) <= 1:
        raise ValueError("sparse density must be between zero and one")
    count = int(round(rows * features * float(density)))
    # Generate coordinates directly; no dense rows-by-features allocation.
    linear = rng.choice(rows * features, size=count, replace=False)
    row = linear // features
    col = linear % features
    data = rng.standard_normal(count).astype(np.float32)
    X = sp.coo_matrix((data, (row, col)), shape=(rows, features), dtype=np.float32).tocsr()
    y = np.asarray(X[:, : min(5, features)].toarray()).sum(axis=1).astype(np.float32)
    y += rng.normal(0.0, 0.1, rows).astype(np.float32)
    return X, y


def _multi_output(rng: np.random.Generator, rows: int, features: int, outputs: int) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((rows, features)).astype(np.float32)
    weights = rng.normal(0.0, 0.7, (features, outputs)).astype(np.float32)
    y = (X @ weights + rng.normal(0.0, 0.1, (rows, outputs))).astype(np.float32)
    return X, y


def _canonical_hash(parts: Sequence[tuple[str, object]]) -> str:
    digest = hashlib.sha256()
    for label, value in parts:
        digest.update(label.encode("utf-8"))
        if sp is not None and sp.issparse(value):
            matrix = value.tocsr()
            digest.update(b"csr")
            for name, array in (("data", matrix.data), ("indices", matrix.indices), ("indptr", matrix.indptr)):
                digest.update(name.encode())
                _hash_array(digest, array)
            _hash_array(digest, np.asarray(matrix.shape, dtype=np.int64))
        else:
            _hash_array(digest, np.asarray(value))
    return digest.hexdigest()


def _hash_array(digest: "hashlib._Hash", array: np.ndarray) -> None:
    arr = np.asarray(array)
    digest.update(str(tuple(arr.shape)).encode())
    digest.update(str(arr.dtype.newbyteorder("<")).encode())
    if arr.dtype.kind in "OUS":
        arr = np.asarray(arr, dtype="<U")
    else:
        arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<"), copy=False))
    digest.update(np.ascontiguousarray(arr).tobytes(order="C"))


def fingerprint_case(case: DatasetCase) -> str:
    parts: list[tuple[str, object]] = [
        ("X_train", case.X_train), ("y_train", case.y_train),
        ("X_test", case.X_test), ("y_test", case.y_test),
        ("train_indices", case.train_indices), ("test_indices", case.test_indices),
        ("categorical_indices", np.asarray(case.categorical_feature_indices, dtype=np.int64)),
    ]
    if case.group_train is not None:
        parts.extend((("group_train", case.group_train), ("group_test", case.group_test)))
    if sp is not None and sp.issparse(case.X_train):
        parts.append(("sparse_shape", np.asarray(case.X_train.shape, dtype=np.int64)))
    return _canonical_hash(parts)


def build_dataset_cases(scenarios: Sequence[Mapping[str, object]], seed: int) -> list[DatasetCase]:
    cases: list[DatasetCase] = []
    for spec in scenarios:
        name = str(spec["name"])
        task = str(spec["task"])
        fixture = str(spec.get("fixture", "legacy"))
        rows = int(spec["rows"])
        rng = np.random.default_rng(seed)
        groups: int | None = None
        cat_indices: tuple[int, ...] = ()
        cat_values: list[np.ndarray] = []
        if fixture in {"legacy", "nightly_dense"} and name == "dense_regression":
            X, y = _dense_regression(rng, rows, int(spec["features"]))
        elif fixture == "nightly_dense":
            X, y = _dense_regression(rng, rows, int(spec["features"]))
        elif fixture == "published_deep_scaling_v1":
            X, y = _published_deep_scaling_dense(rng, rows, int(spec["features"]))
        elif name == "binary":
            X, y = _binary(rng, rows, int(spec["features"]))
        elif name == "grouped_ranking":
            groups = int(spec["groups"])
            X, y, group_ids = _ranking(rng, rows, groups, int(spec["features"]))
        elif name == "native_categorical":
            X, y, cat_indices, cat_values = _categorical(rng, rows, spec["categorical_cardinalities"])  # type: ignore[arg-type]
        elif name == "csr_sparse":
            X, y = _csr(rng, rows, int(spec["features"]), float(spec["density"]))
        elif name == "joint_multi_output":
            X, y = _multi_output(rng, rows, int(spec["features"]), int(spec["outputs"]))
        else:
            raise ValueError(f"unknown scenario: {name}")
        if groups is None:
            train_idx, test_idx = _split(rows)
            group_train = group_test = None
        else:
            train_idx, test_idx = _split(rows, group_size=rows // groups)
            group_train = group_ids[train_idx]
            group_test = group_ids[test_idx]
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        if task == "binary_classification":
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                raise ValueError(
                    "binary fixture requires both classes in both train and test splits; "
                    f"got train={np.unique(y_train).tolist()} test={np.unique(y_test).tolist()}"
                )
        case = DatasetCase(
            name=name, task=task, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test, train_indices=train_idx,
            test_indices=test_idx, group_train=group_train, group_test=group_test,
            categorical_feature_indices=cat_indices,
            input_representation=str(spec.get("input_representation", "dense")),
            dataset_sha256="0" * 64, rounds=int(spec["rounds"]), depth=int(spec["depth"]),
            metric_name=str(spec["metric"]),
        )
        # Category values are part of the feature array bytes and therefore of
        # the same canonical fingerprint as every other feature value.
        object.__setattr__(case, "dataset_sha256", fingerprint_case(case))
        cases.append(case)
    return cases
