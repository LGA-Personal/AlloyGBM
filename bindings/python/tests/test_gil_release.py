import threading
import time

import numpy as np

from alloygbm import GBMRegressor


def _worker_progress_during(call):
    ready = threading.Event()
    stop = threading.Event()
    counter = 0

    def worker():
        nonlocal counter
        ready.set()
        while not stop.is_set():
            counter += 1

    thread = threading.Thread(target=worker)
    thread.start()
    ready.wait(timeout=1.0)
    time.sleep(0.01)
    before = counter
    started = time.perf_counter()
    call()
    elapsed = time.perf_counter() - started
    after = counter
    stop.set()
    thread.join(timeout=1.0)
    return after - before, elapsed


def test_native_predict_numpy_array_releases_gil():
    rng = np.random.default_rng(17)
    x_train = rng.normal(size=(512, 6)).astype(np.float32)
    y_train = (
        0.8 * x_train[:, 0] - 0.4 * x_train[:, 1] + 0.2 * x_train[:, 2]
    ).astype(np.float32)
    model = GBMRegressor(
        n_estimators=80,
        max_depth=5,
        min_data_in_leaf=2,
        learning_rate=0.05,
        seed=17,
        deterministic=True,
    ).fit(x_train, y_train)

    handle = model._native_predictor_handle
    assert handle is not None
    x_predict = np.ascontiguousarray(
        np.resize(x_train, (1_000_000, x_train.shape[1])).astype(np.float32)
    )

    calibration_progress, calibration_elapsed = _worker_progress_during(
        lambda: time.sleep(0.05)
    )
    progress, elapsed = _worker_progress_during(
        lambda: handle.predict_numpy_array(x_predict)
    )

    assert elapsed >= 0.02
    calibration_rate = calibration_progress / calibration_elapsed
    prediction_rate = progress / elapsed
    assert prediction_rate >= calibration_rate * 0.25


def test_native_shap_global_importance_releases_gil():
    rng = np.random.default_rng(23)
    x_train = rng.normal(size=(512, 6)).astype(np.float32)
    y_train = (
        0.7 * x_train[:, 0] - 0.3 * x_train[:, 1] + 0.1 * x_train[:, 3]
    ).astype(np.float32)
    model = GBMRegressor(
        n_estimators=20,
        max_depth=4,
        min_data_in_leaf=2,
        learning_rate=0.05,
        seed=23,
        deterministic=True,
    ).fit(x_train, y_train)

    x_explain = np.ascontiguousarray(
        np.resize(x_train, (10_000, x_train.shape[1])).astype(np.float32)
    )

    calibration_progress, calibration_elapsed = _worker_progress_during(
        lambda: time.sleep(0.05)
    )
    progress, elapsed = _worker_progress_during(
        lambda: model.feature_importances(x_explain, method="shap")
    )

    assert elapsed >= 0.02
    calibration_rate = calibration_progress / calibration_elapsed
    explanation_rate = progress / elapsed
    assert explanation_rate >= calibration_rate * 0.25


def test_native_training_without_python_callbacks_releases_gil():
    rng = np.random.default_rng(31)
    x_train = rng.normal(size=(12_000, 8)).astype(np.float32)
    y_train = (
        0.9 * x_train[:, 0]
        - 0.35 * x_train[:, 2]
        + 0.2 * x_train[:, 5]
        + 0.1 * x_train[:, 0] * x_train[:, 1]
    ).astype(np.float32)

    def fit_model():
        GBMRegressor(
            n_estimators=80,
            max_depth=5,
            min_data_in_leaf=2,
            learning_rate=0.05,
            training_policy="manual",
            continuous_binning_strategy="quantile",
            seed=31,
            deterministic=True,
        ).fit(x_train, y_train)

    calibration_progress, calibration_elapsed = _worker_progress_during(
        lambda: time.sleep(0.05)
    )
    progress, elapsed = _worker_progress_during(fit_model)

    assert elapsed >= 0.02
    calibration_rate = calibration_progress / calibration_elapsed
    training_rate = progress / elapsed
    assert training_rate >= calibration_rate * 0.25


def test_native_training_with_python_callbacks_releases_gil_between_invocations():
    rng = np.random.default_rng(37)
    x_train = rng.normal(size=(50_000, 8)).astype(np.float32)
    y_train = (
        0.75 * x_train[:, 0]
        - 0.45 * x_train[:, 3]
        + 0.3 * x_train[:, 6]
        + 0.15 * x_train[:, 1] * x_train[:, 2]
    ).astype(np.float32)
    x_val = np.ascontiguousarray(x_train[:2_000])
    y_val = np.ascontiguousarray(y_train[:2_000])
    fixed_grad = (-y_train).astype(np.float32)
    fixed_hess = np.ones_like(y_train, dtype=np.float32)

    def squared_error_objective(y_true, y_pred):
        return fixed_grad, fixed_hess

    def custom_rmse(y_true, y_pred):
        return "custom_constant", 1.0, False

    def fit_model():
        GBMRegressor(
            n_estimators=80,
            max_depth=6,
            min_data_in_leaf=2,
            learning_rate=0.05,
            objective=squared_error_objective,
            training_policy="manual",
            continuous_binning_strategy="quantile",
            seed=37,
            deterministic=True,
        ).fit(x_train, y_train, eval_set=(x_val, y_val), eval_metric=custom_rmse)

    calibration_progress, calibration_elapsed = _worker_progress_during(
        lambda: time.sleep(0.05)
    )
    progress, elapsed = _worker_progress_during(fit_model)

    assert elapsed >= 0.02
    calibration_rate = calibration_progress / calibration_elapsed
    training_rate = progress / elapsed
    assert training_rate >= calibration_rate * 0.65
