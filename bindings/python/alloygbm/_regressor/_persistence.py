"""Persistence / pickle methods mixin for GBMRegressor."""

from __future__ import annotations

from . import _base
from ._base import _max_data_bin_for_max_bins


class _PersistenceMixin:
    """Mixin carrying serialization and native-predictor construction methods for GBMRegressor.

    All 9 methods are moved verbatim from GBMRegressor in _core.py.
    No ``GBMRegressor`` class-name references exist in these method bodies,
    so no post-definition injection is required.
    """

    def _reset_fitted_state(self) -> None:
        self._is_fitted = False
        self._artifact_bytes = None
        self._native_predictor_handle = None
        self._float_thresholds_converted = False
        self._n_features_in = 0
        self._uses_continuous_binning = False
        self._continuous_feature_mins = None
        self._continuous_feature_maxs = None
        self._continuous_feature_sorted_values = None
        self._continuous_feature_quantile_cuts = None
        self._continuous_feature_linear_rank_flags = None
        self.feature_names_in_ = None
        self.best_iteration_ = None
        self.best_score_ = None
        self.n_estimators_ = None
        self.rounds_completed_ = None
        self.stop_reason_ = None
        self.diagnostics_per_round_ = None
        self.factor_exposure_diagnostics_ = None
        self.evals_result_ = None
        self.fit_timing_ = None
        self._fit_neutralization = None
        self._fit_factor_neutralization_lambda = None
        self._fit_factor_penalty = None

    # ── Serialization / persistence ──────────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # _native_predictor_handle is a PyO3 object and cannot be pickled.
        # It will be lazily reconstructed from _artifact_bytes on first predict().
        state.pop("_native_predictor_handle", None)
        # Custom objective callables are not serializable.  Store None and let
        # the user re-provide the callable if they need to re-train.
        if callable(state.get("objective")):
            import warnings
            warnings.warn(
                "Custom objective callable cannot be pickled. "
                "The model artifact is preserved for prediction, but "
                "re-training will require re-providing the objective callable.",
                UserWarning,
                stacklevel=2,
            )
            state["objective"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._native_predictor_handle = None
        self._float_thresholds_converted = False
        self._predictor_needs_rebuild = True

    def save_model(self, path: str) -> None:
        """Save the fitted model to a file.

        The file contains a JSON metadata header (constructor params, binning
        metadata, training history) followed by the raw binary artifact.  Use
        :meth:`load_model` to restore.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")
        import json

        saved_params = self.get_params()
        # Callable objectives are not JSON-serializable; store "custom" string.
        if callable(saved_params.get("objective")):
            saved_params["objective"] = None
        metadata = {
            "params": saved_params,
            "n_features_in": self._n_features_in,
            "uses_continuous_binning": self._uses_continuous_binning,
            "continuous_feature_mins": self._continuous_feature_mins,
            "continuous_feature_maxs": self._continuous_feature_maxs,
            "continuous_feature_sorted_values": self._continuous_feature_sorted_values,
            "continuous_feature_quantile_cuts": self._continuous_feature_quantile_cuts,
            "continuous_feature_linear_rank_flags": self._continuous_feature_linear_rank_flags,
            "best_iteration": self.best_iteration_,
            "best_score": self.best_score_,
            "n_estimators_actual": self.n_estimators_,
            "evals_result": self.evals_result_,
            "feature_names_in": self.feature_names_in_,
            "fit_neutralization": self._fit_neutralization,
            "fit_factor_neutralization_lambda": self._fit_factor_neutralization_lambda,
            "fit_factor_penalty": self._fit_factor_penalty,
            "native_cat_mappings": (
                {str(k): v for k, v in self._native_cat_mappings_.items()}
                if self._native_cat_mappings_
                else None
            ),
        }
        # Classifier-specific metadata
        from alloygbm.classifier import GBMClassifier
        if isinstance(self, GBMClassifier):
            metadata["classifier_classes"] = getattr(self, "classes_", None)
            metadata["classifier_n_classes"] = getattr(self, "n_classes_", None)
            encoder = getattr(self, "_label_encoder", None)
            metadata["classifier_label_encoder"] = (
                {str(k): v for k, v in encoder.items()} if encoder is not None else None
            )
            metadata["classifier_num_classes_for_training"] = getattr(
                self, "_num_classes_for_training", None
            )
        metadata_json = json.dumps(metadata).encode("utf-8")
        metadata_len = len(metadata_json)

        with open(path, "wb") as f:
            f.write(b"AGBP")  # magic: AlloyGBM Python model
            f.write(metadata_len.to_bytes(4, "little"))
            f.write(metadata_json)
            f.write(self._artifact_bytes)

    @classmethod
    def load_model(cls, path: str) -> "GBMRegressor":
        """Load a model previously saved with :meth:`save_model`.

        Returns a fitted ``GBMRegressor`` ready for prediction.
        """
        import json

        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"AGBP":
                raise ValueError(
                    f"Not a valid AlloyGBM model file (expected magic b'AGBP', got {magic!r})"
                )
            metadata_len = int.from_bytes(f.read(4), "little")
            metadata_json = f.read(metadata_len)
            artifact_bytes = f.read()

        metadata = json.loads(metadata_json)
        params = metadata["params"]
        # Filter to known params for forward compatibility.
        # Use get_params() keys from a default instance to correctly handle
        # subclasses that use **kwargs (e.g. GBMRanker, GBMClassifier).
        try:
            # Build a temporary default instance to discover valid param names.
            # This works even for subclasses with **kwargs forwarding.
            _probe = cls.__new__(cls)
            cls.__init__(_probe)
            known = set(_probe.get_params().keys())
        except Exception:
            # Fallback: accept all saved params.
            known = set(params.keys())
        model = cls(**{k: v for k, v in params.items() if k in known})
        model._artifact_bytes = artifact_bytes
        model._n_features_in = metadata["n_features_in"]
        model._uses_continuous_binning = metadata["uses_continuous_binning"]
        model._continuous_feature_mins = metadata.get("continuous_feature_mins")
        model._continuous_feature_maxs = metadata.get("continuous_feature_maxs")
        model._continuous_feature_sorted_values = metadata.get(
            "continuous_feature_sorted_values"
        )
        model._continuous_feature_quantile_cuts = metadata.get(
            "continuous_feature_quantile_cuts"
        )
        model._continuous_feature_linear_rank_flags = metadata.get(
            "continuous_feature_linear_rank_flags"
        )
        model.best_iteration_ = metadata.get("best_iteration")
        model.best_score_ = metadata.get("best_score")
        model.n_estimators_ = metadata.get("n_estimators_actual")
        model.evals_result_ = metadata.get("evals_result")
        model.feature_names_in_ = metadata.get("feature_names_in")
        saved_cat_mappings = metadata.get("native_cat_mappings")
        if saved_cat_mappings:
            model._native_cat_mappings_ = {
                int(k): v for k, v in saved_cat_mappings.items()
            }
        else:
            model._native_cat_mappings_ = None
        model._fit_neutralization = metadata.get("fit_neutralization", model.neutralization)
        model._fit_factor_neutralization_lambda = metadata.get(
            "fit_factor_neutralization_lambda",
            model.factor_neutralization_lambda,
        )
        model._fit_factor_penalty = metadata.get(
            "fit_factor_penalty",
            model.factor_penalty,
        )
        model._is_fitted = True
        model._native_predictor_handle = None
        model._float_thresholds_converted = False
        # Eagerly reconstruct the native predictor so predict() uses the fast path.
        model._native_predictor_handle = cls._build_native_predictor_handle(
            artifact_bytes
        )
        model._convert_predictor_thresholds_to_float()

        # Restore subclass-specific fitted attributes.
        from alloygbm.classifier import GBMClassifier

        if isinstance(model, GBMClassifier):
            saved_classes = metadata.get("classifier_classes")
            if saved_classes is not None:
                model.classes_ = saved_classes
                model.n_classes_ = metadata.get("classifier_n_classes", len(saved_classes))
            else:
                model.classes_ = [0, 1]
                model.n_classes_ = 2
            saved_encoder = metadata.get("classifier_label_encoder")
            if saved_encoder is not None:
                model._label_encoder = {int(k): v for k, v in saved_encoder.items()}
                model._label_decoder = {v: int(k) for k, v in saved_encoder.items()}
            else:
                model._label_encoder = None
                model._label_decoder = None
            model._num_classes_for_training = metadata.get(
                "classifier_num_classes_for_training"
            )

        return model

    def save_artifact(self, path: str) -> None:
        """Save only the raw model artifact bytes to a file.

        The resulting file can be loaded with :meth:`predict_from_artifact` for
        lightweight deployment scenarios where retraining is not needed.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving artifact")
        with open(path, "wb") as f:
            f.write(self._artifact_bytes)

    @property
    def artifact_bytes(self) -> bytes:
        """The raw binary model artifact.

        Can be stored externally (database, object store) and used with
        :meth:`predict_from_artifact` for serving without the full model.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to access artifact bytes")
        return self._artifact_bytes

    @staticmethod
    def _build_native_predictor_handle(artifact_bytes: bytes) -> object | None:
        try:
            native_predictor_handle_class = _base._load_native_predictor_handle_class()
        except RuntimeError:
            return None
        try:
            return native_predictor_handle_class(artifact_bytes, strict=True)
        except Exception:
            pass
        # Fallback: non-strict loading (required for multi-class models which
        # use MultiClassTrees section instead of the dual Trees+PredictorLayout
        # format that strict mode requires).
        try:
            return native_predictor_handle_class(artifact_bytes, strict=False)
        except Exception:
            return None

    def _convert_predictor_thresholds_to_float(self) -> None:
        """Convert bin-index thresholds to float thresholds on the native predictor.

        After conversion, predict_dense works directly on raw floats — no quantization needed.
        Supports linear binning, quantile binning, and pre-binned integer data.
        """
        if self._native_predictor_handle is None:
            return
        try:
            if not self._uses_continuous_binning:
                # Pre-binned integer data: threshold_float = bin + 0.5
                convert_fn = getattr(
                    self._native_predictor_handle,
                    "convert_thresholds_to_float_prebinned",
                    None,
                )
                if callable(convert_fn):
                    result = convert_fn()
                    if result is None:
                        self._float_thresholds_converted = True
                return

            strategy = self.continuous_binning_strategy
            if strategy == "linear":
                rank_flags = self._continuous_feature_linear_rank_flags
                if rank_flags is not None and any(rank_flags):
                    return  # rank features need bin-based prediction
                convert_fn = getattr(
                    self._native_predictor_handle, "convert_thresholds_to_float", None
                )
                if not callable(convert_fn):
                    return
                mins, maxs = self._require_continuous_feature_bounds()
                max_data_bin = _max_data_bin_for_max_bins(
                    self.continuous_binning_max_bins
                )
                result = convert_fn(list(mins), list(maxs), max_data_bin)
                # Rust PyO3 method returns None on success; mock objects return Mock.
                if result is None:
                    self._float_thresholds_converted = True
            elif strategy == "quantile":
                convert_fn = getattr(
                    self._native_predictor_handle,
                    "convert_thresholds_to_float_quantile",
                    None,
                )
                if not callable(convert_fn):
                    return
                cuts = self._continuous_feature_quantile_cuts
                if cuts is None:
                    return
                # Convert list[list[float]] → list[list[f32]] for Rust
                result = convert_fn(
                    [[float(v) for v in c] for c in cuts]
                )
                if result is None:
                    self._float_thresholds_converted = True
        except Exception:
            self._float_thresholds_converted = False
