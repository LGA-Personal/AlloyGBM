"""Shared helpers for the MorphBoost-inspired training profile."""

from __future__ import annotations

import numpy as np


def compute_morph_fingerprint(X, y, *, fast_mode: bool = True):
    """Compute a problem-structure fingerprint for morph-mode parameter defaults.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    fast_mode : bool
        If True, use the paper's fixed heuristic constants.
        If False, compute from data.

    Returns
    -------
    dict with keys: complexity, non_linearity, interaction_strength,
    noise_level, suggested_max_depth.
    """
    if fast_mode:
        return {
            "complexity": 0.2,
            "non_linearity": 0.0,
            "interaction_strength": 0.15,
            "noise_level": 0.1,
            "suggested_max_depth": 8,
        }

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples, n_features = X.shape

    feature_stds = np.std(X, axis=0)
    feature_ranges = np.max(X, axis=0) - np.min(X, axis=0)
    complexity = float(np.mean(feature_stds / (feature_ranges + 1e-10)))

    non_linearity = 0.0
    for i in range(min(5, n_features)):
        col = X[:, i]
        try:
            corr_lin = abs(np.corrcoef(col, y)[0, 1])
            corr_quad = abs(np.corrcoef(col ** 2, y)[0, 1])
            non_linearity = max(non_linearity, float(corr_quad - corr_lin))
        except Exception:
            pass

    interaction_strength = 0.0
    if n_features < 100 and n_samples > 100:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_samples, size=min(50, n_samples), replace=False)
        for i in range(min(5, n_features)):
            for j in range(i + 1, min(5, n_features)):
                try:
                    inter = abs(np.corrcoef(
                        X[sample_idx, i] * X[sample_idx, j], y[sample_idx]
                    )[0, 1])
                    if not np.isnan(inter):
                        interaction_strength = max(interaction_strength, float(inter))
                except Exception:
                    pass

    suggested_max_depth = 10 if complexity > 0.5 else 8

    return {
        "complexity": complexity,
        "non_linearity": non_linearity,
        "interaction_strength": interaction_strength,
        "noise_level": 0.1,
        "suggested_max_depth": suggested_max_depth,
    }


def resolve_lr_schedule(schedule, n_estimators, base_lr, warmup_frac=0.1):
    """Resolve an LR schedule name to a per-iteration float list.

    Mirrors the Rust resolve_lr_schedule for Python-side use.
    """
    n = int(n_estimators)
    if schedule == "constant":
        return [float(base_lr)] * n
    if schedule == "warmup_cosine":
        warmup = max(1, min(n, int(round(warmup_frac * n))))
        out = [base_lr * (i + 1) / warmup for i in range(warmup)]
        remaining = n - warmup
        floor = base_lr * 0.01
        for i in range(remaining):
            progress = i / max(remaining, 1)
            cos = 0.5 * (1.0 + np.cos(np.pi * progress))
            out.append(floor + (base_lr - floor) * cos)
        return out
    raise ValueError(f"unknown lr_schedule: {schedule!r}")


def apply_interaction_importance_bonus(
    base_importances, X, gradients_history=None, max_depth_for_interaction=3
):
    """Apply interaction-aware feature-importance bonus (placeholder for v1).

    Returns base_importances unchanged. Tree-walk implementation is deferred.
    """
    return base_importances


def build_morph_config_dict(
    *,
    morph_rate=0.1,
    evolution_pressure=0.2,
    morph_warmup_iters=5,
    info_score_weight=0.1,
    depth_penalty_base=0.9,
    balance_penalty=True,
    lr_schedule="constant",
    lr_warmup_frac=0.1,
):
    """Build the morph_config dict consumed by the PyO3 bridge."""
    cfg = {
        "morph_rate": float(morph_rate),
        "evolution_pressure": float(evolution_pressure),
        "morph_warmup_iters": int(morph_warmup_iters),
        "info_score_weight": float(info_score_weight),
        "depth_penalty_base": float(depth_penalty_base),
        "balance_penalty": bool(balance_penalty),
        "lr_schedule": str(lr_schedule),
    }
    # lr_warmup_frac is only valid (and required) for the warmup_cosine schedule.
    # Omit it for the constant schedule to avoid a Rust-side validation error.
    if str(lr_schedule) == "warmup_cosine":
        cfg["lr_warmup_frac"] = float(lr_warmup_frac)
    return cfg
