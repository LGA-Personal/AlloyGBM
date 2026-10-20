"""Frozen author search space v1; never selected from outer test scores."""

from __future__ import annotations

from tabarena.models.alloygbm.model import AlloyGBMModel
from tabarena.utils.config_utils import CustomAGConfigGenerator


def sample_alloy_configs(n: int, seed: int = 1234) -> list[dict]:
    from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

    space = ConfigurationSpace(
        seed=seed,
        space=[
            Float("learning_rate", (0.01, 0.2), log=True),
            Integer("max_depth", (3, 10)),
            Integer("min_data_in_leaf", (1, 128), log=True),
            Float("row_subsample", (0.6, 1.0)),
            Float("col_subsample", (0.6, 1.0)),
            Categorical("lambda_l1", [0.0, 0.01, 0.1, 1.0, 10.0]),
            Categorical("lambda_l2", [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]),
            Categorical("continuous_binning_max_bins", [64, 128, 256]),
        ],
    )
    if n < 0:
        raise ValueError("configuration count must be nonnegative")
    configs = []
    for _ in range(n):
        sampled = dict(space.sample_configuration())
        sampled = {k: v.item() if hasattr(v, "item") else v for k, v in sampled.items()}
        sampled.update(
            training_policy="manual",
            training_mode="manual",
            n_estimators=10000,
            early_stopping_rounds=50,
            tree_growth="level",
            leaf_model="constant",
            boosting_mode="standard",
        )
        configs.append(sampled)
    return configs


gen_alloygbm = CustomAGConfigGenerator(
    model_cls=AlloyGBMModel, manual_configs=[{}], search_space_func=sample_alloy_configs
)
