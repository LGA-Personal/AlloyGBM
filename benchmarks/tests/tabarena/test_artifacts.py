from copy import deepcopy

import numpy as np
import pytest

from benchmarks.tabarena.artifacts import validate_result


def valid_result():
    return {
        "metric_error": 0.1,
        "validation_protocol": {
            "key": "8x1",
            "protocol": {"enforced": True},
            "num_bag_folds_resolved": 8,
            "num_bag_sets_resolved": 1,
            "num_bag_folds_fitted": 8,
            "num_bag_sets_fitted": 1,
            "num_child_models": 8,
            "custom_splits": False,
        },
        "simulation_artifacts": {
            "y_val_idx": [0, 1],
            "y_test_idx": [2, 3],
            "y_val": [0, 1],
            "y_test": [0, 1],
            "pred_proba_dict_val": {"model": np.array([0.1, 0.9])},
            "pred_proba_dict_test": {"model": np.array([0.2, 0.8])},
        },
    }


def test_requires_complete_standard_protocol_and_disjoint_rows():
    original = valid_result()
    validate_result(original)
    for key, bad in [("num_child_models", 7), ("key", "1x1"), ("custom_splits", True)]:
        changed = deepcopy(original)
        changed["validation_protocol"][key] = bad
        with pytest.raises(ValueError, match="8x1"):
            validate_result(changed)
    changed = deepcopy(original)
    changed["simulation_artifacts"]["y_test_idx"] = [0, 3]
    with pytest.raises(ValueError, match="overlap"):
        validate_result(changed)
    changed = deepcopy(original)
    changed["simulation_artifacts"]["pred_proba_dict_val"]["model"][0] = np.nan
    with pytest.raises(ValueError, match="predictions"):
        validate_result(changed)
