"""Parser and path-reconstruction checks for classification artifacts."""

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accuracy_depth_leaf_diagnostics import (  # noqa: E402
    _child_outputs,
    _diagnostic_prediction_rows,
    _distribution,
    _reconstructed_predictions,
    parse_artifact,
    summarize_artifact,
)


def _stump(node_id, feature, threshold, left, right, left_rows, right_rows):
    return struct.pack(
        "<IIHHfffII",
        node_id,
        feature,
        threshold,
        0,
        1.0,
        left,
        right,
        left_rows,
        right_rows,
    )


def _artifact(section_kind, payload, objective, feature_count, class_count=None):
    metadata = json.dumps(
        {
            "format_version": 1,
            "feature_names": [f"f{i}" for i in range(feature_count)],
            "trained_device": "cpu",
            "objective": objective,
            "num_classes": class_count,
        },
        separators=(",", ":"),
    ).encode()
    offset = 16 + 20 + len(metadata)
    header = struct.pack("<4sIII", b"AGBM", 1, 1, len(metadata))
    descriptor = struct.pack("<IQQ", section_kind, offset, len(payload))
    return header + descriptor + metadata + payload


def _scalar_artifact():
    payload = struct.pack("<IIIf", 1, 1, 2, 0.5)
    payload += _stump(0, 0, 0, 0.75, -0.25, 4, 6)
    payload += _stump(2, 0, 0, 0.25, -1.0, 2, 4)
    return _artifact(1, payload, "binary_crossentropy", 1)


def _multiclass_artifact():
    payload = struct.pack("<III", 1, 2, 1)
    payload += struct.pack("<2f", 0.0, 0.0)
    payload += struct.pack("<2I", 1, 1)
    payload += _stump(0, 0, 0, 1.0, -1.0, 2, 2)
    payload += _stump(0, 0, 0, -0.5, 0.5, 2, 2)
    return _artifact(6, payload, "multiclass_softmax", 1, class_count=2)


def test_scalar_path_deltas_reconstruct_absolute_terminal_and_child_outputs():
    parsed = parse_artifact(_scalar_artifact())
    outputs = _child_outputs(parsed)
    assert [(item.child_node_id, item.value, item.terminal) for item in outputs] == [
        (1, 0.75, True),
        (2, -0.25, False),
        (5, 0.0, True),
        (6, -1.25, True),
    ]
    summary = summarize_artifact(parsed, caps=(1.0,))
    assert summary["distributions_by_cap"]["1"]["terminal_outputs"]["abs_gt_cap_count"] == 1
    assert summary["distributions_by_cap"]["1"]["accepted_children"]["abs_gt_cap_count"] == 1


def test_scalar_reconstructed_probability_follows_quantized_branch():
    parsed = parse_artifact(_scalar_artifact())
    prediction = _reconstructed_predictions(parsed, np.asarray([[0.0], [1.0]], dtype=np.float32))
    expected = 1.0 / (1.0 + np.exp(-np.asarray([1.25, -0.75])))
    np.testing.assert_allclose(prediction[:, 1], expected, rtol=0, atol=1e-7)


def test_multiclass_parser_keeps_class_tree_ids_distinct():
    parsed = parse_artifact(_multiclass_artifact())
    assert set(parsed.trees) == {(0, 0), (1, 0)}
    assert len(_child_outputs(parsed)) == 4
    predicted = _reconstructed_predictions(parsed, np.asarray([[0.0], [1.0]], dtype=np.float32))
    np.testing.assert_allclose(predicted.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert predicted[0, 0] > predicted[0, 1]
    assert predicted[1, 0] < predicted[1, 1]


def test_prebinned_rows_are_used_directly_without_continuous_bounds():
    class PrebinnedModel:
        _uses_continuous_binning = False

        @staticmethod
        def _validate_numeric_features(rows):
            return np.asarray(rows, dtype=np.float32)

        @staticmethod
        def _quantize_rows_for_prediction(_rows):
            raise AssertionError("pre-binned data must bypass continuous quantization")

    rows = _diagnostic_prediction_rows(PrebinnedModel(), [[0], [1], [7]])
    np.testing.assert_array_equal(rows[:, 0], [0.0, 1.0, 7.0])


def test_sub_tolerance_leaf_roundoff_is_separate_from_substantive_exceedance():
    counts = _distribution([1.0000004], cap=1.0)
    assert counts["abs_gt_cap_count"] == 1
    assert counts["abs_gt_cap_substantive_count"] == 0
