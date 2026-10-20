import json
from pathlib import Path

import pytest

from benchmarks.tabarena.provenance import (
    collect_provenance,
    fingerprint,
    record_build_receipt,
)


def test_fingerprint_stable_and_identity_sensitive():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_real_build_receipt_and_stale_rejection(tmp_path):
    root = Path(__file__).resolve().parents[3]
    receipt = tmp_path / "receipt.json"
    record_build_receipt(root, receipt)
    found = collect_provenance(root, receipt)
    assert found["extension_sha256"]
    payload = json.loads(receipt.read_text())
    payload["extension_sha256"] = "stale"
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="extension"):
        collect_provenance(root, receipt)


def test_experiment_environment_fails_closed(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    receipt = tmp_path / "receipt.json"
    record_build_receipt(root, receipt)
    monkeypatch.setenv("ALLOYGBM_EXPERIMENT_TEST", "1")
    with pytest.raises(ValueError, match="experiment"):
        collect_provenance(root, receipt)
