import subprocess
from pathlib import Path

from benchmarks.tabarena.prepare_upstream import prepare, verify


def test_payload_and_protocol_integrity(tmp_path):
    # Use a local git source so this contract never downloads data or needs network.
    source = Path(__file__).resolve().parents[3] / ".cache/tabarena/upstream"
    if not source.exists():
        import pytest

        pytest.skip(
            "live clone fixture absent; CI uses the pinned upstream integration job"
        )
    clone = tmp_path / "upstream"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(source), str(clone)], check=True
    )
    prepare(clone)
    verify(clone)
    target = clone / "packages/tabarena/src/tabarena/benchmark/validation_protocol.py"
    target.write_text(target.read_text() + "\n# tamper\n")
    import pytest

    with pytest.raises(ValueError, match="upstream"):
        verify(clone)


def test_registered_dependency_and_family_match_submission():
    import tomllib

    from tabarena.models import AlloyGBMModel, get_model_registry
    from tabarena.models.alloygbm.info import ALLOYGBM_REQUIREMENT
    from tabarena.website.website_format import Constants

    info = get_model_registry()["AlloyGBM"]
    assert info.model_cls is AlloyGBMModel
    assert info.pip_extra == (ALLOYGBM_REQUIREMENT,)
    source = Path(__file__).resolve().parents[3] / ".cache/tabarena/upstream"
    extras = tomllib.loads((source / "packages/tabarena/pyproject.toml").read_text())[
        "project"
    ]["optional-dependencies"]
    assert extras["alloygbm"] == [ALLOYGBM_REQUIREMENT]
    assert "tabarena[alloygbm]" in extras["extended"]
    # Same registration used by upstream website result formatting.
    from tabarena.website.website_format import get_model_family

    assert get_model_family("ALLOYGBM") == Constants.tree


def test_unknown_untracked_upstream_file_is_rejected(tmp_path):
    source = Path(__file__).resolve().parents[3] / ".cache/tabarena/upstream"
    clone = tmp_path / "upstream"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(source), str(clone)], check=True
    )
    prepare(clone)
    target = clone / "unexpected-protocol.py"
    assert not target.exists()
    try:
        target.write_text("# unexpected additional code\n")
        import pytest

        with pytest.raises(ValueError, match="untracked"):
            verify(clone)
    finally:
        target.unlink()
