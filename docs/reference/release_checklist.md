# Release Checklist

This checklist exists so AlloyGBM releases are repeatable and do not depend on
memory or ad hoc chat notes.

## Current Release Decisions

### Linux wheel policy

For public PyPI releases, AlloyGBM should publish Linux wheels only if they are
built in a proper manylinux-compatible environment.

Current decision:

- do not treat a generic `ubuntu-latest` wheel build as the final Linux release policy
- before public PyPI publication, switch Linux wheel publishing to a
  manylinux-oriented build path
- source distributions are still acceptable as a fallback while that is being
  finalized

### Windows wheel policy

Windows wheels are deferred until after `0.3.0`.

Current decision:

- `0.3.0` targets macOS and Linux first
- Windows wheel support is a later packaging expansion, not a release blocker

## Verified Packaging Checks

The following release-hardening checks should be run locally:

- built a wheel with `maturin build`
- created a fresh virtual environment
- installed the built wheel into that fresh environment
- imported `alloygbm`
- ran `native_runtime_info()`
- trained a small `GBMRegressor`, `GBMClassifier`, and `GBMRanker`
- ran `predict(...)` for all three estimators
- ran `predict_proba(...)` for the classifier
- ran `purged_time_series_splits(...)`
- pickle round-trip for all three estimators
- `save_model` / `load_model` round-trip

The publish workflow also runs wheel smoke tests before the final PyPI publish
step, covering all three estimators.

## Pre-Release Checklist

- confirm `README.md` reflects the current package capabilities and limitations
- confirm user docs under `docs/user/` are up to date
- confirm Sphinx site under `docs/site/` is up to date
- confirm `pyproject.toml` version matches the intended release
- confirm `Cargo.toml` workspace version matches the intended release
- confirm `docs/site/source/conf.py` version matches the intended release
- confirm license file and package metadata are present
- confirm `cargo check --workspace` passes
- confirm `cargo test --workspace` passes
- confirm `cargo clippy --workspace --all-targets -- -D warnings` passes
- confirm Python tests pass (`.venv/bin/python -m pytest bindings/python/tests/ -q`)
- confirm Python smoke CI is green (including classifier and ranker smoke tests)
- confirm a clean wheel install works in a fresh environment
- confirm benchmark messaging is honest about both strengths and weak spots
- confirm the publish workflow and PyPI trusted publisher settings are aligned
- confirm CHANGELOG.md is up to date

## Benchmark Statement For Public Release

The current public-facing benchmark claim covers three task types:

**Regression:**
- AlloyGBM is strongest on `panel_time_series`
- AlloyGBM is strong on `dow_jones_financial`
- AlloyGBM is weaker on `california_housing` and `bike_sharing`

**Classification:**
- AlloyGBM is competitive with established libraries on standard datasets

**Ranking:**
- AlloyGBM competes using native LambdaMART on synthetic ranking scenarios

Do not broaden these claims unless the comparative results change materially.

## Release Steps

1. Update version metadata in `pyproject.toml`, `Cargo.toml`, and `docs/site/source/conf.py`.
2. Update CHANGELOG.md with release notes.
3. Push the final release commit to `main`.
4. Create and push the git tag for the release version.
5. Create the GitHub release with concise release notes.
6. Trigger or verify the publish workflow.
7. Confirm the package appears on PyPI and installs in a clean environment.
8. Record any release-specific notes or regressions before starting the next cycle.
