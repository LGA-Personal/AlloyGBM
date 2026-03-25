# Release Checklist

This checklist exists so AlloyGBM releases are repeatable and do not depend on
memory or ad hoc chat notes.

## Current Release Decisions

### Linux wheel policy

For public PyPI releases, AlloyGBM should publish Linux wheels only if they are
built in a proper manylinux-compatible environment.

Current decision:

- do not treat a generic `ubuntu-latest` wheel build as the final Linux release policy
- before first public PyPI publication, switch Linux wheel publishing to a
  manylinux-oriented build path
- source distributions are still acceptable as a fallback while that is being
  finalized

Reason:

- generic Ubuntu runner wheels are not the same thing as broadly portable Linux
  wheels
- public users will reasonably expect Linux wheels on PyPI to install cleanly
  across common environments

### Windows wheel policy

Windows wheels are deferred until after `0.1.0`.

Current decision:

- `0.1.0` targets macOS and Linux first
- Windows wheel support is a later packaging expansion, not a release blocker

Reason:

- current validation and publishing work has focused on macOS and Linux
- delaying Windows reduces first-release surface area and failure modes

## Verified Packaging Checks

The following release-hardening check has already been run locally for `0.1.0`:

- built a wheel with `maturin build`
- created a fresh virtual environment
- installed the built wheel into that fresh environment
- imported `alloygbm`
- ran `native_runtime_info()`
- trained a small `GBMRegressor`
- ran `predict(...)`
- ran `purged_time_series_splits(...)`

That verifies the package works as a clean installed artifact, not only from a
source checkout.

The publish workflow is also expected to run wheel smoke tests before the final
PyPI publish step:

- build the wheel artifact
- install that exact artifact in a fresh job
- import the package
- fit and predict with a small `GBMRegressor`

## Pre-Release Checklist

- confirm `README.md` reflects the current package capabilities and limitations
- confirm user docs under `docs/user/` are up to date
- confirm `pyproject.toml` version matches the intended release
- confirm license file and package metadata are present
- confirm `cargo check --workspace` passes
- confirm `cargo test --workspace` passes
- confirm `cargo clippy --workspace --all-targets -- -D warnings` passes
- confirm Python smoke CI is green
- confirm a clean wheel install works in a fresh environment
- confirm benchmark messaging is honest about both strengths and weak spots
- confirm the publish workflow and PyPI trusted publisher settings are aligned

## Benchmark Statement For Public Release

The current public-facing benchmark claim should stay narrow and defensible:

- AlloyGBM is strongest on `panel_time_series`
- AlloyGBM is strong on `dow_jones_financial`
- AlloyGBM is weaker on `california_housing` and `bike_sharing`

Do not broaden this claim unless the comparative results change materially.

## Release Steps

1. Update version metadata if needed.
2. Push the final release commit to `main`.
3. Create and push the git tag for the release version.
4. Create the GitHub release with concise release notes.
5. Trigger or verify the publish workflow.
6. Confirm the package appears on PyPI and installs in a clean environment.
7. Record any release-specific notes or regressions before starting the next cycle.
