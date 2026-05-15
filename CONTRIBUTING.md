# Contributing to AlloyGBM

Thanks for your interest. This doc covers what you need to know to make
useful contributions: project structure, dev setup, expected workflow,
and the gates a PR has to pass before it lands.

## Before opening a PR

- **Discuss bigger changes first.** Bug fixes and small enhancements are
  fine to PR directly. Net-new features, API changes, or anything that
  touches the public Python surface should start as an
  [issue](https://github.com/LGA-Personal/AlloyGBM/issues/new/choose) so
  we can sync on scope.
- **Check the roadmap.** [`docs/roadmap/current.md`](docs/roadmap/current.md)
  lists in-flight priorities. [`docs/limitations.md`](docs/limitations.md)
  lists things that are intentionally out of scope today.

## Project structure

See [`CLAUDE.md`](CLAUDE.md) for a guide to the crate layout, key
architectural patterns, and conventions. (`CLAUDE.md` is written for
agent-flavored work, but the structure section applies to every
contributor.)

The repo is:

- **Rust workspace** (6 crates under `crates/`) — core data structures,
  the training engine, the CPU backend, the predictor, TreeSHAP, and
  categorical encoding.
- **Python bindings** (`bindings/python/`) — PyO3 bridge plus
  sklearn-compatible estimators (`GBMRegressor`, `GBMClassifier`,
  `GBMRanker`, `MultiLabelGBMRanker`).
- **Docs** (`docs/`) — user-facing Markdown under `docs/user/` (mirrored
  by the Sphinx site under `docs/site/source/`), release history in
  `CHANGELOG.md` and `docs/site/source/release.rst`, current roadmap in
  `docs/roadmap/`.
- **Benchmarks** (`benchmarks/`) — cross-library comparison harness.

## Dev setup

You need Rust 1.92+ (pinned via `rust-toolchain.toml`) and Python 3.11+.

```bash
# Rust toolchain — auto-installed from rust-toolchain.toml on first cargo run
rustup show

# Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin
pip install pytest pytest-subtests numpy scikit-learn pandas

# Build the Python extension in-place
maturin develop --release
```

After that, `import alloygbm` should work from `.venv/bin/python`.

## Running tests

```bash
# Rust
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo fmt --all --check

# Python
.venv/bin/python -m pytest bindings/python/tests/ -q
```

All four Rust commands and the Python test suite must pass before a PR
is mergeable. CI runs the same gates.

## Coding standards

### Rust

- `unsafe_code = "forbid"` is enforced at the workspace level. No `unsafe`
  blocks anywhere — use `wide` for SIMD, `bytemuck` for transmutes if
  needed.
- Edition 2024, MSRV 1.92.0.
- `cargo fmt` is the formatter. `rustfmt.toml` lives at the repo root if
  config drift becomes an issue.
- `cargo clippy --all-targets -- -D warnings` must pass. If a lint is
  legitimately wrong for a specific call site, `#[allow]` it inline with
  a comment explaining why.
- Add tests next to the code they cover (`#[cfg(test)] mod tests`).
  Aim for a unit test per non-trivial branch.

### Python

- sklearn-compatible: every estimator inherits from `BaseEstimator` and
  the appropriate mixin (`RegressorMixin` / `ClassifierMixin`).
- `__init__` accepts no positional args; every parameter must show up in
  `get_params`, `set_params`, `_params_order`, and `__repr__`.
- New parameters must validate inputs in `__init__` *before* fit-time —
  bad params raise `ValueError` early, not deep in a native call.

### Adding fields to structs

When you add a field to `TrainParams`, `IterationControls`, or similar
serialized structs:

1. Add the field at the end (positional JSON parser is brittle).
2. Add a `Default` value that preserves current behavior.
3. Add validation in `validate_train_params` (or equivalent).
4. Add a corresponding Python parameter and thread it through
   `_resolve_*` helpers.

### Adding a new objective

1. Implement `ObjectiveOps` in `crates/engine/src/lib.rs`.
2. Add a variant to the objective dispatch.
3. Add a post-transform entry to the predictor table.
4. Add Python-side estimator support (a new estimator class or a new
   `ranking_objective`/`objective` value).
5. Add a benchmark arm under `benchmarks/run_model_comparison.py` if
   it's a generally useful objective.

## Commit hygiene

- One logical change per commit. Don't squash unrelated work.
- Commit messages: imperative mood (`add interaction constraints`, not
  `added interaction constraints`), short subject line (<72 chars), body
  explains *why* not *what* if the diff doesn't speak for itself.
- Conventional Commits style (`feat:`, `fix:`, `chore:`, `docs:`,
  `refactor:`, `test:`) is appreciated but not strictly required.

## Documentation updates

If your change is user-visible (new parameter, new behavior, removed
restriction, new estimator) you must update docs in lockstep with the
code. The release checklist
([`docs/reference/release_checklist.md`](docs/reference/release_checklist.md))
has the authoritative list of files to touch.

The minimum:

- The relevant `docs/user/*.md` page (e.g. `gbmregressor.md` for
  regressor params).
- Its Sphinx mirror under `docs/site/source/*.rst`.
- `README.md` if the change widens the project's "Feature Summary" or
  resolves a "Current Limitations" item.
- `CHANGELOG.md` under the next-release section.

## CI and merge gates

PRs against `main` run:

- Rust check / clippy / fmt / test / doc on Ubuntu + macOS
- Python smoke + full pytest suite on Ubuntu + macOS × Python 3.11 / 3.12 / 3.13

All checks must be green for a PR to be mergeable. If CI flakes (macOS
rustup-init has historically been flaky), comment on the PR and we'll
rerun.

## Releases

If you're cutting a release: follow
[`docs/reference/release_checklist.md`](docs/reference/release_checklist.md)
top to bottom. It's the only authoritative version-bump and
doc-update inventory; don't skip it.

## Reporting security issues

See [`SECURITY.md`](SECURITY.md). Do **not** open a public issue for
suspected vulnerabilities.

## License

By contributing, you agree your work will be licensed under the MIT
License (see [`LICENSE`](LICENSE)).
