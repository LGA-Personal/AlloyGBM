# AlloyGBM v0.8.3 Plan (v0.9 Benchmark Reproducibility Slice)

## Summary
- Goal: execute `v0.8.3` by introducing a dedicated benchmark dataset workspace with repeatable preparation workflows for dense numeric, panel time-series, and histogram-stress scenarios.
- Success criteria:
  - repository gains a structured `benchmarks/` workspace with per-scenario manifest + preparation scripts,
  - UCI-based download preparation is available for realistic benchmarks without authentication,
  - synthetic histogram-stress preparation is available for controlled kernel-pressure experiments.
- Audience: engineers running benchmark reproducibility checks during `v0.9` hardening and `1.0.0` gate preparation.

## Scope
### In Scope
- Create dedicated benchmark workspace:
  - `benchmarks/README.md`
  - `benchmarks/.gitignore`
  - `benchmarks/dense_numeric/manifest.yaml`
  - `benchmarks/dense_numeric/prepare.py`
  - `benchmarks/panel_time_series/manifest.yaml`
  - `benchmarks/panel_time_series/prepare.py`
  - `benchmarks/histogram_stress/manifest.yaml`
  - `benchmarks/histogram_stress/prepare.py`
- Data-source conventions:
  - UCI direct download URLs for `dense_numeric` and `panel_time_series`,
  - synthetic generator for `histogram_stress`.
- Produce layer artifacts:
  - `docs/architecture/v1.0/v0.9/v0.8.3/plan.md`
  - `docs/architecture/v1.0/v0.9/v0.8.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.9/v0.8.3/verification_report.md`
- Update `docs/architecture/state/layer_index.yaml`:
  - mark `v0.8.3` verified,
  - advance next target to `docs/architecture/v1.0/v0.9/v0.8.4`.

### Out of Scope
- Introducing benchmark result baselines or performance threshold assertions in CI.
- Runtime algorithm changes for speed or model quality.
- Parent `v0.9` rollup artifacts (`implementation_notes.md`, `verification_report.md`).

## Interfaces and Types
- New benchmark preparation interfaces:
  - per-scenario `prepare.py` command-line entrypoints,
  - per-scenario `manifest.yaml` files as benchmark metadata contract.
- Existing training/inference interfaces remain unchanged.

Backward-compatibility expectations:
- no public API changes,
- no Rust crate behavior changes,
- no changes to existing benchmark binary interfaces in `crates/backend_cpu/benches`.

## Implementation Sequence
1. Add organized benchmark directory structure and metadata manifests.
2. Implement scenario-specific `prepare.py` scripts with deterministic output conventions.
3. Document usage and generated-data policy in benchmark workspace README.
4. Run verification gates and artifact checks for this slice.
5. Publish implementation/verification artifacts and advance layer index to `v0.8.4`.

## Test Cases and Scenarios
- Structure checks:
  - required `benchmarks/` directories and files exist.
- Script checks:
  - each `prepare.py` parses CLI args and compiles without syntax errors.
  - dense/panel scripts encode UCI direct URL patterns.
- Non-regression gates:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. Organized benchmark workspace is present with `dense_numeric`, `panel_time_series`, and `histogram_stress` scenario folders.
2. Each scenario folder includes `manifest.yaml` and `prepare.py`.
3. `dense_numeric` and `panel_time_series` preparation scripts use UCI direct download URLs with no-auth flow.
4. `histogram_stress` preparation script produces deterministic synthetic data from seed-controlled generation.
5. `docs/architecture/v1.0/v0.9/v0.8.3/implementation_notes.md` is present.
6. `docs/architecture/v1.0/v0.9/v0.8.3/verification_report.md` is present with criterion-to-evidence mapping.
7. `cargo fmt -- --check` passes.
8. `cargo clippy --workspace --all-targets -- -D warnings` passes.
9. `cargo test --workspace` passes.
10. `cargo doc --workspace --no-deps` passes.
11. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
12. `docs/architecture/state/layer_index.yaml` marks `v0.8.3` verified and advances next target to `docs/architecture/v1.0/v0.9/v0.8.4`.

## Risks and Mitigations
- Risk: benchmark preparation scripts depend on network availability.
  - Mitigation: scripts are deterministic and explicit; verification checks syntax/structure while downloads occur only when invoked.
- Risk: benchmark workspace drifts into ad-hoc one-off scripts.
  - Mitigation: enforce per-scenario manifest + prepare contract and shared output convention.
- Risk: generated dataset artifacts accidentally enter git.
  - Mitigation: add `benchmarks/.gitignore` to keep generated data untracked by default.

## Assumptions and Defaults
- Scenario manifests are metadata-first and intentionally lightweight.
- Generated outputs default under `benchmarks/data/<scenario>/`.
- `v0.8.4` remains the next slice for migration/compatibility narrative finalization.
