# AlloyGBM v0.9.3 Plan (Temporal Leakage Hardening for Benchmarks)

## Summary
- Goal: harden benchmark data preparation and train/test splitting so benchmark quality metrics are not inflated by temporal leakage.
- Success criteria:
  - `panel_time_series` no longer uses same-timestep target duplication,
  - time-aware scenarios use strict timestamp-boundary splits with zero timestamp overlap between train and test,
  - benchmark runner fails fast when a feature is exactly target-equivalent,
  - hardening is covered by automated regression tests and command-backed verification evidence.

## Scope
### In Scope
- Update `benchmarks/panel_time_series/prepare.py` so `target_co_gt` is future-horizon (next timestamp) rather than same-row duplication.
- Update `benchmarks/run_model_comparison.py` to:
  - split timestamped data using unique timestamp boundaries,
  - enforce zero timestamp overlap between train/test partitions,
  - detect and reject target-equivalent feature columns.
- Add regression tests under `benchmarks/tests/` for:
  - panel future-target generation,
  - timestamp-boundary split behavior,
  - target-equivalent feature rejection.
- Update benchmark docs/manifests with leakage-hardening notes.
- Produce `v0.9.3` implementation and verification artifacts.

### Out of Scope
- New benchmark scenarios.
- Model algorithm changes in Rust engine/backends.
- CI policy hard-fail thresholds for benchmark metrics (deferred to later `v0.9.x` work).

## Interfaces and Types
- `panel_time_series` manifest target stays `target_co_gt` for compatibility, but semantics become next-timestep target.
- Runner CLI remains backward-compatible (`--profile-grid`, `--profile`, and single-profile flags unchanged).
- `run_model_comparison.py` internal split contract changes for timestamped scenarios:
  - old: row cutoff after timestamp sort,
  - new: split by unique timestamp sets.
- New benchmark regression tests live under `benchmarks/tests/test_temporal_leakage.py`.

## Implementation Sequence
1. Patch panel preparation to build future-horizon target while preserving deterministic output.
2. Patch benchmark runner with strict timestamp split helper and direct leakage guard.
3. Add regression tests for prep/split/leakage guard.
4. Update benchmark documentation and scenario notes.
5. Execute verification commands and produce `implementation_notes.md` + `verification_report.md`.
6. Advance layer index target from `v0.9.3` to `v0.9.4`.

## Test Cases and Scenarios
- `python3 -m py_compile benchmarks/panel_time_series/prepare.py benchmarks/dow_jones_financial/prepare.py benchmarks/run_model_comparison.py benchmarks/tests/test_temporal_leakage.py`
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'`
- `python3 -B benchmarks/panel_time_series/prepare.py --force-download --output-dir benchmarks/data/panel_time_series`
- `python3 -B benchmarks/dow_jones_financial/prepare.py --force-download --output-dir benchmarks/data/dow_jones_financial`
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7 --scenarios panel_time_series dow_jones_financial`
- Non-regression gates:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.9.3/plan.md` is present and decision-complete.
2. `panel_time_series` prepared dataset target is future-horizon and not globally identical to same-row `co_gt`.
3. Timestamped benchmark scenarios split with zero timestamp overlap between train/test.
4. Benchmark runner rejects target-equivalent feature leakage with explicit error.
5. Regression tests exist for future-target prep, timestamp split integrity, and leakage guard behavior.
6. `benchmarks/README.md` and scenario notes document temporal leakage safeguards.
7. `docs/architecture/v1.0/v0.9/v0.9.3/implementation_notes.md` is present.
8. `docs/architecture/v1.0/v0.9/v0.9.3/verification_report.md` is present with criterion-to-evidence mapping.
9. Standard Rust/Python non-regression commands pass.
10. `docs/architecture/state/layer_index.yaml` marks `v0.9.3` verified and advances target to `docs/architecture/v1.0/v0.9/v0.9.4`.

## Risks and Mitigations
- Risk: future-target conversion in panel data may reduce apparent quality metrics sharply.
  - Mitigation: treat change as correctness improvement; preserve previous artifacts and report new post-hardening evidence explicitly.
- Risk: strict timestamp splits could yield small train/test partitions on tiny datasets.
  - Mitigation: enforce minimum one timestamp on each side and fail fast with clear errors when impossible.
- Risk: leakage guard may false-positive for truly duplicated columns.
  - Mitigation: explicit error message naming offending columns so datasets can be corrected intentionally.

## Assumptions and Defaults
- Benchmark quality metrics must be causally valid for decision-making.
- It is acceptable for post-hardening metrics to worsen if they are more trustworthy.
- Immediate next target after `v0.9.3` verification is `docs/architecture/v1.0/v0.9/v0.9.4`.
