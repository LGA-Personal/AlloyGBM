# AlloyGBM v0.9 Hardening Matrix Baseline (`v0.8.1`)

## Purpose
- Lock the release hardening matrix before additional `v0.9` implementation slices.
- Establish baseline non-regression commitments from `v0.8`.
- Define unresolved hardening buckets and intended follow-on slice ownership.

## Baseline Evidence Sources
- Parent `v0.8` closeout:
  - [docs/architecture/v1.0/v0.8/verification_report.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/verification_report.md)
  - [docs/architecture/v1.0/v0.8/implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/implementation_notes.md)
- Child closeout command evidence:
  - [docs/architecture/v1.0/v0.8/v0.7.5/verification_report.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.5/verification_report.md)
- CI gate contract:
  - [/.github/workflows/ci.yml](/Users/lashby/Projects/AlloyGBM/.github/workflows/ci.yml)

## Non-Regression Baseline Commitments
1. Preserve SHAP behavior and APIs introduced in `v0.8` (`shap_values`, SHAP-based feature importance path).
2. Preserve artifact compatibility behavior for strict and legacy-supported layouts.
3. Preserve Python estimator contract behavior for numeric-only and categorical-capable flows.
4. Preserve deterministic gate outcomes under fixed inputs unless explicitly re-scoped in a later plan.

## Release Gate Matrix
| Gate | Command / Check | Baseline Evidence | `v0.8.1` Status |
| --- | --- | --- | --- |
| Formatting | `cargo fmt -- --check` | `v0.7.5` verification PASS | Planned for rerun in `v0.8.1` verification |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` | `v0.7.5` verification PASS | Planned for rerun in `v0.8.1` verification |
| Rust tests | `cargo test --workspace` | `v0.7.5` verification PASS | Planned for rerun in `v0.8.1` verification |
| Docs build | `cargo doc --workspace --no-deps` | `v0.7.5` verification PASS | Planned for rerun in `v0.8.1` verification |
| Python tests | `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` | `v0.7.5` verification PASS (`Ran 67 tests`) | Planned for rerun in `v0.8.1` verification |
| SHAP additivity/parity | Artifact-backed SHAP additivity + predictor parity checks | `v0.7.3` and `v0.7.4` verification PASS | Baseline locked; non-regression target |
| Artifact compatibility | strict/legacy artifact compatibility behavior | `v0.7.3` and parent `v0.8` verification PASS | Baseline locked; non-regression target |
| Python contract | `GBMRegressor` fit/predict and SHAP contract behavior | parent `v0.8` verification + CI smoke | Baseline locked; non-regression target |

## Open Hardening Buckets for `v0.8.2+`
| Bucket | Description | Planned Child Slice |
| --- | --- | --- |
| Test-gap closure | Expand deterministic edge-case coverage tied to compatibility/error semantics and cross-boundary contract edges. | `v0.8.2` |
| Benchmark reproducibility | Define command protocol, environment capture, and run-to-run evidence packaging for benchmark repeatability. | `v0.8.3` |
| Migration and compatibility narrative | Finalize migration/checklist guidance and traceability package used for `1.0.0` go/no-go review. | `v0.8.4` |

## Operational Notes
- `v0.8.1` is documentation/state focused; production code changes are not planned in this slice.
- Any regression discovered during rerun gates should be treated as a blocker and resolved before advancing to `v0.8.2`.
