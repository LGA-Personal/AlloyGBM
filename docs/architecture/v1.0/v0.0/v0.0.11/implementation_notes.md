# v0.0.11 Implementation Notes

## Summary of What Was Built
- Updated CI workflow in `.github/workflows/ci.yml` to close `v0.0` acceptance evidence gaps:
  - added Rust docs/build check step: `cargo doc --workspace --no-deps` in the Linux/macOS Rust matrix job.
  - added installed-wheel Python contract smoke step that verifies:
    - `alloygbm.GBMRegressor(...)` constructor is callable
    - invalid parameter update (`learning_rate=0.0`) raises `ValueError`.
- Preserved all existing CI checks (cargo check/fmt/clippy/test, wheel build/install, import smoke).

## Non-Intuitive Decisions
- Decision: run the Python contract smoke after wheel install rather than against source tree.
- Reason: this provides direct evidence for packaged artifact behavior, which is the acceptance target.
- Impact: catches packaging/import regressions that source-tree-only tests can miss.

- Decision: keep the contract smoke minimal and assertion-focused.
- Reason: `v0.0.11` is a closeout layer for evidence and guardrails, not feature expansion.
- Impact: reduces CI brittleness while still proving constructor/validation contract behavior.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.0/v0.0.11/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No Rust crate boundaries or public interfaces changed.
- No Python API surface expansion beyond CI verification behavior.

## Known Gaps Deferred to Next Layer
- `v0.0` parent-layer implementation notes and verification artifacts are still pending at `docs/architecture/v1.0/v0.0/`.
- Historical process gap for `v0.0.1` verification artifact remains unchanged.

## Follow-Up Actions
- Promote to parent-layer closeout (`v0.0`) by consolidating child-layer evidence into parent implementation/verification artifacts.
- Resolve historical `v0.0.1` verification artifact gap if strict artifact completeness is required for release gating.
