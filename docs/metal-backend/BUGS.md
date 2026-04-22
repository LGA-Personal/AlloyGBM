# Metal Backend — Bug / Issue Tracker

Log issues as you discover them. Don't let anything slip into "I'll fix it later in my head."

## Severity legend
- **blocker** — must fix before the current sub-task is "done"
- **high** — must fix before the current stage is done
- **medium** — must fix before the next release
- **low** — nice-to-have / polish

## Status legend
- **open** — not started
- **in-progress** — actively being fixed
- **fixed** — fix is committed, awaiting verification
- **verified** — fixed and confirmed (include commit SHA)
- **wontfix** — explicit decision not to fix (note rationale in `DECISIONS.md`)

---

## Open

| ID | Severity | Opened | Stage | Symptom | Repro | Owner | Notes |
|----|----------|--------|-------|---------|-------|-------|-------|
| B-001 | blocker | 2026-04-21 | S3.7c | `partition.metal` fails library compile via Python-ext (release build) with "program scope variable must reside in constant address space" on the 4 file-scope `threadgroup` arrays (lines 166-169). `cargo test` (debug + release) compiles the same source cleanly, so the bug is context-dependent — likely differs on MSL language-version default between test-harness vs. dylib-loaded Metal context. MetalBackend init falls back to CPU; every Stage-3 Python test that asserts `trained_device == "metal"` fails. | `maturin develop --release --manifest-path bindings/python/Cargo.toml && .venv/bin/python -m pytest bindings/python/tests/test_metal_backend.py::MetalRegressionTests::test_metal_regression_records_trained_device -q` | Claude | Pre-existing on HEAD `4cf7d06`; introduced in `281ae1a` when partition.metal shipped. Fix: move the 4 file-scope `threadgroup` arrays into each kernel body (MSL spec is explicit — `threadgroup` is only valid as a local or parameter). Declaring per-kernel is fine because threadgroup memory isn't shared across kernels regardless. |

---

## Resolved

| ID | Severity | Opened | Resolved | Stage | Summary | Fix / Commit |
|----|----------|--------|----------|-------|---------|--------------|
| — | — | — | — | — | _none yet_ | — |
