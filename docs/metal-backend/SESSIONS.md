# Metal Backend — Session Log

Append-only. One entry per working session. Newest entries at the top.
First thing a new session reads, alongside `STATUS.md`.

---

## 2026-04-18 — S1.2 device + capability probe

**Branch:** `claude/charming-carson-d08c9a` (worktree)

**What moved:**
- Added `objc2 = "0.6"`, `objc2-foundation = "0.3"`, `objc2-metal = "0.3"`
  under `[target.'cfg(target_os = "macos")'.dependencies]`.
- Dropped workspace-inherited lints on `backend_metal`; declared local
  `[lints.rust] unsafe_code = "deny"` so FFI sites opt in per-site.
  Recorded this deviation as **D-007** in `DECISIONS.md`.
- `src/device.rs`: `MetalCapabilities { apple7, metal4, device_name }`
  and `MetalDevice { device, queue, capabilities }`; `MetalDevice::probe()`
  calls `MTLCreateSystemDefaultDevice`, opens a command queue,
  reads `supportsFamily(MTLGPUFamily::Apple7)`, and probes Metal 4 via
  `msg_send![device, supportsFamily: 5002isize]` (raw NSInteger to stay
  forward-compatible with `objc2-metal` 0.3 which may not yet expose the
  Metal 4 variant in its enum).
- `src/lib.rs`: `MetalBackend::new()` wraps `MetalDevice::probe()` and
  rejects devices that don't support Apple7. Stubbed on non-macOS so the
  workspace builds cross-platform.
- Added a smoke test (`tests::probe_default_device`) that calls
  `MetalBackend::new()` on macOS and asserts Apple7 + non-empty device
  name. Passes locally on M-series hardware.

**Commits shipped:** (committed at session end — see git log)

**Verification:**
- `cargo check --workspace`: green.
- `cargo clippy -p alloygbm-backend-metal --all-targets -- -D warnings`: clean.
- `cargo fmt --all --check`: clean.
- `cargo test --workspace --exclude alloygbm-python`: 177 passed, 0 failed,
  including the new Metal probe test.

**Notes:**
- `cargo test --workspace` (without excluding `alloygbm-python`) fails at
  link with missing `_Py_DecRef`/`_Py_IncRef` etc. This is pre-existing
  — `alloygbm-python` is a `cdylib` tested via `maturin develop` + pytest,
  not via `cargo test`. Not a regression.

**Blockers:** none.

**Next session should:** start **S1.3** (MSL histogram kernel). Write
`crates/backend_metal/src/shaders/histogram.metal` implementing
privatized-threadgroup histograms + deterministic tree-reduce, embed via
`include_str!` from `src/kernels/histogram.rs`. Keep the Rust module
pure-source-for-now; actual pipeline compilation + dispatch arrive in
S1.4/S1.5.

---

## 2026-04-18 — S1.1 scaffold

**Branch:** `claude/charming-carson-d08c9a` (worktree)

**What moved:**
- Created `crates/backend_metal` crate: `Cargo.toml` (workspace-inherited
  metadata; deps on `alloygbm-core`, `alloygbm-engine`, `alloygbm-backend-cpu`),
  minimal `build.rs` (no-op; framework linking lands in S1.2),
  `src/lib.rs` with a stub `MetalBackend` unit struct.
- Added `crates/backend_metal` to workspace `members` in root `Cargo.toml`.
- Wired `bindings/python/Cargo.toml`: optional `alloygbm-backend-metal`
  under `[target.'cfg(target_os = "macos")'.dependencies]`, `metal` feature
  default-on via `dep:alloygbm-backend-metal`.
- Verification: `cargo check --workspace` green in 5.79s. `cargo clippy
  -p alloygbm-backend-metal --all-targets -- -D warnings` clean.
  `cargo fmt --all --check` clean.

**Commits shipped:** (committed at session end — see git log for SHA)

**Blockers:** none.

**Next session should:** start **S1.2** (device probe). Add `objc2` +
`objc2-metal` deps, extend `build.rs` with framework linking, create
`src/device.rs` that probes `MTLCreateSystemDefaultDevice` and family
flags (`MTLGPUFamilyApple7`, `MTLGPUFamilyMetal4`), and thread device +
command queue + capability flags onto `MetalBackend`. Keep `MetalBackend`
still not implementing `BackendOps` — that arrives in S1.4.

---

## 2026-04-18 — Planning session

**Branch:** `claude/charming-carson-d08c9a` (worktree)

**What moved:**
- Confirmed MLX was the wrong foundation (NotebookLM MLX Expert: `scatter_add`
  non-deterministic, macOS 14+/Apple-Silicon-only distribution, forces MSL anyway).
- Confirmed raw-Metal design with 3 rounds against NotebookLM Metal 4 Expert
  (sessions `df440836` MLX, `09f9a81e` Metal 4). Validated: no float atomics,
  two-pass deterministic reduce, level-parallel dispatch, `MTLResidencySet`
  pattern, runtime MSL compile + pipeline harvesting cache, ~250k-row
  breakeven, 4-5× decisive win >1M rows × 100 features.
- Wrote and approved the Stage 1 plan
  (see `/Users/lashby/.claude/plans/okay-add-this-notebook-structured-star.md`).
- User decisions locked: Metal 3 baseline + Metal 4 fast path; full 4-stage
  plan with Stage 1 in scope; cargo feature `metal` default-on for macOS.
- Created this progress-tracking scaffold (`STATUS.md`, `SESSIONS.md`,
  `BUGS.md`, `DECISIONS.md`) and CLAUDE.md anchor.

**Commits shipped:** _(scaffold only — no Rust code yet)_

**Blockers:** none.

**Next session should:** read `STATUS.md`, then start **S1.1** (scaffold
`crates/backend_metal` + workspace wiring + `cargo check --workspace` green)
as a single small commit. Update `STATUS.md` and append here before ending.

---
