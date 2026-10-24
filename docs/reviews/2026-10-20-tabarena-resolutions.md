# TabArena PR review follow-up

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|
| 2026-10-20 | Claude Code and Codex GitHub reviews; fixes verified by Codex | v1.0.0 development, PRs #172–174 | `ecbafa1f`, `54b8ae16`, `ce180f66` | Code findings addressed; Linux pilot and full Lite acceptance pending |

## PR #172: fit deadlines

1. **Fixed** in `a8e2da28`: deadline expiration selects a best validation/custom
   metric round only when validation early stopping was enabled. Otherwise it
   retains the completed training prefix. Regression, binary, multiclass
   per-class trees and vector leaves have regression coverage with validation
   deliberately opposed to training. The deadline remains a resource control.
2. **Fixed** in `a8e2da28`: a classifier timeout after label/evaluation preparation
   invalidates the old fitted state, matching the base estimator's timeout
   behavior. A binary-to-multiclass refit test reproduces the original mismatch
   and checks prediction is rejected after the timeout.

## PR #173: standard execution and interpretation

1. **Fixed:** upstream Ray execution is now the default. Native debug/sequential
   execution remains an explicit diagnostic choice. The macOS cross-library
   warmup crash justified process isolation, not treating one-thread sequential
   fits as equivalent to standard compute-limited execution.
2. **Fixed:** retirement acceptance requires Linux x86_64, >=16 available CPUs,
   an allocated CPU budget >=16 within the available allocation, >=60 GiB usable
   RAM (nominal 64 GB), no smaller numerical thread caps, and upstream Ray.
   Provenance records affinity-limited CPU availability and the Linux CPU model.
   This is a project policy based on the pinned upstream CPU partition;
   TabArena itself permits hardware variation. Resource tests reject reduced
   budgets, sequential execution, smaller hosts and hidden thread caps.
3. **Fixed:** the wrapper keeps a usable model returned by the cooperative
   deadline, including a final-round overrun. It still reserves 5% headroom,
   subtracts preprocessing time and maps a no-completed-round timeout to
   `TimeLimitExceeded`. AutoGluon retains responsibility for bag-level limits.
4. **Fixed, with a correction to the review's premise:** AlloyGBM follows
   `self.stopping_metric`. This AutoGluon pin maps ROC AUC evaluation to log-loss
   stopping by default (`AbstractModel._get_default_stopping_metric`). Explicit
   AUC stopping is supported; we do not override that common peer default only
   for AlloyGBM. Callback tests compare raw-margin conversion, metric direction
   and errors against AutoGluon scorers. Regression uses RMSE; multiclass already
   evaluates native softmax log loss. Fixed-round warmup without early stopping
   does not create an unused callback or reject its accuracy scorer. Both
   registered upstream smoke/warmup tests pass.
5. **Fixed:** default-only RandomForest is included in the matched local run.
   Local reporting uses upstream `[Local] RF (default)` calibration and fallback.
   RF must complete every planned task; absent entire methods/variants block
   ranking. Narrowed diagnostic runs without RF require complete coverage.
   A regression test exercises upstream RF imputation of a missing hard task.
   Both comparison contexts are scoped to the **planned** tasks, so a wholly
   failed AlloyGBM dataset cannot disappear before published RF imputation.
6. **Fixed:** published-context registration adds only AlloyGBM; peer reruns stay
   local. Reports record the actual sampled-candidate budget and explain the
   usual 25-vs-200 Lite/published difference. Acceptance checks the local RF
   calibrator, published entrant list, and matching report-code/ledger hashes.
7. **Intentional limitations:** keep the requested public-default cap/policy.
   A fold missing a global class fails explicitly because the public estimator
   cannot safely represent every absent-class case; no fake rows or altered
   training distribution are introduced. A static memory estimate is **deferred**
   to a measured Linux profiling follow-up tracked here: an uncalibrated guess
   could make Ray scheduling less reliable. Upstream permits submission without
   it. Inspect RAM and parallel-fold behavior during the Linux pilot before HPO.

8. **Documented, with reviewed training behavior retained:** the binary/regression
   metric callback uses absolute improvement and configured patience, bypassing
   the native relative-improvement plateau rule. Multiclass retains the native
   rule. “Public defaults” therefore means constructor defaults with the
   framework's validation/stopping overrides, not identical stopping to a direct
   public estimator fit. The README and JSON/Markdown report summary disclose
   this distinction. No native or wrapper training behavior changes for this note.

Adding the RF default makes the complete Lite HPO plan **5,355 outer jobs /
42,840 child fits**: 5,304 booster jobs plus 51 calibration jobs, all using the
enforced 8x1 protocol. RF receives no sampled search configurations.

## PR #174 and disposable infrastructure

Retirement remains a separate draft PR, gated on full Lite evidence on the chosen
Linux resource policy, successful processing and examined failure explanations.
TabArena covers regression/binary/multiclass CPU prediction; it supplies no new
ranking, temporal or special-mode benchmark claims. The historical harness and
its evidence remain archived by that PR; estimator correctness tests remain.

The [droplet runbook](../../benchmarks/tabarena/DROPLET.md) and scripts implement
bootstrap, preflight, explicit fit-time caps, report diagnostics and export.
Export retains raw/processed caches, results, source Git bundle, lock and build
receipt. Off-server verification precedes manual destruction. Shell workflow
tests cover success, failure and timeout exits; archive tests recover source and
predictions after deleting the simulated server copy and detect corruption.
The portable tests also run in an Ubuntu CI job without provisioning resources.

No droplet was created and no cloud charges were incurred for this follow-up.
A fresh Ubuntu bootstrap/Ray pilot and full Lite run remain operational gates.
The wall-clock script cap does not stop provider billing; destroy the droplet
after verifying the downloaded evidence.

## Verification

- Before changes: Rust workspace **909 passed / 3 ignored**; retirement-tree
  Python suite **1,251 passed + 64 subtests**.
- Native fixes: Rust workspace **909 passed / 3 ignored**; integration-tree
  Python suite **1,259 passed + 64 subtests**; deadline-specific tests **27 passed**.
- Integration: **50** benchmark/cloud contracts (final results recorded in the
  execution evidence), plus **2** upstream registered model/warmup tests.
- Rust formatting and Clippy with warnings denied; Python Ruff lint/format;
  shell syntax checks. The Ubuntu bootstrap itself is not claimed executed.
- Real official binary-task workflow at `4d176b4128ab175d`: all **9/9 outer jobs**
  completed with upstream Ray, 2-CPU development budget, four boosters plus RF,
  **72 validated children**. Resume preserved the 18-event ledger without fits.
  Standard local and published reports were generated from caches. This reduced
  macOS run is deliberately ineligible for retirement acceptance and says nothing
  about full-suite competitive rank.

## 2026-10-23: affordable AlloyGBM-only Lite execution

The author approved a dedicated DigitalOcean `c-16` host (16 vCPU, 32 GB RAM,
200 GB SSD; API quote $0.50/hour) and the official upstream seed schedule. The
previous 64-GB gate and mandatory local peer HPO were project choices, not
TabArena submission requirements. Acceptance now requires all 51 AlloyGBM Lite
tasks with default + 25 sampled configurations (1,326 jobs / 10,608 children),
standard published RF calibration/fallback, all three variants, terminal job
coverage and examined failure explanations. The separate retirement PR remains
gated; these preparation tests do not satisfy Lite acceptance.

- Default scope is AlloyGBM only; peer reruns remain optional for matched timing.
- Published-only reporting no longer depends on local RF success. Planned failed
  tasks remain in the upstream comparison, and partial coverage stays explicit.
- Linux uses the standard context in one process, reusing Ray across outer jobs.
  Per-job native/Python diagnostics remain captured. Local/macOS diagnostics
  retain fresh-process isolation.
- Stage selectors execute smoke, large-data and default checks from one frozen
  full Lite plan, so HPO reuses their completed, checksum-verified results.
- Sessions select AlloyGBM, use the full 16-CPU budget, capture `vmstat`, preserve
  stage logs, apply explicit time bounds, and export recoverable evidence.

Validation before provisioning: Rust workspace 909 passed / 3 ignored before
and after; library Python suite 1,259 passed + 64 subtests before and after;
62 benchmark/cloud contracts. Ruff lint/format, shell syntax and patch whitespace
checks pass. The pinned upstream registration/shared-code check passes.

Real local checks: two configurations of the official binary smoke task fitted
16 total children through one reused upstream Ray context. Logs recorded two
parallel workers with one CPU each within the two-CPU diagnostic budget. A
separate real staged CLI run recorded the full 51-task / 1,326-job plan, fitted
only two selected jobs, and resumed with the same four-event ledger (no refits).
Published-only reporting generated all three variants and a 108-row published
leaderboard, retaining published RF calibration/fallback and marking 1,324 jobs
pending. These are workflow checks, not competitive results. A fresh Linux
bootstrap and 16-CPU smoke/large-data pilot remain required before full Lite.
