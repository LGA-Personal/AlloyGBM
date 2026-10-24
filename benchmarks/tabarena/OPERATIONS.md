# Running and recovering AlloyGBM benchmarks

Read [README.md](README.md) for statistical protocol and [DROPLET.md](DROPLET.md)
for provisioning. The reusable agent skill lives at
[.agents/skills/alloygbm-tabarena](../../.agents/skills/alloygbm-tabarena/SKILL.md).
Install that directory under `~/.codex/skills/` for `$alloygbm-tabarena` discovery.

## Select the scope before renting a machine

All counts below are for **AlloyGBM alone**, including the public default.
Competitors come from published results; they are not refitted.

| Profile | Outer splits | Sampled candidates | Jobs | Inner child fits |
|---|---:|---:|---:|---:|
| `lite-defaults` | 51 | 0 | 51 | 408 |
| `lite-hpo` | 51 | 25 | 1,326 | 10,608 |
| `lite-200` | 51 | 200 | 10,251 | 82,008 |
| `full` | 816 | 200 | 164,016 | 1,312,128 |

Full has 30 outer splits on 17 datasets and nine on 34. Lite uses the first
official split of each dataset; both retain eight inner folds. Do not confuse
outer coverage with tuning budget. The published LightGBM, XGBoost, CatBoost and
RF artifacts each have 201 configuration results on every Lite dataset:
one default plus 200 random candidates. Default rows use the default alone;
tuned and tuned+ensemble use standard validation selection. Models with other
search spaces and end-to-end systems need their own source-budget inspection.

The earlier 25-candidate Lite run completed all 1,326 jobs, passed acceptance
and resource checks, and was exported and verified before its VM was deleted.
Its total allocation cost was about $1.61. Using those per-dataset measured
runtimes, Lite-200 projects to about 18 hours of fitting before reuse; the full
200-candidate grid projects to about 226 hours. These estimates cover v1 only;
the special modes require a pilot. Allocation also bills setup/export/idle time.
Current authorized follow-up: **Lite-200, modes-v2, 27-hour total allocation cap**
(about $13.50 at $0.50/hour), with early cleanup after verified completion.
This historical authorization is not permission for unrelated future rentals.

## Search-space versions

`--search-space v1` is the original frozen eight-parameter random search.
`--search-space modes-v2` preserves candidates 1–25 exactly, then candidates
26–200 use the same base parameter stream with independently sampled toggles:

| Training mode | Leaf model | Leaf solver |
|---|---|---|
| manual or MorphBoost | constant | standard |
| manual or MorphBoost | constant | DRO |
| manual or MorphBoost | linear (PL) | standard |

This permits all six combinations, and never linear+DRO. Mode-specific knobs
keep public defaults (for example DRO radius and Morph rate); the existing tree,
sampling and regularization search still varies. All six combinations have
actual adapter fit tests for regression, binary and multiclass classification.
This space was requested after inspecting v1 Lite results; disclose that it is
a new author space, rather than a blind replication. Do not keep changing the
space or choose candidates after looking at the new outer test scores.

The public default is still `{}`. The official bundle applies seeds, 8x1 folds,
preprocessing and validation-based selection. Neither the mode toggle nor reuse
changes the standard task metric. Search sampling is 1234; preserve the upstream
training/fold seed schedule, including any numerical seed that happens to be 42.

## Prepare, reuse and verify before fitting

On a clean, committed Ubuntu checkout, use the bootstrap in DROPLET.md. Run it
inside tmux with a foreground timeout and noninteractive apt, for example:

```bash
timeout --foreground --signal=INT --kill-after=120s 2h \
  bash benchmarks/tabarena/remote/bootstrap-ubuntu.sh </dev/null
```

Restore the verified old evidence into a separate directory, then copy its
`cache/` into `.cache/tabarena/cache/`. This includes the official OpenML data
and split files, avoiding fresh network downloads. Use raw OpenML XML/ARFF/parquet
cache files; never manufacture splits. Retain checksums and source receipts.
Build the current selected source in release mode; never bless an old binary
with a freshly issued receipt.

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.run \
  --profile lite-200 --search-space modes-v2 --threads 16 --models AlloyGBM \
  --output-root benchmarks/results/tabarena/cloud-lite-200 --plan-only
# Use the printed directory as RUN_DIR. OLD_RUN is extracted evidence's run/.
.venv-tabarena/bin/python -m benchmarks.tabarena.import_fits OLD_RUN RUN_DIR
.venv-tabarena/bin/python -m benchmarks.tabarena.checkpoints create RUN_DIR
```

Import compares model source, wrapper, upstream, dependencies and build settings,
and matches the **entire serialized job**, including config/seed/split/resources.
Changed jobs are not reused. Old manifests, binary hashes, hardware, ledger and
historical failures remain in `origins/`; imported rows reference that origin.
An identity-changing rebuild/host replacement requires this explicit import into
a new plan, not `--resume` bypasses. Imported timings describe the original host.
The original full Lite archive can supply 1,326 fits for this new scope; validate
the imported count. Preserve the old archive independently.

## Periodic verified backups

Each fit's result and auxiliary files are synced before its completed event is
appended and fsynced to `jobs.jsonl`. Interrupted writers' unacknowledged caches
are quarantined, then that job may be rerun. Completed checksummed fits are
validated and skipped. The run lock prevents two fit/report drivers racing.

The runner creates incremental checkpoints after **50 attempted jobs or 15
minutes**, checked between jobs, plus on a normal stage exit. A long in-flight
job can delay the next checkpoint up to its official fit deadline. A killed
process can lose only unacknowledged/in-flight work, not previous checkpoints.
Each immutable archive has a SHA256 receipt, per-file hashes, predecessor hash,
new result caches/logs and the ledger delta. The first checkpoint also contains
plan/config/task/manifest and fit-origin receipts. Keep the whole chain.
Archives are published before updating the local index, and recovery reconciles
that crash window. A final report is not required to recover raw fits.

On the **local machine**, run this before enabling guarded fitting and then
on each monitoring wake (about five minutes):

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.backup SSH_ALIAS \
  /root/AlloyGBM/benchmarks/results/tabarena/cloud-lite-200/RUN_ID \
  /absolute/local/evidence/checkpoints
```

It incrementally rsyncs published archives, verifies new archive/member hashes
and chain order, then acknowledges the off-server copy over SSH. Failed or
corrupt transfers do not advance acknowledgement. It never deletes remote files.
Keep an initial full cloud export as well: checkpoints contain run results, while
the full bundle also preserves dependency/data caches and source Git history.
Only Python 3.12 and rsync/OpenSSH are needed for backup; no cloud API token.

Test a recovery into a **new** directory before the long run:

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.checkpoints restore \
  /absolute/local/evidence/checkpoints --destination /absolute/new/recovered-run
```

Restore rechecks every archive and completed-result checksum without executing
pickle data. It works independently of the VM or reporting code. A new server
must still rebuild/attest its environment and explicitly import compatible fits.

## Pilot, run, monitor and stop

All new stages use the same immutable Lite-200/modes-v2 plan. Imported completed
fits are skipped; staged filters affect execution only.

```bash
bash benchmarks/tabarena/remote/run-session.sh modes-smoke 2 RUN_DIR
bash benchmarks/tabarena/remote/run-session.sh modes-large 2 RUN_DIR
bash benchmarks/tabarena/remote/run-session.sh lite-200 FIT_HOURS RUN_DIR
```

The pilots cover all three task types, then APSFailure/kddcup09_appetency, through
candidate 40 (including legacy candidates and sampled special modes). Check real
child metadata, mode parameters, eight-fold completion, ParallelLocalFoldFittingStrategy
logs, per-child CPU budgets/native `n_jobs`, vmstat and disk before continuing.
These stages enable `--backup-max-age-seconds 3600`; missing/stale verified backup
acknowledgement stops the next fit. Continue pulling backups during reporting.
A stale guard exit is an operational stop, not a failed model candidate.

A fit-hour cap is not an allocation cap. The operator/controller must record an
absolute provider-creation-based deadline and reserve time for reporting, final
checkpoint, export, transfer and verification. For the current run, stop fitting
with at least two hours left inside the 27-hour allocation. Never silently
extend it because a long mode is slow. Maintain a same-chat heartbeat to pull
backups, inspect failures/resources/progress and billable elapsed time, and
perform authorized deletion at completion or budget stop. Notify on material
changes, failures, needed input, completion or meaningful cost changes.

Read latest statuses by event `key`, retaining all previous events. Inspect each
failure before `--retry-failed`. OpenML HTTP503 is a data-fetch failure, not an
accuracy result; bounded retries using the official loader/cache preserve the
protocol. Do not run another fit driver while the old one is active. Keep raw
logs and examined failures even when a retry succeeds.

## Finalization and cleanup

When the driver is terminal, checkpoint once more and pull/verify it. Generate
`benchmarks.tabarena.report RUN_DIR --published-only`, then run
`benchmarks.tabarena.acceptance RUN_DIR` for complete intended coverage. Latest
failures require explicit explanations; historical failures remain in the ledger.
Lite-200 acceptance requires all 10,251 planned outcomes, not 25-candidate
acceptance. Report code and ledger hashes must match. Partial or imputed outcomes
must never be described as a complete competitive ranking.

Retain standard default/tuned/tuned+ensemble scores, Elo/CI, normalized error,
ranks, win rates, improvability, per-dataset scores, training/inference/Pareto,
selection weights, raw predictions, manifests and origin/source hashes. Elo is
relative to the scoped entrant pool and RF(default) calibration; website pool
or snapshot differences can move its numerical value. Published timings are
mixed-hardware context, not matched speedup measurements.

Export with `benchmarks.tabarena.cloud export`, copy archive **and sidecar** off
server, then `benchmarks.tabarena.cloud verify` locally. Include final acceptance,
operator/resource logs, backup receipts and measured cost. Export failure leaves
all raw caches/checkpoints intact. Never delete the sole remaining evidence copy.
Once the off-host copy is verified, destroy only the recorded authorized droplet,
confirm provider deletion, record end time/cost and remove its monitor. Do this
immediately on early completion or significant pause; powered-off VMs still bill.
Do lengthy PR/documentation work after the VM is deleted.
