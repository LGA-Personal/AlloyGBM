# Disposable benchmark droplet

Use this when a development change needs a substantive comparison. Keep normal
adapter development and small diagnostic tests local. Nothing in this repository
creates a cloud resource or stores a DigitalOcean token.

## Create only when ready to run

In the [DigitalOcean create panel](https://cloud.digitalocean.com/droplets/new),
choose Ubuntu **24.04 LTS x86_64**, the **dedicated CPU CPU-Optimized `c-16`** plan with
**16 Intel vCPUs / 32 GB RAM / 200 GB SSD / 7 TB transfer**. On 2026-10-23, the
provider API lists $336/month or **$0.50/hour** for this exact plan; verify again
before provisioning. Allow disk space for dependencies, caches and an export copy. Choose an available nearby region and
add your SSH public key. Do not enable backups, snapshots or separate volumes
for this disposable workflow. Check the displayed current rate before creating.

Record the droplet **ID, name, public IP, rate and creation time** locally. The
project acceptance policy requires upstream Ray execution, at least 16 available
CPUs, at least 28 GiB usable RAM on a nominal 32-GB machine, and no smaller thread
caps. This is our reproducible development target, not a universal TabArena
hardware rule. A dedicated VM still does not reproduce the official CPU model.
See [plan selection](https://docs.digitalocean.com/products/droplets/concepts/choosing-a-plan/)
and [creation instructions](https://docs.digitalocean.com/products/droplets/how-to/create/).

The Codex DigitalOcean plugin's small default development image/size is not this
benchmark target. Use the provider console or the installed Codex DigitalOcean app with the exact
provider-verified size slug; do not substitute the plugin’s small default size.
Check account eligibility and available regions before creation; an active account
and credit balance alone do not prove permission to create this size. No cluster or GPU is needed.

## Bootstrap a frozen checkout

Connect with `ssh root@DROPLET_IP`. Replace `BENCHMARK_COMMIT` below with the full
commit SHA you intend to evaluate; use a reviewed branch or merged commit that
contains these scripts. Keep it unchanged through fitting, reporting and export.

```bash
apt-get update && apt-get install -y git ca-certificates
git clone https://github.com/LGA-Personal/AlloyGBM.git
cd AlloyGBM
git fetch origin BENCHMARK_COMMIT
git checkout --detach BENCHMARK_COMMIT
bash benchmarks/tabarena/remote/bootstrap-ubuntu.sh
```

Bootstrap installs system build dependencies, Python 3.12, Rust 1.92.0 and the
locked benchmark environment, prepares the pinned upstream clone, builds AlloyGBM
in release mode, records its build identity, runs contracts and checks resources.
The upstream dependency union is large; Linux's locked PyTorch distribution also
installs CUDA libraries even though these experiments allocate zero GPUs. Allow
time and disk for installation. Do not reinterpret an install failure as a model
benchmark failure.

Start with a clean checkout and no manually exported `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, or `NUMEXPR_NUM_THREADS` caps. Preflight rejects smaller
caps and active AlloyGBM experiment/profiling variables. Do not run other workloads
alongside the benchmark. The runner leaves CPU allocation to upstream; it does
not assign 16 threads to each of eight simultaneous children.

## Run a pilot, then choose the substantive scope

Use tmux so an SSH disconnect does not end the run. The second script argument is
an explicit **whole-hour fit wall-clock cap**, not a dollar budget or deletion
timer. It also allows up to one hour for reporting, plus export time.

```bash
tmux new -s alloy-benchmark
bash benchmarks/tabarena/remote/run-session.sh smoke 2
# Detach: Ctrl-b, then d. Reconnect: tmux attach -t alloy-benchmark
```

The script prints its session folder under `/root/alloygbm-session-*`. Follow its
`run.log`, then inspect the individual `job_logs/`, coverage and report before
running a larger profile. The pilot uses the three official smoke tasks, **AlloyGBM only**, default + one sample:
**6 outer jobs / 48 child fits**. The 51-dataset, 26-configuration full plan is
recorded from the first stage so these successful fits are reused later. It checks execution, not competitiveness.

After a successful pilot, choose an explicit cap using observed fit times and
the current rate. These are examples of caps, not completion estimates:

```bash
# Larger/feature-rich official tasks, default + first sample (4 jobs):
bash benchmarks/tabarena/remote/run-session.sh large 2
# All 51 default configurations (408 child fits), reusing completed defaults:
bash benchmarks/tabarena/remote/run-session.sh lite-defaults 6
# Standard initial author scope: AlloyGBM default + 25 samples across 51 datasets.
# 1,326 outer jobs / 10,608 child fits TOTAL, including earlier stages.
# A 12-hour fitting cap costs at most $6 of fit time at $0.50/hour;
# provisioning, installation, reporting, export and idle allocation also bill.
bash benchmarks/tabarena/remote/run-session.sh lite-hpo 12
```

The session script records fit/report exit codes, keeps failed-job diagnostics,
and attempts an evidence export on normal completion, a fit failure or expiration
of the fit cap. If it is killed externally, runs out of disk, or export fails,
follow the manual export below. Failures require inspection; do not report them
as scores. Reports use `--published-only` with upstream’s published RF calibration/fallback.
Peer libraries are not refitted. Coverage retains every pending/failed job, and
intermediate ranks are explicitly partial until the intended scope completes.

To resume on the **same unchanged host/checkout/environment**, provide the exact
run directory printed in the log:

```bash
bash benchmarks/tabarena/remote/run-session.sh lite-hpo 12 /absolute/run/directory
```

Interrupted jobs are retried; completed jobs are not refitted. Recorded failures
need deliberate `--retry-failed` through the underlying runner after inspecting
them. Changed code, dependencies, execution or hardware invalidate resume.
Recreating a droplet can change hardware: archived results remain useful, but do
not force an old run identity onto a new host or combine timings as one run.

For a completed Lite HPO run, run the acceptance audit described in [README](README.md)
before exporting the final evidence. The script does not automatically declare a
run accepted or retire the old harness.

## Save, verify, then destroy

When fitting/reporting has stopped, the session script exports the run, raw
prediction/result caches, processed and downloaded artifacts, lock, build receipt,
and Git source bundle. Manual recovery/export is:

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.cloud export /absolute/run/directory \
  --output /root/alloy-evidence.tar.gz
```

On your **local computer**, copy both files from the printed session directory:

```bash
mkdir -p ~/AlloyGBM-evidence/RUN_LABEL
scp root@DROPLET_IP:/root/SESSION/evidence.tar.gz ~/AlloyGBM-evidence/RUN_LABEL/
scp root@DROPLET_IP:/root/SESSION/evidence.tar.gz.sha256.json ~/AlloyGBM-evidence/RUN_LABEL/
python3 -m benchmarks.tabarena.cloud verify ~/AlloyGBM-evidence/RUN_LABEL/evidence.tar.gz
tar -tzf ~/AlloyGBM-evidence/RUN_LABEL/evidence.tar.gz | head
```

Run verification from a checkout containing this module; it uses only Python's
standard library and does not require the benchmark environment. A checksum or
archive validation failure means the server copy is still needed. The receipt
proves archive integrity, **not benchmark success**; inspect `run/session-status.json`,
coverage, reports and any acceptance receipt. Save your droplet/rate notes beside
the downloaded bundle. To inspect offline, extract to a new directory; `run/`
contains reports/raw results, `cache/` the upstream caches, and `source.bundle`
can be cloned with `git clone source.bundle recovered-source`.

Once the off-server copy is verified and readable, select that droplet by its
recorded ID/name in DigitalOcean and **Destroy** it. Verify it disappears from
the project. Powering off or using `shutdown` does **not** end allocation billing.
Check that no optional volumes, snapshots or backups remain billable. The scripts
never delete a server automatically, so failed exports cannot trigger data loss.
Follow [DigitalOcean's destruction instructions](https://docs.digitalocean.com/products/droplets/how-to/destroy/).

## Validation boundary

The original Ubuntu deployment, official Linux Ray pilots and all 1,326 Lite
jobs completed successfully, including 10,608 child fits, mechanical acceptance,
resource/native-thread audits and verified off-host export. The VM was deleted
after verification (about $1.61 allocation). Each new environment and search
space still needs its own pilot. See [OPERATIONS.md](OPERATIONS.md) for the
200-candidate extension, allocation deadline and incremental backup/restore workflow.
Local macOS smoke results cannot certify a new Linux deployment.

## Cost control and parallelism

At the verified $0.50/hour rate, two hours of allocated time costs $1, 12 hours
costs $6, and a full day costs $12 (excluding tax or separately billed services).
These are spending examples, not completion estimates. The fit cap is **not** a
provider billing cap: the droplet keeps billing during setup, reporting, export
and any idle pause. Record creation/destruction timestamps and rate, track elapsed
allocation cost, and monitor failures/completion so the VM is not left idle.
Copy verified evidence before a significant pause and destroy the droplet. The
user has authorized cleanup after preserving evidence; do not delete unrelated
resources. A replacement VM has a new runtime identity and cannot silently resume
as the old host.

The runner gives the machine’s 16-CPU budget to upstream. Ray/AutoGluon schedules
folds in parallel and budgets CPUs per child (often eight children at two CPUs,
subject to memory). The AlloyGBM adapter passes that child budget as `n_jobs`.
Verify the pilot’s upstream fold-strategy logs, `method_metadata.num_cpus` and
`num_cpus_child`, plus `resources.log` (CPU utilization, memory and steal time),
before launching the longer stages. Linux reuses its Ray runtime across jobs;
local macOS/native diagnostics retain process isolation. Eight folds remain eight
folds even when upstream reduces concurrency for memory.

The committed source checkout, not the current PyPI release, is built in release
mode. Record the exact source SHA and extension hash; freeze them for every stage.
Do not change model code or its search space after inspecting outer test scores.
