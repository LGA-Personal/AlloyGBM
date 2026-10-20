# Disposable benchmark droplet

Use this when a development change needs a substantive comparison. Keep normal
adapter development and small diagnostic tests local. Nothing in this repository
creates a cloud resource or stores a DigitalOcean token.

## Create only when ready to run

In the [DigitalOcean create panel](https://cloud.digitalocean.com/droplets/new),
choose Ubuntu **24.04 LTS x86_64**, a **dedicated CPU General Purpose** plan with
**16 vCPUs / 64 GB RAM**, and at least **150 GB disk** (200 GB is preferable for
dependencies, caches and an export copy). Choose an available nearby region and
add your SSH public key. Do not enable backups, snapshots or separate volumes
for this disposable workflow. Check the displayed current rate before creating.

Record the droplet **ID, name, public IP, rate and creation time** locally. The
project acceptance policy requires upstream Ray execution, at least 16 available
CPUs, at least 60 GiB usable RAM on a nominal 64-GB machine, and no smaller thread
caps. This is our reproducible development target, not a universal TabArena
hardware rule. A dedicated VM still does not reproduce the official CPU model.
See [plan selection](https://docs.digitalocean.com/products/droplets/concepts/choosing-a-plan/)
and [creation instructions](https://docs.digitalocean.com/products/droplets/how-to/create/).

The Codex DigitalOcean plugin's small default development image/size is not this
benchmark target. This guide uses the provider console so the required machine
can be selected explicitly. No cluster or GPU is needed.

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
running a larger profile. The pilot uses the three official smoke tasks, four
boosters with default + one sample, and a default RandomForest calibrator:
**27 outer jobs / 216 child fits**. It checks execution, not competitiveness.

After a successful pilot, choose an explicit cap using observed fit times and
the current rate. These are examples of caps, not completion estimates:

```bash
# Defaults only: 255 outer jobs, including 51 default RF calibration jobs.
bash benchmarks/tabarena/remote/run-session.sh lite-defaults 6
# Standard initial HPO scope: default + 25 samples for each booster, plus RF.
# 5,355 outer jobs / 42,840 child fits; may require many sessions/days.
bash benchmarks/tabarena/remote/run-session.sh lite-hpo 12
```

The session script records fit/report exit codes, keeps failed-job diagnostics,
and attempts an evidence export on normal completion, a fit failure or expiration
of the fit cap. If it is killed externally, runs out of disk, or export fails,
follow the manual export below. Failures require inspection; do not report them
as scores. With RF present, reports use upstream calibration and RF fallback.
If RF fails a scoped task, ranking is refused until it is repaired.

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

The archive round trip, corruption checks, resource gate and shell syntax are
tested locally. A fresh Ubuntu installation and full Linux Ray pilot still need
to be executed on the selected droplet before committing money to Lite HPO.
Local macOS smoke results cannot certify that deployment or satisfy acceptance.
