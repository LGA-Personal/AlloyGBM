---
name: alloygbm-tabarena
description: Run, monitor, compare and recover AlloyGBM TabArena benchmarks, including disposable Linux hosts, candidate budgets and verified result backups.
---

# AlloyGBM TabArena

Locate the AlloyGBM checkout, read its project instructions, then read
`benchmarks/tabarena/README.md`, `DROPLET.md`, and `OPERATIONS.md`.
These maintained repository documents own commands and recovery procedures.
If the skill is installed outside the repo, resolve the checkout from the current
project or ask for its location. Do not assume an old host or checkout is active.

Before paid compute, record the chosen library commit, upstream pin, profile,
search-space version, entrant list, candidate count, expected jobs, hardware,
creation time, hourly rate and an allocation deadline. Establish authorization
and a budget from the current request; this skill does not grant either.
Reuse an already authorized active host; do not create duplicate controllers.
Use the installed DigitalOcean app for provider operations, with its provision
skill and bundled SSH helper. Do not use ad hoc API credentials or doctl.

Preserve TabArena's official tasks/splits, preprocessing, 8x1 inner folds,
seed schedule, stopping/scoring/selection/ensemble logic. Use published-only
competitors unless peer refits are explicitly requested. Distinguish default,
tuned and tuned+ensemble. Lite changes outer coverage, not the published tree
baselines' 200-candidate tuning budget. Inspect source artifacts and coverage;
Elo depends on the entrant pool, scope and snapshot.

Freeze the model/search space before fitting. `v1` preserves historical runs;
`modes-v2` adds the user-requested Morph/DRO/PL combinations, retaining the first
25 v1 candidates. DRO cannot combine with linear leaves. Treat new spaces as
new experiments and disclose prior benchmark inspection. Never select by test
scores or silently substitute successful datasets for the planned scope.

Use a clean release build and its receipt. Stage pilots under one immutable
plan. Resume only an identical identity; use `import_fits` for explicit exact-job
reuse across scopes/hosts, retaining origin manifests and mixed-hardware timing.
Trust only self-generated, checksum-verified pickle evidence.

Confirm real eight-child completion, Ray fold-worker CPU allocation and native
`n_jobs` before long runs. Set a fit cap inside the allocation deadline, reserve
time for export/transfer, monitor memory/disk/failures, and examine failures
before retrying. Preserve failure history.

Before the long run, test checkpoint transfer and restore with real pilot data.
Pull and verify incremental checkpoints off-host periodically. With the backup
age guard enabled, a stale acknowledgement stops new fits. Checkpoints happen
between jobs; an in-flight fit can run up to its official deadline. Reporting
is separate from durable raw results. A partial/imputed table is not a complete
competitive result.

On completion or a significant pause, stop the controller, make the final
checkpoint, generate standard reports and acceptance, export the complete
bundle, copy it plus sidecar off-host and verify it. If cleanup was authorized,
delete only the recorded benchmark droplet promptly, confirm deletion and
record final cost; remove its monitor. Power-off still bills. Never delete the
sole evidence copy. Do lengthy local documentation or PR work after cleanup.
