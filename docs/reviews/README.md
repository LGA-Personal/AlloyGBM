# Code Reviews

Periodic deep-dive reviews of the codebase — design, efficiency, accuracy, and robustness —
and the follow-up documents recording how their findings were addressed. These are point-in-time
assessments: file/line references and benchmark numbers are pinned to the version and commit in
each document's header and may drift afterward.

## Conventions

**Naming**

- Review: `YYYY-MM-DD-<version>-<topic>.md`
  (e.g. `2026-07-02-v0.12.10-core.md`)
- Follow-up / resolution doc: `<review-name>-resolutions.md`
  (e.g. `2026-07-02-v0.12.10-core-resolutions.md`), written when the findings are worked
  through — typically alongside the release(s) that address them.

**Header** — every document starts with a metadata table:

| Date | Reviewer | Version reviewed | Commit | Status |
|---|---|---|---|---|

- `Reviewer`: who/what performed the review (e.g. `Claude Fable 5`, a human, or both).
- `Status` (reviews): `Open` → `In progress` → `Addressed in <version>` (link the resolutions
  doc). Partially-addressed reviews stay `In progress` with per-item status tracked in the
  resolutions doc.

**Resolutions docs** mirror the review's numbering: one entry per finding, each marked
`Fixed in <version>` / `Won't fix (reason)` / `Deferred (tracking ref)`, with links to the
commits/PRs and, where the review included measurements, the same A/B re-run showing the
after numbers.

**Cross-linking**: reviews link to code with repo-relative paths from this directory
(`../../crates/...`). When a finding graduates into `docs/roadmap/current.md` or an issue,
link it both ways.

## Index

| Review | Version | Reviewer | Status |
|---|---|---|---|
| [Core review](2026-07-02-v0.12.10-core.md) — whole-workspace design/efficiency/accuracy | v0.12.10 | Claude Fable 5 | [In progress](2026-07-02-v0.12.10-core-resolutions.md) |
| [Special-modes review](2026-07-02-v0.12.10-special-modes.md) — MorphBoost, DRO, PL trees, neutralization, objectives, DART/GOSS (incl. 2 correctness bugs) | v0.12.10 | Claude Fable 5 | [In progress](2026-07-02-v0.12.10-special-modes-resolutions.md) |
