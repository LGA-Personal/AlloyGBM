# Release Guide & Checklist

This is the operating manual for cutting an AlloyGBM release. It is meant to be
followed top-to-bottom: every item is something we got wrong (or nearly got
wrong) on a prior release, so the checklist exists to prevent regressions in
release hygiene — not as theoretical best practice.

Treat each section as a gate. Do not skip ahead; later steps assume the earlier
gates have passed.

---

## 0. Decide What Kind Of Release This Is

Patch releases (`0.7.1 → 0.7.2`) only fix bugs, doc errors, or close
limitations explicitly queued in the previous release's "documented for the
next release" list. They do not add net-new user-facing features.

Minor releases (`0.7.x → 0.8.0`) add net-new features, change defaults, or
adjust the public API in a backward-compatible way.

Major releases (`0.x → 1.0`) are reserved for compatibility breaks. AlloyGBM
has not done one yet — when it happens, this guide will get a new section.

Write the release type into the PR description. The type determines the
CHANGELOG style (new section vs. addendum), the announcement copy, and how
aggressively the limitations section needs editing.

---

## 1. Versioning — All The Places To Bump

Bump the version string in **every** one of these files. This is the part we
have historically missed pieces of, so keep this list authoritative and edit
it as the repo grows.

### Required version bumps (release-blocking)

- [ ] `Cargo.toml` — workspace `version = "X.Y.Z"`
- [ ] `pyproject.toml` — `version = "X.Y.Z"`
- [ ] `docs/site/source/conf.py` — `version = "X.Y.Z"` (Sphinx site footer)

These three must match exactly. If they drift, the publish workflow may upload
a wheel whose internal metadata claims the wrong version.

### Required content updates (release-blocking)

- [ ] `CHANGELOG.md` — prepend a new `## X.Y.Z` section above the previous
      release. Keep the previous release's section intact.
- [ ] `docs/site/source/release.rst` — prepend a new `What's new in X.Y.Z`
      section above the previous release. Keep the previous section intact.
- [ ] `docs/roadmap/current.md` — prepend a new `## What Shipped In X.Y.Z`
      section and update the top-of-file release-direction paragraph.
- [ ] `docs/limitations.md` — update "Last updated for vX.Y.Z" line; move any
      newly-resolved limitations from "Remaining Limitations" to "Resolved
      (Previously Limitations)"; add any new known limitations (the ones
      queued for the next release) under "Remaining Limitations".

### Version-conditional updates (only if the restriction still applies)

Search the repo for the previous version string and review every hit:

```bash
git grep -n '0\.7\.0\|v0\.7\.0' \
  -- 'docs/**' 'README.md' 'CLAUDE.md' '*.toml' '*.py' '*.rs'
```

For each hit that says something like *"X is not supported in v0.7.0"* or
*"X works as of v0.7.0"*, decide:

- If the restriction still applies in the new release, bump the version to
  the current one (e.g. `0.7.0 → 0.7.1`).
- If the restriction has been lifted, remove the version reference and
  rewrite the surrounding sentence.

Common places these references live:

- Python error messages in `bindings/python/alloygbm/regressor.py`,
  `classifier.py`, `ranker.py`, `multi_label_ranker.py`
- `docs/user/*.md` parameter docs and limitation notes
- `docs/site/source/*.rst` (mirrors of `docs/user/*.md`)
- README "Current Limitations" section
- Wheel target list in `README.md` and `docs/user/installation.md`

### Documentation updates (release-blocking when shipped features change)

When a release ships net-new user-facing surface (new estimator, new
parameter, new behavior), every one of these has to be updated in lockstep.
Skipping any one of them is what made v0.7.1 docs drift.

- [ ] `README.md`
  - "When To Use AlloyGBM" bullet list (add new capability if it widens fit)
  - Quick Examples section (add a snippet for any new top-level API)
  - "Feature Summary" → Estimators / Training Features / Inference and
    Explanations / Validation Helpers / Metrics
  - "Current Limitations" (add new, remove resolved)
- [ ] `docs/user/quickstart.md` — add a section for any new top-level API
- [ ] `docs/user/gbmregressor.md` — parameters table, post-fit attributes,
      compatibility tables, limitations subsections
- [ ] `docs/user/gbmclassifier.md` — inherited-parameters list, post-fit attrs
- [ ] `docs/user/gbmranker.md` — inherited-parameters list, "Current Scope",
      cross-references to multi-output ranker if applicable
- [ ] `docs/user/explanations.md` — SHAP / feature importance compatibility
- [ ] `docs/user/morphboost.md` — when MorphBoost composability changes
- [ ] `docs/user/validation.md` — when validation helpers change
- [ ] `docs/user/benchmarks.md` — when benchmark arms or stories change
- [ ] `docs/site/source/index.rst` — the top-of-page release note paragraph
- [ ] `docs/site/source/estimator.rst` — parallel to `gbmregressor.md`
- [ ] `docs/site/source/classifier.rst` — parallel to `gbmclassifier.md`
- [ ] `docs/site/source/ranker.rst` — parallel to `gbmranker.md`
- [ ] `docs/site/source/quickstart.rst` — parallel to `quickstart.md`
- [ ] `docs/site/source/explanations.rst` — parallel to `explanations.md`
- [ ] `benchmarks/README.md` — default model arm list, scenario table

### Bookkeeping (release-blocking)

- [ ] `CLAUDE.md` — crate count, module map, artifact section list, smoke
      tests. Refresh whenever the corresponding code structure changes, not
      just on releases. But verify on every release.
- [ ] `docs/README.md` — directory map (only if directories were added or
      removed)
- [ ] `docs/roadmap/current.md` "Longer-Term Themes" — drop items that just
      shipped, replace with concrete v(N+1) follow-ups

---

## 2. Audit Pass — Stale Content Sweep

Before tagging, do a single mechanical sweep for stale phrases. These are
patterns that have outlived past releases and cause confusing user-facing
docs.

```bash
# Anything claiming a feature is rejected/unsupported should still be true.
git grep -nE \
  'rejected in this release|not yet supported|currently raises an error|currently requires|No interaction constraints|No multi-?label|warm.start.*rejected' \
  -- 'docs/**' README.md CLAUDE.md benchmarks/README.md

# Old version references that should have been bumped or removed.
git grep -nE 'v?0\.7\.0' \
  -- 'docs/**' README.md CLAUDE.md '*.toml' '*.py' '*.rs' \
  | grep -v 'archive\|CHANGELOG\|release\.rst\|roadmap/current\.md'
```

Both queries should return zero non-historical hits. Historical hits in
`CHANGELOG.md`, `docs/site/source/release.rst`, `docs/roadmap/current.md`,
and anything under `docs/archive/` are expected — those are intentionally
preserved release history.

---

## 3. Verify — Local

Run the full verification suite from a clean tree. Do not skip any of these.

- [ ] `cargo check --workspace`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `cargo fmt --all --check`
- [ ] `maturin develop --release` (build the Python extension)
- [ ] `.venv/bin/python -m pytest bindings/python/tests/ -q`
- [ ] Smoke check every public top-level API:

```bash
.venv/bin/python -c "from alloygbm import GBMRegressor; m = GBMRegressor(n_estimators=3); m.fit([[1],[2],[3]], [1,2,3]); print(m.predict([[2]]))"
.venv/bin/python -c "from alloygbm import GBMClassifier; m = GBMClassifier(n_estimators=3); m.fit([[1],[2],[3],[4]], [0,0,1,1]); print(m.predict([[2]]))"
.venv/bin/python -c "from alloygbm import GBMRanker; m = GBMRanker(n_estimators=3); m.fit([[1],[2],[3],[4]], [0,1,0,1], group=[0,0,1,1]); print(m.predict([[2]]))"
.venv/bin/python -c "from alloygbm import MultiLabelGBMRanker; import numpy as np; m = MultiLabelGBMRanker(n_estimators=3); m.fit([[1],[2],[3],[4]], np.array([[0,1],[1,0],[0,1],[1,0]]), group=[0,0,1,1]); print(m.predict([[2]]))"
```

If you added new top-level API in this release, add a one-liner for it here
before opening the PR.

### Fresh-env wheel install

The publish workflow does this too, but doing it locally first catches
packaging breakage before it ships.

- [ ] `maturin build --release` produces a wheel in `target/wheels/`
- [ ] Create a fresh venv outside the repo and `pip install` that wheel
- [ ] `python -c "import alloygbm; print(alloygbm.native_runtime_info())"`
- [ ] Run the 4 smoke snippets above against the installed wheel

---

## 4. Verify — CI

- [ ] Open the release PR against `main`. The branch protection rules require
      `ci.yml` to pass before merge.
- [ ] Confirm there are no `CANCELLED` (red but not failed) checks listed on
      the PR — these show up if the same workflow fires twice on push+PR. The
      `push:` trigger is restricted to `main` precisely to prevent this; if
      you see it, investigate before tagging.
- [ ] If CI flakes (macOS rustup-init has historically been flaky), do not
      tag until a clean rerun is green. Document the flake on the PR for
      future debugging.

---

## 5. Merge, Tag, Release

Order matters. Each step depends on the previous one being clean.

1. **Squash-merge the release PR into `main`.** Delete the branch.
2. **Sync local `main`:**
   ```bash
   git fetch origin main && git checkout main && git pull --ff-only
   ```
3. **Create the annotated tag** (annotated, not lightweight — the release
   workflow keys off tag metadata):
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```
4. **Create the GitHub release.** Use the same content as the
   `CHANGELOG.md` section. Keep it concise — the changelog is the source of
   truth; the release notes link to it. Example:
   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" --notes "$(cat <<'EOF'
   ## Highlights
   - <new feature 1>
   - <new feature 2>

   ## Documented for vX.Y.(Z+1)
   - <known limitation 1>

   Full notes: https://github.com/LGA-Personal/AlloyGBM/blob/main/CHANGELOG.md
   EOF
   )"
   ```

---

## 6. Publish

The `publish.yml` workflow fires on tag push and builds + uploads wheels.

- [ ] Confirm `publish.yml` ran on the tag and finished green. If it fails,
      fix forward in a new patch release — never delete a published tag.
- [ ] Confirm the new version appears on PyPI:
      `pip index versions alloygbm` should list `X.Y.Z`.
- [ ] In a fresh venv outside the repo, `pip install alloygbm==X.Y.Z` and run
      the 4 smoke snippets. If install fails on a target platform, file an
      issue and decide whether to yank the release.

---

## 7. Post-Release Bookkeeping

- [ ] Confirm the Read the Docs build picked up the tag. The site footer
      should show the new version.
- [ ] If there are open issues or PRs labeled with the just-released version,
      close or relabel them.
- [ ] Open a tracking issue for any limitation marked as queued for the next
      release. This is the bridge to the next release's plan.

---

## Standing Policies

These do not change per-release, but they constrain release decisions, so
they belong in the same place.

### Wheel target policy

- macOS `arm64`: required for every public release.
- Linux `x86_64` (manylinux): required for every public release.
- Windows: deferred indefinitely. Do not publish a Windows wheel without
  explicit policy discussion first.
- macOS Intel (`x86_64`): deferred. Same gate as Windows.
- Source distribution: always built, always uploaded as a fallback.

### Linux wheel build policy

Linux wheels must be built in a `manylinux`-compatible environment. A bare
`ubuntu-latest` build is not acceptable for a public PyPI release because the
resulting wheel has system-glibc-pinned dependencies that fail on the wide
deployment matrix users actually have.

### Benchmark messaging policy

Public-facing benchmark claims live in `README.md` and `docs/user/benchmarks.md`.
They must stay honest about both strengths and weak spots — never claim
parity or dominance unless the comparative results actually support it. The
current public claim (as of v0.7.1):

- **Regression:** strongest on `panel_time_series`; strong on
  `dow_jones_financial`; competitive on `dense_numeric`; trails on
  `california_housing` and `bike_sharing`.
- **Classification:** competitive with established libraries on standard
  datasets (`breast_cancer`, `synthetic_classification`).
- **Ranking:** competes via native LambdaMART on synthetic ranking scenarios.

Do not broaden these claims unless a new benchmark run materially changes
the picture. If the picture changes, update both the README and the
benchmark guide in the same PR.

### Pre-1.0 stability policy

AlloyGBM is pre-1.0. Backward-incompatible API changes are allowed in minor
releases but should be called out in `CHANGELOG.md` under a `### Breaking
Changes` heading. Artifacts written by an older minor are best-effort
readable by a newer minor — we make a serious effort to back-compat artifact
sections (e.g. `FeatureBaseline` added in v0.7.1 reads as zero means on
older artifacts) but do not guarantee it.

---

## When This Guide Itself Needs To Be Updated

Update this file whenever:

- a new directory of user-facing docs is added (the "documentation updates"
  list grows)
- a new top-level Python API is added (the smoke-test list grows)
- a release fails for a reason not covered here (add a new gate)
- a wheel-target policy changes
- the CI workflow structure changes (the "Verify — CI" section needs to
  match reality)

This guide is part of the release surface. Treat doc edits to it as
release-blocking the same way you would treat a stale version pin.
