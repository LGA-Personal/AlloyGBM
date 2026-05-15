<!--
Thanks for opening a PR. Fill in every section below. If a section
doesn't apply, write "N/A" rather than deleting it.
-->

## Summary

<!-- One or two sentences: what does this PR change, and why? -->

## Type of change

- [ ] Bug fix (no API change)
- [ ] New feature (adds user-visible API or behavior)
- [ ] Breaking change (existing API or behavior changes incompatibly)
- [ ] Documentation only
- [ ] Internal refactor / chore (no user-visible change)

## How was this tested?

<!--
- For Rust changes: which `cargo test` targets exercise this?
- For Python changes: which tests under `bindings/python/tests/`?
- For new features: did you add a new test file?
- For doc-only changes: confirm `cargo doc` and Sphinx build cleanly.
-->

## Checklist

- [ ] `cargo test --workspace` passes locally
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes locally
- [ ] `cargo fmt --all --check` passes locally
- [ ] `.venv/bin/python -m pytest bindings/python/tests/ -q` passes locally
- [ ] Added or updated tests covering the change
- [ ] Updated user-facing docs (`docs/user/*.md` + Sphinx mirror under `docs/site/source/*.rst`) if the API surface changed
- [ ] Updated `CHANGELOG.md` under the unreleased section
- [ ] If this is part of a release: followed [`docs/reference/release_checklist.md`](../docs/reference/release_checklist.md)

## Related issues / PRs

<!-- Link to the issue this closes, or related work. -->
