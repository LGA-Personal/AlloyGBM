# Security Policy

## Supported Versions

AlloyGBM is pre-1.0. Only the latest published release receives security
fixes; older minor releases are not patched.

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

**Do not** open a public GitHub issue for suspected security
vulnerabilities. Public issues are visible to everyone and a premature
disclosure can put downstream users at risk before a fix is available.

Instead, use GitHub's private vulnerability reporting:

1. Go to the
   [Security tab](https://github.com/LGA-Personal/AlloyGBM/security/advisories/new).
2. Click **Report a vulnerability**.
3. Include a minimal reproduction, the affected version, and the
   impact you've observed.

If for some reason you can't use the Security tab, email the maintainer
listed in [`pyproject.toml`](pyproject.toml) directly with the same
information. Put `[AlloyGBM security]` in the subject line.

## What to expect

- An initial acknowledgement within 5 business days.
- A triage assessment (impact, affected versions, severity) within 14
  days of the initial report.
- For confirmed issues, a private fix branch and a coordinated
  disclosure timeline. We aim for a fix release within 30 days of
  triage for high-severity issues, longer for low-severity issues that
  require a non-trivial redesign.

## Scope

In scope:

- Memory-safety bugs in the Rust crates (the workspace forbids `unsafe`,
  but bugs in dependencies or in safe code that lead to UB are still in
  scope).
- Crashes triggered by malformed model artifacts (`.agbm` files) or
  malformed input data.
- Pickle / artifact deserialization issues that allow code execution.
- Supply-chain advisories on dependencies AlloyGBM directly depends on.

Out of scope:

- Issues that require the attacker to already have arbitrary code
  execution on the machine running AlloyGBM.
- Numerical correctness bugs without a security impact (file these as
  regular bug reports).
- Performance regressions.
