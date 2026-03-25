Release and platform policy
===========================

AlloyGBM ``0.1.0`` is intentionally conservative about packaging claims.

Validated release surface
-------------------------

For the first public release, the intended release surface is:

- macOS ``arm64`` wheel
- Linux ``x86_64`` manylinux wheel
- source distribution

Deferred targets
----------------

These are intentionally deferred until later:

- Windows wheels
- macOS Intel wheels

Why the scope is narrow
-----------------------

The project currently prioritizes honesty over broad packaging claims.

That means:

- no claiming support for targets that have not been validated
- no treating generic ``ubuntu-latest`` builds as broadly portable Linux wheels
- keeping source builds available as a fallback

Release checklist summary
-------------------------

Before a public release:

- confirm package metadata and version
- confirm user docs are up to date
- confirm CI is green
- confirm the built wheel installs in a fresh environment
- confirm the publish workflow smoke-tests its wheel artifacts before upload
- confirm benchmark messaging stays narrow and defensible
