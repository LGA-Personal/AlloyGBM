#!/usr/bin/env bash
# Run on a fresh Ubuntu 24.04 x86_64 droplet, in a checkout at an exact commit.
set -euo pipefail
cd "$(dirname "$0")/../../.."
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Requires Linux x86_64' >&2; exit 1; }
sudo_cmd=()
if [[ "$EUID" != 0 ]]; then sudo_cmd=(sudo); fi
"${sudo_cmd[@]}" apt-get update
"${sudo_cmd[@]}" apt-get install -y build-essential curl ca-certificates git pkg-config libssl-dev libgomp1 tmux rsync
# Download installers to files so a failed download cannot become a partial pipe.
installer_dir=$(mktemp -d)
trap 'rm -rf "$installer_dir"' EXIT
if ! command -v uv >/dev/null; then
  curl --proto '=https' --tlsv1.2 -fsSL https://astral.sh/uv/install.sh -o "$installer_dir/uv.sh"
  sh "$installer_dir/uv.sh"
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v rustup >/dev/null; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o "$installer_dir/rustup.sh"
  sh "$installer_dir/rustup.sh" -y --profile minimal --default-toolchain none
fi
rustup toolchain install 1.92.0 --profile minimal
uv venv --python 3.12 .venv-tabarena
uv pip install --python .venv-tabarena/bin/python --prerelease=allow -r benchmarks/tabarena/requirements.lock
.venv-tabarena/bin/python -m benchmarks.tabarena.prepare_upstream
uv pip install --python .venv-tabarena/bin/python --no-deps -e .cache/tabarena/upstream/packages/tabarena -e .cache/tabarena/upstream/packages/bencheval
VIRTUAL_ENV="$PWD/.venv-tabarena" .venv-tabarena/bin/maturin develop --release
.venv-tabarena/bin/python -m benchmarks.tabarena.provenance record-build
.venv-tabarena/bin/python -m benchmarks.tabarena.prepare_upstream --check
.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena benchmarks/tests/test_tabarena_cloud.py -q
.venv-tabarena/bin/python -m benchmarks.tabarena.cloud preflight
