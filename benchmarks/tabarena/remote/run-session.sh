#!/usr/bin/env bash
# Explicitly bounded execution. This script never deletes a droplet.
set -euo pipefail
cd "$(dirname "$0")/../../.."
profile=${1:?Usage: run-session.sh PROFILE MAX_HOURS [RESUME_DIR]}
hours=${2:?Specify a positive whole-number wall-clock cap in hours}
[[ "$hours" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_HOURS must be positive' >&2; exit 1; }
case "$profile" in smoke|lite-defaults|lite-hpo|full) ;; *) echo 'Unknown profile' >&2; exit 1 ;; esac
python=.venv-tabarena/bin/python
"$python" -m benchmarks.tabarena.cloud preflight > .cache/tabarena/cloud-preflight.json
session=$(mktemp -d "${ALLOYGBM_BENCH_SESSION_ROOT:-$HOME}/alloygbm-session-XXXXXXXX")
output_root="$PWD/benchmarks/results/tabarena/$(basename "$session")"
runner=("$python" -m benchmarks.tabarena.run --profile "$profile" --execution upstream --output-root "$output_root")
if [[ -n "${3:-}" ]]; then
  runner+=(--resume "$3")
fi
echo "Logs and evidence: $session"
set +e
timeout --signal=INT --kill-after=120s "${hours}h" "${runner[@]}" > "$session/run.log" 2>&1
status=$?
set -e
if [[ -n "${3:-}" ]]; then
  run_dir=$3
else
  manifest=$(find "$output_root" -name run_manifest.json -print -quit)
  [[ -n "$manifest" ]] || { echo "No run created; inspect $session/run.log" >&2; exit 1; }
  run_dir=$(dirname "$manifest")
fi
# Reporting may fail when a run is interrupted or the calibration fit failed.
# Preserve the original ledger, caches and diagnostics in either case.
set +e
timeout --signal=INT --kill-after=120s 1h "$python" -m benchmarks.tabarena.report "$run_dir" --published-context > "$session/report.log" 2>&1
report_status=$?
set -e
printf '{"fit_exit":%s,"report_exit":%s}\n' "$status" "$report_status" > "$run_dir/session-status.json"
cp "$session/"*.log "$run_dir/"
"$python" -m benchmarks.tabarena.cloud export "$run_dir" --output "$session/evidence.tar.gz"
echo 'Copy evidence.tar.gz AND evidence.tar.gz.sha256.json off the droplet, verify locally, then DESTROY the droplet. Stopping it does not end billing.'
if [[ "$status" != 0 ]]; then exit "$status"; fi
exit "$report_status"
