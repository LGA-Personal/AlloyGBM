#!/usr/bin/env bash
# Explicitly bounded execution. This script never deletes a droplet.
set -euo pipefail
cd "$(dirname "$0")/../../.."
stage=${1:?Usage: run-session.sh STAGE MAX_HOURS [RESUME_DIR]}
hours=${2:?Specify a positive whole-number wall-clock cap in hours}
[[ "$hours" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_HOURS must be positive' >&2; exit 1; }
profile=lite-hpo
stage_args=()
case "$stage" in
  smoke) stage_args=(--stage-datasets blood-transfusion-service-center QSAR_fish_toxicity anneal --stage-candidates 1);;
  large) stage_args=(--stage-datasets APSFailure kddcup09_appetency --stage-candidates 1);;
  lite-defaults) stage_args=(--stage-candidates 0);;
  lite-hpo) ;;
  full) profile=full;;
  *) echo 'Unknown stage' >&2; exit 1;;
esac
python=.venv-tabarena/bin/python
"$python" -m benchmarks.tabarena.cloud preflight > .cache/tabarena/cloud-preflight.json
session=$(mktemp -d "${ALLOYGBM_BENCH_SESSION_ROOT:-$HOME}/alloygbm-session-XXXXXXXX")
output_root="$PWD/benchmarks/results/tabarena/cloud-$profile"
runner=("$python" -m benchmarks.tabarena.run --profile "$profile" --execution upstream --threads 16 --models AlloyGBM --output-root "$output_root")
if [[ ${#stage_args[@]} -gt 0 ]]; then
  runner+=("${stage_args[@]}")
fi
if [[ -n "${3:-}" ]]; then
  runner+=(--resume "$3")
fi
echo "Logs and evidence: $session"
monitor_pid=
if command -v vmstat >/dev/null; then
  vmstat -w 10 > "$session/resources.log" &
  monitor_pid=$!
fi
cleanup() { if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
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
timeout --signal=INT --kill-after=120s 1h "$python" -m benchmarks.tabarena.report "$run_dir" --published-only > "$session/report.log" 2>&1
report_status=$?
set -e
printf '{"fit_exit":%s,"report_exit":%s}\n' "$status" "$report_status" > "$run_dir/session-status.json"
cp "$session/"*.log "$run_dir/"
session_evidence="$run_dir/sessions/$(basename "$session")"
mkdir -p "$session_evidence"
cp "$session/"*.log "$run_dir/session-status.json" "$session_evidence/"
printf '%s\n' "$stage" > "$session_evidence/stage.txt"
"$python" -m benchmarks.tabarena.cloud export "$run_dir" --output "$session/evidence.tar.gz"
echo 'Copy evidence.tar.gz AND evidence.tar.gz.sha256.json off the droplet, verify locally, then DESTROY the droplet. Stopping it does not end billing.'
if [[ "$status" != 0 ]]; then exit "$status"; fi
exit "$report_status"
