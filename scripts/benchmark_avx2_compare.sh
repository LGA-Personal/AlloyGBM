#!/usr/bin/env bash
set -euo pipefail

TARGET=""
BENCH="histogram_kernels"
RUNS=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --bench)
      BENCH="${2:-}"
      shift 2
      ;;
    --runs)
      RUNS="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run_dir="$(mktemp -d)"
cmd=(cargo bench -p alloygbm-backend-cpu --bench "${BENCH}")
if [[ -n "${TARGET}" ]]; then
  cmd+=(--target "${TARGET}")
fi

median_of() {
  printf '%s\n' "$@" | sort -n | awk '
    { a[NR] = $1 }
    END {
      n = NR
      if (n == 0) {
        exit 1
      } else if (n % 2 == 1) {
        print a[(n + 1) / 2]
      } else {
        print (a[n / 2] + a[(n / 2) + 1]) / 2
      }
    }
  '
}

declare -a default_values=()
declare -a fallback_values=()
default_avx2=""
fallback_avx2=""
default_override=""
fallback_override=""

for i in $(seq 1 "${RUNS}"); do
  log="${run_dir}/default_${i}.log"
  echo "Running default benchmark ${i}/${RUNS}: ${cmd[*]}"
  "${cmd[@]}" | tee "${log}"

  medium="$(awk -F'ns_per_iter=' '/histogram_build_medium_backend:/ {print $2; exit}' "${log}")"
  if [[ -z "${medium}" ]]; then
    echo "Failed to parse medium benchmark ns_per_iter value from ${log}." >&2
    exit 1
  fi
  default_values+=("${medium}")

  if [[ "${i}" -eq 1 ]]; then
    default_avx2="$(awk -F': ' '/runtime_avx2_enabled:/ {print $2; exit}' "${log}")"
    default_override="$(awk -F': ' '/runtime_avx2_override:/ {print $2; exit}' "${log}")"
  fi
done

echo
for i in $(seq 1 "${RUNS}"); do
  log="${run_dir}/forced_scalar_${i}.log"
  echo "Running forced-scalar benchmark ${i}/${RUNS}: ALLOYGBM_DISABLE_AVX2=1 ${cmd[*]}"
  ALLOYGBM_DISABLE_AVX2=1 "${cmd[@]}" | tee "${log}"

  medium="$(awk -F'ns_per_iter=' '/histogram_build_medium_backend:/ {print $2; exit}' "${log}")"
  if [[ -z "${medium}" ]]; then
    echo "Failed to parse medium benchmark ns_per_iter value from ${log}." >&2
    exit 1
  fi
  fallback_values+=("${medium}")

  if [[ "${i}" -eq 1 ]]; then
    fallback_avx2="$(awk -F': ' '/runtime_avx2_enabled:/ {print $2; exit}' "${log}")"
    fallback_override="$(awk -F': ' '/runtime_avx2_override:/ {print $2; exit}' "${log}")"
  fi
done

default_median="$(median_of "${default_values[@]}")"
fallback_median="$(median_of "${fallback_values[@]}")"
delta_vs_forced_scalar=""
if [[ "${default_avx2}" == "false" && "${fallback_avx2}" == "false" ]]; then
  delta_vs_forced_scalar="n/a (runtime AVX2 unavailable)"
else
  delta_percent="$(awk -v d="${default_median}" -v f="${fallback_median}" 'BEGIN { printf("%.2f", ((d - f) / f) * 100.0) }')"
  delta_vs_forced_scalar="${delta_percent}%"
fi

echo
echo "AVX2 comparison summary"
echo "  logs: ${run_dir}"
echo "  runs_per_mode: ${RUNS}"
echo "  runtime_avx2_enabled(default): ${default_avx2:-unknown}"
echo "  runtime_avx2_enabled(forced_scalar): ${fallback_avx2:-unknown}"
echo "  runtime_avx2_override(default): ${default_override:-unknown}"
echo "  runtime_avx2_override(forced_scalar): ${fallback_override:-unknown}"
echo "  medium_ns_per_iter(default_runs): ${default_values[*]}"
echo "  medium_ns_per_iter(forced_scalar_runs): ${fallback_values[*]}"
echo "  medium_ns_per_iter(default_median): ${default_median}"
echo "  medium_ns_per_iter(forced_scalar_median): ${fallback_median}"
echo "  medium_delta_vs_forced_scalar_median: ${delta_vs_forced_scalar}"
