#!/usr/bin/env bash
# Run a ten-minute CUDA stress test every thirty minutes. Stop with Ctrl-C.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/.build"
STRESS_SOURCE="${SCRIPT_DIR}/cuda_gpu_stress.cu"
STRESS_BINARY="${BUILD_DIR}/cuda_gpu_stress"

GPU_INDEX="${GPU_INDEX:-0}"
TEST_SECONDS="${TEST_SECONDS:-600}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
MEMORY_MIB="${MEMORY_MIB:-4096}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
RUN_ONCE=false

usage() {
  cat <<'EOF'
Usage: gpu_stress_every_30min.sh [--once]

Environment variables:
  GPU_INDEX         GPU index for CUDA and nvidia-smi (default: 0)
  TEST_SECONDS      Duration of one test in seconds (default: 600)
  INTERVAL_SECONDS  Time between test starts in seconds (default: 1800)
  MEMORY_MIB        Maximum memory used by the test in MiB (default: 4096)
  LOG_DIR           Directory for timestamped logs (default: scripts/logs)

The CUDA program uses the lower of MEMORY_MIB and 70% of currently free GPU
memory, so it leaves capacity for the display driver and other workloads.
EOF
}

case "${1:-}" in
  "") ;;
  --once) RUN_ONCE=true ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

require_positive_integer() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    printf '%s must be a positive integer, got: %s\n' "${name}" "${value}" >&2
    exit 2
  }
}

[[ "${GPU_INDEX}" =~ ^[0-9]+$ ]] || {
  printf 'GPU_INDEX must be a non-negative integer, got: %s\n' "${GPU_INDEX}" >&2
  exit 2
}
require_positive_integer TEST_SECONDS "${TEST_SECONDS}"
require_positive_integer INTERVAL_SECONDS "${INTERVAL_SECONDS}"
require_positive_integer MEMORY_MIB "${MEMORY_MIB}"

if (( TEST_SECONDS > INTERVAL_SECONDS )); then
  printf 'TEST_SECONDS (%s) cannot exceed INTERVAL_SECONDS (%s).\n' \
    "${TEST_SECONDS}" "${INTERVAL_SECONDS}" >&2
  exit 2
fi

command -v nvcc >/dev/null || {
  printf 'nvcc is required. Install the CUDA toolkit or use a gpu-burn-based runner.\n' >&2
  exit 1
}
command -v nvidia-smi >/dev/null || {
  printf 'nvidia-smi is required.\n' >&2
  exit 1
}

mkdir -p "${BUILD_DIR}" "${LOG_DIR}"
if [[ ! -x "${STRESS_BINARY}" || "${STRESS_SOURCE}" -nt "${STRESS_BINARY}" ]]; then
  nvcc -O3 -lineinfo "${STRESS_SOURCE}" -o "${STRESS_BINARY}"
fi

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

run_test() {
  local timestamp log_file
  timestamp="$(date +%Y%m%d-%H%M%S)"
  log_file="${LOG_DIR}/gpu${GPU_INDEX}-${timestamp}.log"

  log "Starting GPU ${GPU_INDEX} stress test for ${TEST_SECONDS}s; log: ${log_file}"
  nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader -i "${GPU_INDEX}" | tee -a "${log_file}"

  "${STRESS_BINARY}" --gpu "${GPU_INDEX}" --seconds "${TEST_SECONDS}" \
    --memory-mib "${MEMORY_MIB}" 2>&1 | tee -a "${log_file}"

  nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader -i "${GPU_INDEX}" | tee -a "${log_file}"
  log "GPU ${GPU_INDEX} stress test finished."
}

trap 'log "Stopping GPU stress-test scheduler."; exit 0' INT TERM

while true; do
  started_at="$(date +%s)"
  run_test
  "${RUN_ONCE}" && exit 0

  next_start=$((started_at + INTERVAL_SECONDS))
  sleep_seconds=$((next_start - $(date +%s)))
  if (( sleep_seconds > 0 )); then
    log "Next test starts in ${sleep_seconds}s."
    sleep "${sleep_seconds}"
  fi
done
