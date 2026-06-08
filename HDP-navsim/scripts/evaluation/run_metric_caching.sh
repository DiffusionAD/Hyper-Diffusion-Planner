#!/usr/bin/env bash
# Pre-compute the PDM metric cache (one .lz per token).
#
# Usage:
#   ./scripts/evaluation/run_metric_caching.sh [SPLIT]
#
# SPLIT defaults to "test"; pass "trainval" (or any other split shipped in
# `hdp_navsim/config/train_test_split/`) to cache for the RL reward.
# Extra Hydra overrides may be appended after the split argument.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/env.sh"

SPLIT="${1:-navtest}"
shift || true

EXTRA_ARGS=()
if [ "${SPLIT}" = "trainval" ]; then
    # Match every frame instead of every scene's keyframe -- needed by RL.
    EXTRA_ARGS+=(train_test_split.scene_filter.frame_interval=1)
fi

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
    train_test_split="${SPLIT}" \
    +metric_cache_path="${NAVSIM_METRIC_CACHE_PATH}" \
    "${EXTRA_ARGS[@]}" \
    hydra.searchpath="[pkg://navsim.planning.script.config, pkg://navsim.planning.script.config.common, pkg://hdp_navsim.config]" \
    "$@"
