#!/usr/bin/env bash
# Pre-compute features and targets into the per-token cache.
#
# Usage:
#   ./scripts/training/run_cache_training.sh [AGENT] [SPLIT]
#
# AGENT defaults to "dp_vla_agent"; pass "dp_vla_rl_agent" to additionally
# pre-run the Florence-2 encoder (needed by HDP RL training).
# SPLIT defaults to "trainval" for dp_vla_agent and "test" for the RL one.
# Extra Hydra overrides may be appended after the split argument.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/env.sh"

AGENT="${1:-dp_vla_agent}"
DEFAULT_SPLIT="navtrain"
SPLIT="${2:-${DEFAULT_SPLIT}}"
CACHE_DATA_LIST_PATH= # path to the data list json file

DEFAULT_RL_CHECKPOINT_PATH= # path to the default pretrained ckpt. Only used for RL branch

shift || true

if [ "${AGENT}" = "dp_vla_rl_agent" ]; then
    # DpVlaRlAgent.initialize() only loads the encoder when this flag is on.
    # Caching the RL branch requires pretrained HDP model to collect the input 
    # features in advance. 
    # If correctly configured, you should see something like "Loaded Dp-VLA checkpoint from ${CHECKPOINT_PATH}"
    export DP_VLA_RL_EVAL=1
    PY_ENTRY="hdp_navsim/training/run_cache_training_multi_node.py"
    LAUNCHER=(torchrun --standalone --nproc_per_node "${DP_VLA_NPROC:-1}")
    CACHE_PATH="${HDP_RL_CACHE_PATH}"
    CHECKPOINT_PATH="${3:-${DEFAULT_RL_CHECKPOINT_PATH}}" 
else
    # Instead in the IL branch the caching process does not require pretrained HDP model
    PY_ENTRY="hdp_navsim/training/run_cache_training.py"
    LAUNCHER=(python)
    CACHE_PATH="${HDP_NAVSIM_CACHE_PATH}"
    CHECKPOINT_PATH=null
fi

shift || true

"${LAUNCHER[@]}" "${HDP_NAVSIM_ROOT}/${PY_ENTRY}" \
    agent="${AGENT}" \
    experiment_name="cache_${AGENT}" \
    train_test_split="${SPLIT}" \
    cache_path="${CACHE_PATH}" \
    force_cache_computation=true \
    agent.config.pretrain_config.checkpoint_path="${CHECKPOINT_PATH}" \
    +cache_data_list_path="${CACHE_DATA_LIST_PATH}" \
    "$@"
