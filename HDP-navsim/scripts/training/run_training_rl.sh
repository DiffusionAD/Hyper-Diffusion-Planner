#!/usr/bin/env bash
# HDP RL training (DpVlaRlAgent) on top of a pretrained DpVlaAgent.
#
# Same launch-mode auto-detection as run_training.sh: forwards MLP_* env vars
# to torchrun when present, otherwise falls back to torchrun --standalone with
# DP_VLA_NPROC processes (default 1).
#
# Usage:
#   ./scripts/training/run_training_rl.sh [hydra_overrides...]
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/env.sh"
PRETRAINED_CKPT_FILE= # path to the pretrained model, e.g. /home/user/huggingface/models--ZhengYinan2001--DP-VLA/snapshots/6cc2e40b61ed55b55b865f478a4ccae0d2434d8c
CACHE_DATA_LIST_PATH= # path to the data list json file, e.g. /home/user/DIPOLE-navsim/hdp_navsim/training/training_utils/navtrain.json

if [ -n "${MLP_WORKER_NUM:-}" ] && [ -n "${MLP_WORKER_GPU:-}" ]; then
    DEVICES_PER_NODE="${MLP_WORKER_GPU}"
    LAUNCHER=(torchrun
        --nproc_per_node "${DEVICES_PER_NODE}"
        --master_addr "${MLP_WORKER_0_HOST}"
        --node_rank "${MLP_ROLE_INDEX}"
        --master_port "${MLP_WORKER_0_PORT}"
        --nnodes "${MLP_WORKER_NUM}")
else
    DEVICES_PER_NODE="${DP_VLA_NPROC:-1}"
    LAUNCHER=(torchrun --standalone --nproc_per_node "${DEVICES_PER_NODE}")
fi

"${LAUNCHER[@]}" "${HDP_NAVSIM_ROOT}/hdp_navsim/training/run_training.py" \
    agent=dp_vla_rl_agent \
    experiment_name="${DP_VLA_EXP_NAME:-dp_vla_rl_agent}" \
    train_test_split="${DP_VLA_SPLIT:-smoke_test}" \
    +trainer.params.devices="${DEVICES_PER_NODE}" \
    cache_path="${HDP_RL_CACHE_PATH}" \
    dataloader.params.batch_size="${DP_VLA_BATCH_SIZE:-40}" \
    lightning_agent.params.lr="${DP_VLA_LR:-3e-4}" \
    agent.config.rl_config.data_list_path="${CACHE_DATA_LIST_PATH}" \
    +agent.config.pretrain_config.checkpoint_path="${PRETRAINED_CKPT_FILE}" \
    "$@"
