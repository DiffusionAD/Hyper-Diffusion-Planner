#!/usr/bin/env bash
# Supervised diffusion training (DpVlaAgent).
#
# Auto-detects the launch mode:
#   - if MLP_WORKER_GPU / MLP_WORKER_NUM are set (multi-node job scheduler),
#     it forwards them to torchrun;
#   - otherwise it falls back to torchrun --standalone with one process per
#     visible GPU (DP_VLA_NPROC, default 1).
#
# Usage:
#   ./scripts/training/run_training.sh [hydra_overrides...]
#
# Common overrides:
#   train_test_split=navtrain
#   dataloader.params.batch_size=4
#   lightning_agent.params.lr=1e-4
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/env.sh"

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
    agent=dp_vla_agent \
    experiment_name="${DP_VLA_EXP_NAME:-dp_vla_agent}" \
    train_test_split="${DP_VLA_SPLIT:-trainval}" \
    +trainer.params.devices="${DEVICES_PER_NODE}" \
    "$@"
