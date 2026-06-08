#!/usr/bin/env bash
# PDMS evaluation for the dp_vla agent. Override the two checkpoint
# paths via env vars or a CLI flag before launching.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/env.sh"

# Point the HF cache at the directory that already holds
# models--microsoft--Florence-2-large/snapshots/<sha>/...
export HF_HUB_CACHE="/home/tanty/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/tanty/huggingface"   # 兼容旧版 huggingface_hub
export TRANSFORMERS_CACHE="/home/tanty/huggingface"      # 兼容旧版 transformers
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Required: where to load the trained agent weights from.
: "${DP_VLA_CKPT:?Set DP_VLA_CKPT to a Lightning .ckpt path}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py" \
    train_test_split=navtest \
    experiment_name=pdm_score_dp_vla \
    agent=dp_vla_agent_base \
    worker=sequential \
    agent.config.test_config.checkpoint_path="${DP_VLA_CKPT}" \
    hydra.searchpath="[pkg://hdp_navsim.config, pkg://navsim.planning.script.config, pkg://navsim.planning.script.config.common]"