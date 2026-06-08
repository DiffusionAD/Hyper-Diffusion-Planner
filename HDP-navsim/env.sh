#!/usr/bin/env bash
# =============================================================================
# Single entry point for ALL HDP-NavSim paths.
#
# Source this file once (`source ./env.sh`) before launching any training,
# caching or evaluation script. Every other shell script in `scripts/` and
# every Hydra config under `hdp_navsim/config/` reads from these
# environment variables, so changing a path here propagates everywhere.
#
# Quick start on a new machine:
#   1. cp env.sh env.local.sh
#   2. Edit the "CHANGE ME" paths in env.local.sh (see below)
#   3. source env.local.sh
#
# Only two paths are required before training / evaluation:
#   NAVSIM_DEVKIT_ROOT   upstream NAVSIM checkout
#   OPENSCENE_DATA_ROOT  NAVSIM / OpenScene dataset root
#
# Everything else has sensible defaults (repo-relative outputs, HF model ID,
# derived cache / map paths). Override any variable in env.local.sh as needed.
# =============================================================================

# Resolve repo root from this file's location.
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# -----------------------------------------------------------------------------
# Code roots
# -----------------------------------------------------------------------------
# Root of THIS repository. Auto-detected; override only if you source from elsewhere.
export HDP_NAVSIM_ROOT="${HDP_NAVSIM_ROOT:-${_ENV_SH_DIR}}"

# CHANGE ME: upstream NAVSIM dev-kit checkout (Hydra search path + `navsim` package).
# Clone from https://github.com/autonomousvision/navsim
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/path/to/navsim}"

# -----------------------------------------------------------------------------
# Dataset roots (read-only inputs)
# -----------------------------------------------------------------------------
# CHANGE ME: OpenScene / NAVSIM raw data root.
# Expected layout:
#   ${OPENSCENE_DATA_ROOT}/navsim_logs/{split}
#   ${OPENSCENE_DATA_ROOT}/sensor_blobs/{split}
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/navsim-dataset}"

# nuPlan / NAVSIM maps. Defaults to ${OPENSCENE_DATA_ROOT}/maps; override if needed.
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"

# Florence-2 encoder: Hugging Face model ID, or a local snapshot directory.
# Optional override example:
#   /path/to/huggingface/models--microsoft--Florence-2-large/snapshots/<revision>
export DP_VLA_ENCODER_PATH="${DP_VLA_ENCODER_PATH:-microsoft/Florence-2-large}"

# -----------------------------------------------------------------------------
# Experiment roots (writable outputs)
# -----------------------------------------------------------------------------
# Top-level experiment directory. Defaults under the repo; point to a large disk
# in env.local.sh if caches / checkpoints grow too big.
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${HDP_NAVSIM_ROOT}/navsim-exp}"

# Cache for compute-features pre-cache (one .gz per token).
export HDP_NAVSIM_CACHE_PATH="${HDP_NAVSIM_CACHE_PATH:-${NAVSIM_EXP_ROOT}/training_cache}"
export HDP_RL_CACHE_PATH="${HDP_RL_CACHE_PATH:-${NAVSIM_EXP_ROOT}/rl_training_cache}"

# Cache for PDM metrics (one .lz per token, used by reward and PDMS eval).
export NAVSIM_METRIC_CACHE_PATH="${NAVSIM_METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}"

# TensorBoard log root for `run_training.py`.
export TENSORBOARD_LOG_PATH="${TENSORBOARD_LOG_PATH:-${NAVSIM_EXP_ROOT}/tensorboard_logs}"

# -----------------------------------------------------------------------------
# Optional / visualisation
# -----------------------------------------------------------------------------
# Direction-icon (L/S/R/U.png) directory used by
# `hdp_navsim.visualization.plots.plot_bev_with_agent_verbose`. Leave unset
# to skip the icons.
export HDP_NAVSIM_ASSETS_DIR="${HDP_NAVSIM_ASSETS_DIR:-${NAVSIM_EXP_ROOT}}"

# -----------------------------------------------------------------------------
# Runtime knobs
# -----------------------------------------------------------------------------
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Make the repo importable without `pip install -e .` (still recommended).
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${HDP_NAVSIM_ROOT}"

unset _ENV_SH_DIR

# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------
_hdp_navsim_paths_warn() {
    local var
    for var in "$@"; do
        if [ -z "${!var}" ] || [[ "${!var}" == /path/to/* ]]; then
            echo "[env.sh] WARNING: ${var} is not configured (currently '${!var}')." >&2
            echo "[env.sh]          Copy env.sh to env.local.sh and set your local paths." >&2
        fi
    done
}
_hdp_navsim_paths_warn \
    NAVSIM_DEVKIT_ROOT \
    OPENSCENE_DATA_ROOT
unset -f _hdp_navsim_paths_warn
