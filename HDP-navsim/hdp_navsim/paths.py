"""Single Python entry point for every HDP-NavSim filesystem path.

Every other Python module that needs a hard-coded path or directory should
import the corresponding helper from this file rather than reading
``os.environ`` directly. The shell counterpart is :file:`env.sh` at the repo
root; both files reference the **same** environment variables, so editing
``env.sh`` is enough to redirect the whole pipeline to a new machine.

Variables (resolution order: explicit env var -> default):

================================  =======================================
Variable                          Purpose
================================  =======================================
``HDP_NAVSIM_ROOT``            Root of THIS repository.
``NAVSIM_DEVKIT_ROOT``            Upstream NAVSIM repo (Hydra search path).
``OPENSCENE_DATA_ROOT``           OpenScene / NAVSIM raw input data.
``NUPLAN_MAPS_ROOT``              Maps for nuplan / NAVSIM.
``NUPLAN_MAP_VERSION``            Map schema version (e.g. ``nuplan-maps-v1.0``).
``DP_VLA_ENCODER_PATH``           Florence-2 (or HF) backbone checkpoint.
``NAVSIM_EXP_ROOT``               Top-level experiment outputs.
``HDP_NAVSIM_CACHE_PATH``      Per-token feature/target cache.
``NAVSIM_METRIC_CACHE_PATH``      Per-token PDM metric cache.
``TENSORBOARD_LOG_PATH``          TensorBoard log directory.
``HDP_NAVSIM_ASSETS_DIR``      L/S/R/U.png icons (visualisation only).
``DP_VLA_RL_EVAL``            ``"1"`` to load eval ckpt in DpVlaRlAgent.
================================  =======================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Optional[Path] = None) -> Optional[Path]:
    """Read ``$name`` as a :class:`pathlib.Path`; return ``default`` if unset."""
    value = os.environ.get(name)
    if value:
        return Path(value)
    return default


# ---------------------------------------------------------------------------
# Code roots
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Root of this repository (overridable via ``HDP_NAVSIM_ROOT``)."""
    return _env_path("HDP_NAVSIM_ROOT", _REPO_ROOT)


def navsim_devkit_root() -> Path:
    """Upstream NAVSIM checkout. Required by the run_pdm_score scripts."""
    p = _env_path("NAVSIM_DEVKIT_ROOT")
    if p is None:
        raise RuntimeError(
            "NAVSIM_DEVKIT_ROOT is not set; source env.sh or export it manually."
        )
    return p


# ---------------------------------------------------------------------------
# Dataset roots
# ---------------------------------------------------------------------------

def openscene_data_root() -> Path:
    """OpenScene root, contains ``navsim_logs/{split}`` and ``sensor_blobs/{split}``."""
    p = _env_path("OPENSCENE_DATA_ROOT")
    if p is None:
        raise RuntimeError(
            "OPENSCENE_DATA_ROOT is not set; source env.sh or export it manually."
        )
    return p


def nuplan_maps_root() -> Path:
    p = _env_path("NUPLAN_MAPS_ROOT")
    if p is None:
        raise RuntimeError(
            "NUPLAN_MAPS_ROOT is not set; source env.sh or export it manually."
        )
    return p


def encoder_path() -> str:
    """Florence-2 / HF AutoModelForCausalLM identifier or local directory."""
    return os.environ.get("DP_VLA_ENCODER_PATH", "microsoft/Florence-2-large")


# ---------------------------------------------------------------------------
# Experiment outputs
# ---------------------------------------------------------------------------

def navsim_exp_root() -> Path:
    """Top-level experiment directory; sub-paths default below it."""
    return _env_path("NAVSIM_EXP_ROOT", repo_root() / "navsim-exp")


def cache_path() -> Path:
    """Per-token feature / target cache (``{exp_root}/test_cache`` by default)."""
    return _env_path("HDP_NAVSIM_CACHE_PATH", navsim_exp_root() / "test_cache")


def metric_cache_path() -> Path:
    """Per-token PDM metric cache (``{exp_root}/metric_cache`` by default)."""
    return _env_path("NAVSIM_METRIC_CACHE_PATH", navsim_exp_root() / "metric_cache")


def tensorboard_log_path() -> Path:
    return _env_path("TENSORBOARD_LOG_PATH", navsim_exp_root() / "tensorboard_logs")


# ---------------------------------------------------------------------------
# Repo-relative defaults (data list / scene-filter JSON, etc.)
# ---------------------------------------------------------------------------

def data_list_path(split: str = "test") -> Path:
    """Default scene-token JSON shipped with the repo (``test.json``, etc.)."""
    return repo_root() / "hdp_navsim" / "training" / "training_utils" / f"{split}.json"


def scene_filter_tokens_path(name: str) -> Path:
    """Optional ``log_names_tokens_path`` JSON used by scene filters."""
    return (
        repo_root()
        / "hdp_navsim"
        / "config"
        / "train_test_split"
        / "scene_filter"
        / f"{name}.json"
    )


# ---------------------------------------------------------------------------
# Optional knobs
# ---------------------------------------------------------------------------

def assets_dir() -> Optional[Path]:
    """Direction-icon asset directory used by visualisation; ``None`` if unset."""
    return _env_path("HDP_NAVSIM_ASSETS_DIR")


def is_rl_evaluation() -> bool:
    """``True`` iff ``DP_VLA_RL_EVAL=1``; controls eval-time ckpt loading."""
    return os.environ.get("DP_VLA_RL_EVAL", "0") == "1"


__all__ = [
    "repo_root",
    "navsim_devkit_root",
    "openscene_data_root",
    "nuplan_maps_root",
    "encoder_path",
    "navsim_exp_root",
    "cache_path",
    "metric_cache_path",
    "tensorboard_log_path",
    "data_list_path",
    "scene_filter_tokens_path",
    "assets_dir",
    "is_rl_evaluation",
]
