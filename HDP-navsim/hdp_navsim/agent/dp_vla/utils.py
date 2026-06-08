"""Shared helpers for the Dp-VLA agents.

This module is the **single modification point** for the four small pieces of
boilerplate that previously repeated across ``dp_vla_agent.py``,
``dp_vla_rl_agent.py`` and the model file:

- :func:`load_dp_vla_state_dict`      -- accepts both legacy Lightning ``.ckpt``
                                          files (with the ``agent.model.``
                                          prefix) and HuggingFace-style
                                          directories produced by
                                          :meth:`DpVlaModel.save_pretrained`.
- :func:`build_default_sensor_config` -- the standard "all cameras, history-
                                          steps from ``num_history_frames``,
                                          no lidar" sensor configuration.
- :func:`build_pdm_components`        -- bundled PDM simulator + scorer +
                                          proposal sampling.
- :func:`get_default_device`          -- centralised cuda/cpu pick.

Adding new helpers here is preferable to copy-pasting logic into the agents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from navsim.common.dataclasses import SensorConfig
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from hdp_navsim.agent.dp_vla.model.modeling_dp_vla import (
    LORA_SUBDIR,
    WEIGHTS_NAME,
)


logger = logging.getLogger(__name__)


def get_default_device() -> torch.device:
    """Return ``cuda`` when available, otherwise ``cpu``."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _looks_like_hf_dir(path: Path) -> bool:
    """A HuggingFace-style dump always carries a ``config.json``."""
    return path.is_dir() and (path / "config.json").exists()


def load_dp_vla_state_dict(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    prefix: str = "agent.model.",
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load weights into ``model`` from either a Lightning ``.ckpt`` or a
    HuggingFace-style directory.

    HuggingFace directories
    -----------------------
    When ``ckpt_path`` is a directory containing ``config.json``, the loader
    accepts every layout produced by :meth:`DpVlaModel.save_pretrained`:

    - ``model.safetensors``  -> full backbone (encoder + decoder)
    - ``pytorch_model.bin``  -> full backbone (legacy fallback)
    - ``lora/<adapter>/...`` -> PEFT-style LoRA adapters; loaded after
                                ``init_lora_adapter()`` is run on the model.

    Either of the above is sufficient (RL finetune dumps may carry only a
    ``lora/`` directory).

    Legacy Lightning ``.ckpt`` files
    --------------------------------
    Lightning prepends every key with ``agent.model.``; this helper strips
    that prefix and forwards to :meth:`torch.nn.Module.load_state_dict`.
    Missing / unexpected keys are logged rather than raised, so the helper
    works for partial backbones (e.g. with ``with_encoder=False`` or before
    :meth:`DpVlaModel.init_lora_adapter` has been called).
    """
    ckpt_path = Path(ckpt_path)

    if _looks_like_hf_dir(ckpt_path):
        return _load_hf_directory(model, ckpt_path, strict=strict)

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state

    cleaned = {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)

    if missing:
        logger.warning(
            "load_dp_vla_state_dict: %d missing key(s) (e.g. %s)",
            len(missing), missing[:3],
        )
    if unexpected:
        logger.warning(
            "load_dp_vla_state_dict: %d unexpected key(s) (e.g. %s)",
            len(unexpected), unexpected[:3],
        )
    logger.info("Loaded Dp-VLA checkpoint from %s", ckpt_path)
    return missing, unexpected


def _load_hf_directory(
    model: nn.Module,
    ckpt_dir: Path,
    strict: bool,
) -> Tuple[List[str], List[str]]:
    """Load HF-style dump into ``model`` in-place; mirrors
    :meth:`DpVlaModel.from_pretrained` but without re-instantiating the model.

    Used by callers that already built the backbone (matching the agent's
    runtime config) and only want to overlay the saved tensors.
    """
    from safetensors.torch import load_file as safetensors_load_file

    weights_st = ckpt_dir / WEIGHTS_NAME
    weights_pt = ckpt_dir / "pytorch_model.bin"

    missing: List[str] = []
    unexpected: List[str] = []
    if weights_st.exists():
        state_dict = safetensors_load_file(str(weights_st), device="cpu")
        m, u = model.load_state_dict(state_dict, strict=strict)
        missing.extend(m); unexpected.extend(u)
    elif weights_pt.exists():
        state_dict = torch.load(weights_pt, map_location="cpu")
        m, u = model.load_state_dict(state_dict, strict=strict)
        missing.extend(m); unexpected.extend(u)
    else:
        logger.info(
            "No backbone weights file in %s -- only LoRA (if any) will be loaded",
            ckpt_dir,
        )

    lora_dir = ckpt_dir / LORA_SUBDIR
    if lora_dir.is_dir() and hasattr(model, "load_lora_pretrained"):
        model.load_lora_pretrained(lora_dir)

    if missing:
        logger.warning(
            "load_dp_vla_state_dict[hf]: %d missing key(s) (e.g. %s)",
            len(missing), missing[:3],
        )
    if unexpected:
        logger.warning(
            "load_dp_vla_state_dict[hf]: %d unexpected key(s) (e.g. %s)",
            len(unexpected), unexpected[:3],
        )
    logger.info("Loaded Dp-VLA checkpoint from %s", ckpt_dir)
    return missing, unexpected


def build_default_sensor_config(num_history_frames: int) -> SensorConfig:
    """Default Dp-VLA sensor config: every camera at every history step, no lidar.

    Replaces the hard-coded ``history_steps = [0, 1, 2, 3]`` literal that was
    duplicated across both agents; the number of history frames is derived
    from the active scene filter.
    """
    history_steps = list(range(num_history_frames))
    return SensorConfig(
        cam_f0=history_steps,
        cam_l0=history_steps,
        cam_l1=history_steps,
        cam_l2=history_steps,
        cam_r0=history_steps,
        cam_r1=history_steps,
        cam_r2=history_steps,
        cam_b0=history_steps,
        lidar_pc=False,
    )


def build_pdm_components(
    time_horizon: float = 4.0,
    interval_length: float = 0.1,
) -> Tuple[PDMSimulator, PDMScorer, TrajectorySampling]:
    """Build the PDM simulator + scorer pair used for both training reward
    and validation evaluation. Returns the underlying ``TrajectorySampling`` as
    a third value so callers do not need to re-derive it.
    """
    proposal_sampling = TrajectorySampling(
        time_horizon=time_horizon,
        interval_length=interval_length,
    )
    simulator = PDMSimulator(proposal_sampling)
    scorer = PDMScorer(proposal_sampling, PDMScorerConfig())
    return simulator, scorer, proposal_sampling
