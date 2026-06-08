"""Distributed PDM scoring + replay-buffer trajectory augmentation helpers.

These utilities are used by :class:`DpVlaRlAgent` during HDP RL training and
by :func:`hdp_navsim.visualization.plots` for trajectory metric
computations. They do **not** depend on the model backbone, so they live next
to the agents rather than inside ``model/``.

Heavy work (per-token PDM evaluation) is dispatched on a node-local Ray cluster
spun up by :func:`init_ray_local_per_node`. ``parallel_pdm_scores`` returns
results in input order so downstream code can index them by row.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import ray
import torch
import torch.distributed as dist

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score


logger = logging.getLogger(__name__)


METRIC_NAME_LABEL: Dict[str, str] = {
    "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC",
    "traffic_light_compliance": "TLC",
    "ego_progress": "EP",
    "time_to_collision_within_bound": "TTC",
    "comfort": "Comf",
    "driving_direction_compliance": "DDC",
    "lane_keeping": "LK",
    "history_comfort": "HC",
    "score": "PDMS",
    "pdm_score": "PDMS",
}


def init_ray_local_per_node(reserved_cpus: int = 4) -> None:
    """Start (or attach to) a per-node Ray cluster.

    On rank 0 of each node a fresh head is launched with ``num_cpus = os.cpu_count() - reserved_cpus``
    free for DataLoader / model workers. Other local ranks attach via
    ``address='auto'`` after a process-group barrier.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        num_cpus = max(1, os.cpu_count() - reserved_cpus)
        ray.init(
            num_cpus=num_cpus,
            include_dashboard=False,
            ignore_reinit_error=True,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    ray.init(address="auto", ignore_reinit_error=True)


@ray.remote
def compute_pdm_score_single(
    i: int,
    token: str,
    pred_np_i: np.ndarray,
    cache_dict: Dict[str, Any],
    simulator: Any,
    train_scorer: Any,
    proposal_sampling: Any,
) -> Tuple[int, float, Any]:
    """Score a single predicted trajectory; failures are logged and return NaN."""
    trajectory = Trajectory(pred_np_i)
    metric_cache = cache_dict[token]
    try:
        pdm_result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=proposal_sampling,
            simulator=simulator,
            scorer=train_scorer,
        )
        return i, pdm_result.score, pdm_result
    except Exception as exc:
        logger.warning("PDM score failed for token %s: %s", token, exc)
        return i, np.nan, None


def parallel_pdm_scores(
    pred_traj: torch.Tensor,
    tokens_list: List[str],
    cache_dict_ref: Any,
    simulator_ref: Any,
    train_scorer_ref: Any,
) -> Tuple[np.ndarray, List[Any]]:
    """Score a batch of predicted trajectories in parallel.

    ``simulator_ref`` must be a Ray object reference (used to fetch
    ``proposal_sampling`` once on the driver); ``cache_dict_ref`` and
    ``train_scorer_ref`` may be either raw objects (Ray auto-puts them) or
    object references. Returns ``(rewards, details)`` in original order.
    """
    pred_np = pred_traj.detach().cpu().numpy()
    proposal_sampling = ray.get(simulator_ref).proposal_sampling

    futures = [
        compute_pdm_score_single.remote(
            i, token, pred_np[i],
            cache_dict_ref, simulator_ref, train_scorer_ref,
            proposal_sampling,
        )
        for i, token in enumerate(tokens_list)
    ]

    results: List[Tuple[int, float, Any]] = []
    while futures:
        done, futures = ray.wait(futures, num_returns=1, timeout=None)
        if done:
            results.append(ray.get(done[0]))

    results.sort(key=lambda x: x[0])
    rewards = np.array([r[1] for r in results])
    details = [r[2] for r in results]
    return rewards, details


def augment_trajectory_batch(trajectories: torch.Tensor) -> torch.Tensor:
    """Apply small random along-track / lateral offsets in the local frame.

    ``trajectories`` is shaped ``(B, N, 4)`` with the last axis carrying
    ``[x, y, cos_yaw, sin_yaw]``. The heading (``cos_yaw``, ``sin_yaw``) is left
    untouched; only positions are perturbed by tangent / normal noise.
    """
    B = trajectories.shape[0]
    a = torch.randn(B, 1, device=trajectories.device) * 0.5
    b = torch.randn(B, 1, device=trajectories.device) * 0.5

    x = trajectories[..., 0]
    y = trajectories[..., 1]
    cos_yaw = trajectories[..., 2]
    sin_yaw = trajectories[..., 3]

    x_new = x + a * cos_yaw - b * sin_yaw
    y_new = y + a * sin_yaw + b * cos_yaw
    return torch.stack((x_new, y_new, cos_yaw, sin_yaw), dim=-1)
