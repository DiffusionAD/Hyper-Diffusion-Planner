from typing import Any, Dict, List, Tuple
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder

from hdp_navsim.agent.dp_vla.preprocessing.dp_vla_target_builder import DpVlaTargetBuilder


class DpVlaRlTargetBuilder(DpVlaTargetBuilder):
    """Output target builder for Dp-VLA."""

    def __init__(
        self, 
        trajectory_sampling: TrajectorySampling, 
    ) -> None:
        """
        Initializes target builder.
        :param trajectory_sampling: the trajectory sampling to use for future waypoints.
        """
        super().__init__(trajectory_sampling)

    @classmethod
    def get_unique_name(cfg) -> str:
        """Inherited, see superclass."""
        return "dp_vla_rl_target"
