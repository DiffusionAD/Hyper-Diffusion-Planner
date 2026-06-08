from typing import Any, Dict, List, Tuple
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


class DpVlaTargetBuilder(AbstractTargetBuilder):
    """Output target builder for Dp-VLA."""

    def __init__(
        self, 
        trajectory_sampling: TrajectorySampling, 
    ) -> None:
        """
        Initializes target builder.
        :param trajectory_sampling: the trajectory sampling to use for future waypoints.
        """
        self._trajectory_sampling = trajectory_sampling

    @classmethod
    def get_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "dp_vla_target"

    def compute_targets_training(self, scene: Scene) -> Dict[str, torch.Tensor]:

        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses).poses
        )
        ego_future_trajectory = torch.cat(
        [
            trajectory[..., :2],
            torch.stack(
                [trajectory[..., 2].cos(), trajectory[..., 2].sin()], dim=-1
            ),
        ],
        dim=-1,
        )
              
        return {
            "ego_future_trajectory": ego_future_trajectory, # 8 * 4
        }
        

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        return self.compute_targets_training(scene)