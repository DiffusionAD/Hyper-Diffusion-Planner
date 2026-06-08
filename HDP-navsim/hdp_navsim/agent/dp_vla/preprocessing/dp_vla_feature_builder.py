from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

import torch
from torchvision import transforms

from navsim.common.dataclasses import AgentInput
from navsim.common.enums import LidarIndex
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder

from hdp_navsim.training.training_utils.dataset import get_image, get_language_promt

class DpVlaFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for Dp-VLA."""

    def __init__(self):
        pass

    @classmethod
    def get_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "dp_vla_feature"

    def compute_features_training(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:

        features = {}

        features["meta_images"] = self._get_camera_feature(agent_input)

        features["meta_status"] = self._get_state_feature(agent_input)

        return features
    
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["image_obs"] = get_image(self._get_camera_feature(agent_input), training=False)

        meta_status = self._get_state_feature(agent_input)
        
        features["language"] = get_language_promt(meta_status)

        features['history'] = meta_status[:, :3].reshape(-1)

        return features

    def _get_camera_feature(self, agent_input: AgentInput):
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        l0 = cameras.cam_l0.image
        f0 = cameras.cam_f0.image
        r0 = cameras.cam_r0.image

        images = [l0, f0, r0]

        return images

    def _get_state_feature(self, agent_input: AgentInput):
        
        status_feature = []
        

        for frame_idx in range(len(agent_input.ego_statuses)):

            status = agent_input.ego_statuses[frame_idx]


            frame_status = torch.concatenate(
                [
                    torch.tensor(status.ego_pose, dtype=torch.float32), # x, y, heading
                    torch.tensor(status.driving_command, dtype=torch.float32), # going left, straight, right or unknown
                    torch.tensor(status.ego_velocity, dtype=torch.float32),    # vx, vy
                    torch.tensor(status.ego_acceleration, dtype=torch.float32),# ax, ay
                ],
            )

            status_feature.append(frame_status)

        return torch.stack(status_feature, dim=0)