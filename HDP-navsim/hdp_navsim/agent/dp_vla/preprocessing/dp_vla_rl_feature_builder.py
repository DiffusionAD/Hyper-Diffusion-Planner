"""Feature builder used while caching the offline RL dataset.

The base :class:`DpVlaFeatureBuilder` knows how to turn an ``AgentInput``
into ``(meta_images, meta_status)``; this subclass additionally runs the
frozen Florence-2 vision-language backbone so that the (heavy) encoder output
is cached to disk and the RL training loop only needs to load tensors.

The camera-stitching and ego-status helpers come straight from the base
class -- there is no implementation difference, so we do not redefine them
here.
"""

from __future__ import annotations

from typing import Dict

import torch

from navsim.common.dataclasses import AgentInput

from hdp_navsim.training.training_utils.dataset import (
    get_image,
    get_language_promt,
)

from .dp_vla_feature_builder import DpVlaFeatureBuilder


class DpVlaRlFeatureBuilder(DpVlaFeatureBuilder):
    """Feature builder that pre-computes the VLM encoder output."""

    def __init__(self, vlm_backbone, text_proprocessor):
        super().__init__()
        self.vlm_backbone = vlm_backbone
        self.vlm_backbone.eval()
        self.text_proprocessor = text_proprocessor

    @classmethod
    def get_unique_name(cls) -> str:
        return "dp_vla_rl_feature"

    def _forward_vlm_backbone(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.Tensor:
        """Run the Florence-2 vision tower + BART encoder once.

        Uses the transformers-style :meth:`DpVlaModel.encode` API rather than
        poking at the encoder's internals from the outside.
        """
        encoder_outputs = self.vlm_backbone.encode(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
        return encoder_outputs.last_hidden_state

    def compute_features_training(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        meta_images = self._get_camera_feature(agent_input)
        meta_status = self._get_state_feature(agent_input)

        device = next(self.vlm_backbone.encoder.parameters()).device
        image_obs = get_image(meta_images).unsqueeze(0).to(device)
        language = get_language_promt(meta_status)
        language_tokenized = self.text_proprocessor.encode_language(
            [language] if isinstance(language, str) else language
        ).to(device)

        encoder_output = self._forward_vlm_backbone(language_tokenized, image_obs)

        return {
            "encoder_output": encoder_output.detach().cpu(),
            "meta_status": meta_status,
        }

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        return self.compute_features_training(agent_input)
