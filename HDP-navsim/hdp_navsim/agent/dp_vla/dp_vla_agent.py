"""Dp-VLA base agent -- plain diffusion (MSE) training.

A thin wrapper around :class:`DpVlaModel` that performs supervised diffusion
training: sample a noise level via :class:`DiffusionSDE`, ask the backbone to
predict the noise, and minimise the MSE between prediction and ground truth.
No LoRA, no classifier-free guidance, no replay buffer -- those live in
:class:`DpVlaRlAgent`.

Single modification entries used here:

- checkpoint loading            -> :func:`agent.dp_vla.utils.load_dp_vla_state_dict`
- sensor configuration          -> :func:`agent.dp_vla.utils.build_default_sensor_config`
- device selection              -> :func:`agent.dp_vla.utils.get_default_device`
- model hyper-parameters        -> :class:`DpVlaConfig`
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from hdp_navsim.agent.dp_vla.model.configuration_dp_vla import DpVlaConfig
from hdp_navsim.agent.dp_vla.model.language import DpVlaLanguagePreprocessor
from hdp_navsim.agent.dp_vla.model.modeling_dp_vla import DpVlaModel
from hdp_navsim.agent.dp_vla.utils import (
    build_default_sensor_config,
    get_default_device,
    load_dp_vla_state_dict,
)

from hdp_navsim.agent.dp_vla.preprocessing.dp_vla_feature_builder import (
    DpVlaFeatureBuilder
)
from hdp_navsim.agent.dp_vla.preprocessing.dp_vla_target_builder import (
    DpVlaTargetBuilder
)
from hdp_navsim.training.training_utils.hf_export import HFExportCheckpoint

logger = logging.getLogger(__name__)


_PREDICTION_TYPES = {"noise", "score", "x_start"}


_MODEL_CFG_KEYS = {
    "hidden_size", "depth", "num_heads", "num_actions",
    "dim_action", "dim_y", "mlp_ratio", "cfg_scale",
    "model_type", "kinematic_type",
    "lora_r", "lora_alpha", "lora_dropout",
}


def _get(node, key: str, default=None):
    """Tolerant getter for OmegaConf / SimpleNamespace / dict configs."""
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _extract_model_overrides(model_node) -> Dict[str, Any]:
    """Pull the recognised DpVlaConfig keys out of ``model_node``.

    Works for raw dicts (``_convert_: 'all'``) and OmegaConf DictConfigs alike,
    silently ignoring any unknown key so users can't accidentally inject
    arbitrary arguments into the backbone constructor.
    """
    if model_node is None:
        return {}
    if OmegaConf.is_config(model_node):
        items = OmegaConf.to_container(model_node, resolve=True) or {}
    elif isinstance(model_node, dict):
        items = model_node
    else:
        items = {k: getattr(model_node, k) for k in dir(model_node) if not k.startswith("_")}
    return {k: v for k, v in items.items() if k in _MODEL_CFG_KEYS}



def waypoint_to_diff(actions: torch.Tensor) -> torch.Tensor:
    """Convert (x, y, cos, sin) waypoints to (dx, dy, cos, sin)."""
    xy = actions[..., :2]
    origin = torch.zeros_like(xy[..., :1, :])
    prev_xy = torch.cat([origin, xy[..., :-1, :]], dim=-2)
    return torch.cat([xy - prev_xy, actions[..., 2:4]], dim=-1)


def diff_to_waypoint(actions: torch.Tensor) -> torch.Tensor:
    """Convert (dx, dy, cos, sin) actions back to (x, y, cos, sin)."""
    xy = torch.cumsum(actions[..., :2], dim=-2)
    return torch.cat([xy, actions[..., 2:4]], dim=-1)


def _time_coeff(coeff: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return coeff.to(device=ref.device, dtype=ref.dtype).view(-1, 1, 1)


def waypoint_to_model_action(
    actions: torch.Tensor,
    kinematic_type: str,
) -> torch.Tensor:
    """Convert waypoint targets to the representation predicted by the model."""
    if kinematic_type == "diff":
        return waypoint_to_diff(actions)
    if kinematic_type == "waypoint":
        return actions
    raise ValueError(f"Unsupported kinematic_type: {kinematic_type!r}")


def model_action_to_waypoint(
    actions: torch.Tensor,
    kinematic_type: str,
) -> torch.Tensor:
    """Convert model-space actions to absolute waypoints."""
    if kinematic_type == "diff":
        return diff_to_waypoint(actions)
    if kinematic_type == "waypoint":
        return actions
    raise ValueError(f"Unsupported kinematic_type: {kinematic_type!r}")


def should_compute_hybrid_loss(
    kinematic_type: str,
    hybrid_loss_weight: float,
) -> bool:
    """Hybrid waypoint loss only applies to integrated diff predictions."""
    return kinematic_type == "diff" and hybrid_loss_weight > 0


def prediction_target(
    model_type: str,
    actions: torch.Tensor,
    time: torch.Tensor,
    target: Dict[str, torch.Tensor],
    diffusion_sde,
) -> torch.Tensor:
    if model_type == "noise":
        return target["noise"]
    if model_type == "x_start":
        return actions
    if model_type == "score":
        sigma = _time_coeff(diffusion_sde.sde.marginal_std(time), actions)
        return -target["noise"] / sigma
    raise ValueError(f"Unsupported model_type: {model_type!r}")


def prediction_to_x_start(
    model_type: str,
    prediction: torch.Tensor,
    action_with_noise: torch.Tensor,
    time: torch.Tensor,
    diffusion_sde,
) -> torch.Tensor:
    if model_type == "x_start":
        return prediction

    alpha = _time_coeff(diffusion_sde.sde.marginal_alpha(time), prediction)
    sigma = _time_coeff(diffusion_sde.sde.marginal_std(time), prediction)
    if model_type == "noise":
        pred_noise = prediction
    elif model_type == "score":
        pred_noise = -sigma * prediction
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    return (action_with_noise - sigma * pred_noise) / alpha


def prediction_to_supervision(
    model_type: str,
    supervision_type: str,
    prediction: torch.Tensor,
    action_with_noise: torch.Tensor,
    time: torch.Tensor,
    diffusion_sde,
) -> torch.Tensor:
    """Convert a model prediction to the parameterisation used by the loss."""
    if model_type not in _PREDICTION_TYPES:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    if supervision_type not in _PREDICTION_TYPES:
        raise ValueError(
            f"Unsupported supervision_type: {supervision_type!r}; "
            f"expected one of {sorted(_PREDICTION_TYPES)}"
        )
    if model_type == supervision_type:
        return prediction

    alpha = _time_coeff(diffusion_sde.sde.marginal_alpha(time), prediction)
    sigma = _time_coeff(diffusion_sde.sde.marginal_std(time), prediction)

    if model_type == "noise":
        pred_noise = prediction
    elif model_type == "x_start":
        pred_noise = (action_with_noise - alpha * prediction) / sigma
    else:
        pred_noise = -sigma * prediction

    if supervision_type == "noise":
        return pred_noise
    if supervision_type == "score":
        return -pred_noise / sigma
    return (action_with_noise - sigma * pred_noise) / alpha


def detached_integral(u, detach_window_size=1):
    # u: (B, T=8, D)
    cum_detach = torch.cumsum(u.detach(), dim=-2)
    cum_normal = torch.cumsum(u, dim=-2)

    # number of gradient from previous timesteps contained in:
    # shifted: [0, 1, 2, ..., window_size-1, window_size, ...., T] ->
    # shifted: [T-window_size+1, T-window_size+2, ...,T, 0, 1, 2, ...., T - window_size] ->
    # sum_recent: [0, 1, 2, ..., window_size-1, window_size, ...., window_size]
    shifted = torch.roll(cum_normal, shifts=detach_window_size, dims=-2)
    shifted[..., :detach_window_size, :] = 0
    sum_recent = cum_normal - shifted

    cum_detach_shifted = torch.roll(cum_detach, shifts=detach_window_size, dims=-2)
    cum_detach_shifted[..., :detach_window_size, :] = 0

    cumulative_sum = cum_detach_shifted + sum_recent
    return cumulative_sum

class DpVlaAgent(AbstractAgent):
    """Dp-VLA agent for plain diffusion training (no RL, no LoRA)."""

    def __init__(
        self,
        config,
        feature_builder: AbstractFeatureBuilder,
        target_builder: AbstractTargetBuilder,
        diffusion_sde,
    ):
        super().__init__()
        self.config = config
        self.feature_builder = feature_builder
        self.target_builder = target_builder
        self.diffusion_sde = diffusion_sde

    # ------------------------------------------------------------------ navsim API

    def name(self) -> str:
        return "DpVla_Agent"

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [self.target_builder]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [self.feature_builder]

    def get_feature_name(self) -> str:
        return DpVlaFeatureBuilder.get_unique_name()

    def get_target_name(self) -> str:
        return DpVlaTargetBuilder.get_unique_name()

    def get_sensor_config(self) -> SensorConfig:
        # The legacy DpVlaAgent always loaded four history frames; preserve
        # that exact behaviour rather than reaching into a scene filter that
        # this agent does not own.
        return build_default_sensor_config(num_history_frames=4)

    # ------------------------------------------------------------------ build

    def _build_model(self, with_encoder: bool = True) -> DpVlaModel:
        dp_vla_config = DpVlaConfig(
            encoder_name=_get(self.config, "encoder_name"),
            with_encoder=with_encoder,
            **_extract_model_overrides(_get(self.config, "model")),
        )
        return DpVlaModel(dp_vla_config)

    def _build_language_preprocessor(self, device: torch.device) -> DpVlaLanguagePreprocessor:
        lang_node = _get(self.config, "language")
        max_length = int(_get(lang_node, "max_length", 40))
        return DpVlaLanguagePreprocessor(
            encoder_name=_get(self.config, "encoder_name"),
            device=str(device),
            max_length=max_length,
        )

    def initialize_training(self, lr: float) -> None:
        self.lr = lr
        device = get_default_device()
        self.model = self._build_model(with_encoder=True)
        self.text_proprocessor = self._build_language_preprocessor(device)

        # Lower lr for the (pretrained) VLM encoder, full lr for the diffusion head.
        encoder_params = list(self.model.encoder.parameters())
        encoder_ids = set(map(id, encoder_params))
        other_params = [p for p in self.model.parameters() if id(p) not in encoder_ids]
        self.optimizer_params = [
            {"params": encoder_params, "lr": lr / 10, "name": "vlm_encoder"},
            {"params": other_params, "lr": lr, "name": "decoder"},
        ]

    def initialize(self) -> None:
        test_node = _get(self.config, "test_config")
        ckpt_path = _get(test_node, "checkpoint_path")
        assert ckpt_path is not None, "need checkpoint for test"

        cfg_path = _get(test_node, "config_path")
        if cfg_path is not None:
            external = OmegaConf.load(cfg_path)
            logger.info("Using test config from %s", cfg_path)
            self.config = external.agent.config

        device = get_default_device()
        self.model = self._build_model(with_encoder=True)
        self.text_proprocessor = self._build_language_preprocessor(device)
        load_dp_vla_state_dict(self.model, ckpt_path)
        self.model = self.model.to(device, dtype=torch.float32)

    # ------------------------------------------------------------------ inference

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        device = get_default_device()

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        with torch.no_grad():
            input_ids = self.text_proprocessor.encode_language([features["language"]])
            image_obs = features["image_obs"].unsqueeze(0).to(device)
            proprio = features["history"].unsqueeze(0).to(device)
            encoder_outputs = self.model.encode(input_ids, image_obs)
            predictions = self.model.generate(
                diffusion_sde=self.diffusion_sde,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                proprio=proprio,
                sample_temperature=0.5,
                attention_mask=encoder_outputs.attention_mask,
            )

        predictions = model_action_to_waypoint(
            predictions, self.model.config.kinematic_type,
        )
        predictions = predictions.squeeze(0).detach().cpu().numpy()
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        poses = np.concatenate([predictions[..., :2], heading], axis=-1)

        return Trajectory(poses)

    def compute_batch_trajectory(
        self,
        agent_input: AgentInput,
        batch_size: int = 10,
    ) -> List[Trajectory]:
        self.eval()
        device = get_default_device()

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        with torch.no_grad():
            input_ids = self.text_proprocessor.encode_language([features["language"]])
            image_obs = features["image_obs"].unsqueeze(0).to(device)
            proprio = features["history"].unsqueeze(0).to(device)

            encoder_outputs = self.model.encode(input_ids, image_obs)
            enc_rep = encoder_outputs.last_hidden_state.repeat(batch_size, 1, 1)
            mask_rep = encoder_outputs.attention_mask.repeat(batch_size, 1)
            proprio_rep = proprio.expand(batch_size, *proprio.shape[1:])

            predictions = self.model.generate(
                diffusion_sde=self.diffusion_sde,
                encoder_hidden_states=enc_rep,
                proprio=proprio_rep,
                attention_mask=mask_rep,
                sample_temperature=0.5,
                steps=10,
            )

        predictions = model_action_to_waypoint(
            predictions, self.model.config.kinematic_type,
        )
        predictions = predictions.detach().cpu().numpy()
        heading = np.arctan2(predictions[..., 3], predictions[..., 2])[..., None]
        poses = np.concatenate([predictions[..., :2], heading], axis=-1)
        return [Trajectory(p) for p in poses]

    # ------------------------------------------------------------------ training

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        tokens_list: List[str] = None,  # accepted for API parity with DpVlaRlAgent
    ) -> Dict[str, torch.Tensor]:
        """One supervised diffusion step.

        Samples a noise level over model-space actions, runs encode + decode,
        and optimises the configured diffusion parameterisation.
        """
        input_ids = self.text_proprocessor.encode_language(features["language"])
        image_obs = features["image_obs"]
        proprio = features["history"]
        actions = targets["ego_future_trajectory"]
        kinematic_type = self.model.config.kinematic_type
        model_actions = waypoint_to_model_action(actions, kinematic_type)
        model_type = self.model.config.model_type
        supervision_type = _get(self.config, "supervision_type") or model_type

        action_with_noise, time, target = self.diffusion_sde.sample(model_actions)
        output = self.model(
            input_ids=input_ids,
            image_obs=image_obs,
            action_with_noise=action_with_noise,
            time=time,
            proprio=proprio,
        )
        prediction = output.prediction
        supervised_prediction = prediction_to_supervision(
            model_type,
            supervision_type,
            prediction,
            action_with_noise,
            time,
            self.diffusion_sde,
        )
        target_prediction = prediction_target(
            supervision_type, model_actions, time, target, self.diffusion_sde,
        )
        diffusion_loss = F.mse_loss(supervised_prediction, target_prediction)

        hybrid_loss_weight = float(_get(self.config, "hybrid_loss_weight", 0.0))
        if should_compute_hybrid_loss(kinematic_type, hybrid_loss_weight):
            pred_x_start = prediction_to_x_start(
                model_type, prediction, action_with_noise, time, self.diffusion_sde,
            )
            detach_window_size = int(_get(self.config, "detach_window_size", 1))
            pred_xy = detached_integral(
                pred_x_start[..., :2],
                detach_window_size=max(detach_window_size, 1),
            )
            waypoint_loss = F.mse_loss(pred_xy, actions[..., :2])
            loss = diffusion_loss + hybrid_loss_weight * waypoint_loss
        else:
            waypoint_loss = prediction.new_zeros(())
            loss = diffusion_loss

        return {
            "loss": loss,
            "diffusion_loss": diffusion_loss.detach(),
            "waypoint_loss": waypoint_loss.detach(),
        }

    # ------------------------------------------------------------------ lifecycle

    def on_train_epoch_start(self, epoch: int) -> None:
        pass

    def on_train_epoch_end(self, epoch: int) -> None:
        pass

    # ------------------------------------------------------------------ optim

    def get_optimizers(self):
        return optim.AdamW(self.optimizer_params)

    def get_scheduler(self, optimizer, total_steps):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def get_training_callbacks(self, save_epochs: int = 10) -> List[pl.Callback]:
        # Lightning ``.ckpt`` keeps optimiser state for resume; the HF export
        # writes a portable ``config.json`` + ``model.safetensors`` snapshot
        # for downstream loading via ``DpVlaModel.from_pretrained``.
        return [
            ModelCheckpoint(
                filename="epoch{epoch:02d}_step{step}",
                save_top_k=-1,
                every_n_epochs=save_epochs,
                save_last=True,
                save_on_train_epoch_end=True,
                auto_insert_metric_name=False,
            ),
            HFExportCheckpoint(
                export_dirname="hf_checkpoints",
                every_n_epochs=save_epochs,
                save_last=True,
                mode="full",
            ),
        ]
