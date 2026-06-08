"""Dp-VLA RL agent -- HDP training and evaluation.

Owns the full training loop: replay-buffer rollout, Ray-based PDM reward, the
reward-weighted diffusion loss, checkpoint loading, and optimizer / scheduler
plumbing. The backbone (:class:`DpVlaModel`) only provides ``encode`` /
``decode`` / ``generate`` primitives -- every gradient step is composed here.

Single modification entries enforced by this file:

- checkpoint loading            -> :func:`agent.dp_vla.utils.load_dp_vla_state_dict`
- sensor configuration          -> :func:`agent.dp_vla.utils.build_default_sensor_config`
- PDM simulator + scorer        -> :func:`agent.dp_vla.utils.build_pdm_components`
- device selection              -> :func:`agent.dp_vla.utils.get_default_device`
- model hyper-parameters        -> :class:`DpVlaConfig`
- PDM scoring (Ray)             -> :mod:`agent.dp_vla.scoring`
"""

from __future__ import annotations

import json
import logging
import lzma
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import ray
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import (
    AgentInput,
    SensorConfig,
    Trajectory,
)
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from hdp_navsim.agent.dp_vla.model.configuration_dp_vla import DpVlaConfig
from hdp_navsim.agent.dp_vla.model.language import DpVlaLanguagePreprocessor
from hdp_navsim.agent.dp_vla.model.modeling_dp_vla import DpVlaModel
from hdp_navsim.agent.dp_vla.model.rl_utils import (
    CacheExtractor,
    ReplayBuffer,
)
from hdp_navsim.agent.dp_vla.dp_vla_agent import (
    detached_integral,
    model_action_to_waypoint,
    prediction_target,
    prediction_to_x_start,
    should_compute_hybrid_loss,
    waypoint_to_model_action,
)
from hdp_navsim.agent.dp_vla.preprocessing.dp_vla_feature_builder import (
    DpVlaFeatureBuilder,
)
from hdp_navsim.agent.dp_vla.preprocessing.dp_vla_rl_feature_builder import (
    DpVlaRlFeatureBuilder,
)
from hdp_navsim.agent.dp_vla.preprocessing.dp_vla_rl_target_builder import (
    DpVlaRlTargetBuilder,
)
from hdp_navsim.agent.dp_vla.scoring import (
    METRIC_NAME_LABEL,
    augment_trajectory_batch,
    init_ray_local_per_node,
    parallel_pdm_scores,
)
from hdp_navsim.agent.dp_vla.utils import (
    build_default_sensor_config,
    build_pdm_components,
    get_default_device,
    load_dp_vla_state_dict,
)
from hdp_navsim.paths import is_rl_evaluation, openscene_data_root
from hdp_navsim.training.training_utils.dataloader import SceneLoader
from hdp_navsim.training.training_utils.hf_export import HFExportCheckpoint
from hdp_navsim.agent.dp_vla.preprocessing.dp_vla_rl_feature_builder import (
    DpVlaRlFeatureBuilder
)

from hdp_navsim.agent.dp_vla.preprocessing.dp_vla_rl_target_builder import (
    DpVlaRlTargetBuilder
)



logger = logging.getLogger(__name__)


# Keys of ``config.model`` that flow into ``DpVlaConfig``. Keeping the set
# explicit (rather than ``co_varnames``) makes typos visible at config load.
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


class DpVlaRlAgent(AbstractAgent):
    """Dp-VLA agent that performs HDP RL training (decoder full fine-tune)."""

    def __init__(
        self,
        config,
        target_builder: AbstractTargetBuilder,
        scene_filter,
        diffusion_sde,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.target_builder = target_builder
        self.scene_filter = scene_filter
        self.diffusion_sde = diffusion_sde

        # Visualisation helper -- loads the raw scenes for replay-style val plots.
        data_root = openscene_data_root()
        split = str(_get(config, "split", "test"))
        self.scene_loader = SceneLoader(
            data_root / f"navsim_logs/{split}",
            data_root / f"sensor_blobs/{split}",
            scene_filter,
            sensor_config=SensorConfig.build_all_sensors(),
        )

        # State populated lazily by ``initialize_training`` / ``initialize``.
        self.model: Optional[DpVlaModel] = None
        self.text_proprocessor: Optional[DpVlaLanguagePreprocessor] = None
        self.optimizer_params: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ navsim API

    def name(self) -> str:
        return "DpVla_RL_Agent"

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [self.target_builder]

    def get_feature_name(self) -> str:
        return DpVlaRlFeatureBuilder.get_unique_name()

    def get_target_name(self) -> str:
        return DpVlaRlTargetBuilder.get_unique_name()

    def get_sensor_config(self) -> SensorConfig:
        return build_default_sensor_config(self.scene_filter.num_history_frames)

    def get_feature_builders(
        self,
        only_decoder_mode: bool = True,
    ) -> List[AbstractFeatureBuilder]:
        """Feature builders used by the dataset.

        ``only_decoder_mode=True`` (the cache pass) pre-runs the VLM encoder
        and saves the hidden states; ``False`` returns the plain feature
        builder so the encoder runs online inside ``compute_trajectory``.
        """
        if only_decoder_mode:
            self.initialize_pretrain_model(with_encoder=True)
            # The diffusion decoder is unused during cache time -- drop it to
            # free a few hundred MB before forking workers.
            if hasattr(self.model, "decoder"):
                del self.model.decoder
            self.model.eval()
            return [DpVlaRlFeatureBuilder(
                vlm_backbone=self.model,
                text_proprocessor=self.text_proprocessor,
            )]
        self.model.eval()
        return [DpVlaFeatureBuilder()]

    # ------------------------------------------------------------------ build

    def _build_model(self, with_encoder: bool) -> DpVlaModel:
        """Build a fresh :class:`DpVlaModel` from ``self.config.model`` + ``encoder_name``."""
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

    def initialize_pretrain_model(self, with_encoder: bool) -> None:
        """Construct the backbone and load the pretrain checkpoint (if any)."""
        device = get_default_device()

        pretrain_node = _get(self.config, "pretrain_config")
        pretrain_ckpt = _get(pretrain_node, "checkpoint_path")
        pretrain_cfg_path = _get(pretrain_node, "config_path")
        if pretrain_cfg_path is not None:
            logger.info("Pretrain config available at %s (used only for logging)", pretrain_cfg_path)

        self.model = self._build_model(with_encoder=with_encoder)
        self.text_proprocessor = self._build_language_preprocessor(device)

        if pretrain_ckpt is not None:
            load_dp_vla_state_dict(self.model, pretrain_ckpt)
        else:
            logger.info("DpVlaModel randomly initialised (no pretrain ckpt).")

        self.model = self.model.to(device, dtype=torch.float32)

    def initialize_training(self, lr: float) -> None:
        self.lr = lr
        self.initialize_pretrain_model(with_encoder=False)

        rl_cfg = _get(self.config, "rl_config")
        self._init_rl(rl_cfg)

        continue_ckpt = _get(self.config, "continue_training_ckpt")
        if continue_ckpt is not None:
            load_dp_vla_state_dict(self.model, continue_ckpt)
            logger.info("Loaded continue-training checkpoint from %s", continue_ckpt)

        self.optimizer_params = [
            {"params": list(self.model.decoder.parameters()),
             "lr": lr, "name": "decoder"},
        ]

        # Test-time PDM cache loader (used by ``validation_step``).
        self.test_metric_cache_loader = MetricCacheLoader(
            Path(_get(self.config, "test_metric_cache_path"))
        )

    def initialize(self) -> None:
        """Load weights for evaluation when ``DP_VLA_RL_EVAL=1``."""
        if not is_rl_evaluation():
            return

        test_node = _get(self.config, "test_config")
        test_ckpt = _get(test_node, "checkpoint_path")
        test_cfg_path = _get(test_node, "config_path")
        if test_cfg_path is not None:
            external = OmegaConf.load(test_cfg_path)
            logger.info("Using test config from %s", test_cfg_path)
            self.config = external.agent.config

        self.initialize_pretrain_model(with_encoder=True)

        if test_ckpt is None:
            logger.info("No test checkpoint provided; using pretrain model.")
        else:
            load_dp_vla_state_dict(self.model, test_ckpt)

        device = get_default_device()
        self.model = self.model.to(device, dtype=torch.float32)

    # ------------------------------------------------------------------ RL

    def _init_rl(self, rl_cfg) -> None:
        """Bind RL hyper-parameters and start the local Ray cluster."""
        self.replay_buffer_update_epoch = _get(rl_cfg, "replay_buffer_update_epoch")
        self.group_size = _get(rl_cfg, "group_size")
        self.diffusion_repeat_size = _get(rl_cfg, "diffusion_repeat_size")
        self.max_rollout_iter = _get(rl_cfg, "max_rollout_iter")
        self.rollout_steps = int(_get(rl_cfg, "rollout_steps", 5))

        self.replay_buffer = ReplayBuffer()

        with open(_get(rl_cfg, "data_list_path"), "r", encoding="utf-8") as f:
            data_list = json.load(f)
        self.cache_extractor = CacheExtractor(
            data_list=data_list,
            cache_path=_get(rl_cfg, "cache_path"),
            unique_name=DpVlaRlFeatureBuilder.get_unique_name(),
        )

        self.metric_cache_loader = MetricCacheLoader(Path(_get(rl_cfg, "metric_cache_path")))

        ps = _get(rl_cfg, "proposal_sampling")
        if ps is None:
            simulator, scorer, _ = build_pdm_components()
        else:
            simulator, scorer, _ = build_pdm_components(
                time_horizon=float(_get(ps, "time_horizon", 4.0)),
                interval_length=float(_get(ps, "interval_length", 0.1)),
            )

        runtime_node = _get(self.config, "runtime")
        reserved_cpus = int(_get(runtime_node, "ray_reserved_cpus", 4))
        init_ray_local_per_node(reserved_cpus=reserved_cpus)
        self.train_simulator_ref = ray.put(simulator)
        self.train_scorer_ref = ray.put(scorer)

        # Running statistics used to decide ``only_ep`` mode.
        self.avg_ep = 0.0
        self.total_num = 1
        self.only_ep = True
        self.current_epoch = 0

    def _activate_training_parameters(self) -> None:
        """Re-enable gradients on decoder params after a frozen rollout pass."""
        for p in self.model.decoder.parameters():
            p.requires_grad = True

    # ---------------------------------------------------------------- inference

    def compute_trajectory(
        self,
        agent_input: AgentInput,
        with_encoder: bool = False,
    ) -> Trajectory:
        """Sample a trajectory for a scene or cached feature dict.

        Reuse cached encoder states when available; otherwise run Florence-2
        online from either raw ``AgentInput`` or IL-cache features.
        """
        self.eval()
        device = get_default_device()

        with torch.no_grad():
            has_cached_encoder = (
                isinstance(agent_input, dict)
                and "encoder_output" in agent_input
            )
            needs_online_encoder = with_encoder or not has_cached_encoder

            if needs_online_encoder:
                if isinstance(agent_input, dict):
                    features = agent_input
                else:
                    features: Dict[str, torch.Tensor] = {}
                    for builder in self.get_feature_builders(only_decoder_mode=False):
                        features.update(builder.compute_features(agent_input))

                input_ids = self.text_proprocessor.encode_language(
                    [features["language"]]
                    if isinstance(features["language"], str)
                    else features["language"]
                ).to(device)

                image_obs = features["image_obs"].to(device)
                if image_obs.dim() == 5:
                    image_obs = image_obs.unsqueeze(0)

                proprio = features["history"].to(device)
                if proprio.dim() == 1:
                    proprio = proprio.unsqueeze(0)

                encoder_outputs = self.model.encode(input_ids, image_obs)
                predictions = self.model.generate(
                    diffusion_sde=self.diffusion_sde,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    proprio=proprio,
                    attention_mask=encoder_outputs.attention_mask,
                )
            else:
                encoder_output = agent_input["encoder_output"].to(device)
                if encoder_output.dim() == 2:
                    encoder_output = encoder_output.unsqueeze(0)

                proprio = agent_input["history"].to(device)
                if proprio.dim() == 1:
                    proprio = proprio.unsqueeze(0)

                predictions = self.model.generate(
                    diffusion_sde=self.diffusion_sde,
                    encoder_hidden_states=encoder_output,
                    proprio=proprio,
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
        """Sample ``batch_size`` trajectories from the same scene (stochastic)."""
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

    # ---------------------------------------------------------------- training

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        tokens_list: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Single training step.

        Alternates between two phases on a fixed epoch schedule:

        - rollout (``epoch % replay_buffer_update_epoch == 0``): sample
          trajectories, score them with PDM, push into the buffer and return
          only metrics (no ``"loss"`` key).
        - train step (every other epoch): pull a batch from the buffer and
          compute reward-weighted diffusion loss: ``exp(reward) * MSE``.
        """
        encoder_output = features["encoder_output"]
        proprio = features["history"]
        actions = targets["ego_future_trajectory"]

        if self.current_epoch % self.replay_buffer_update_epoch == 0:
            return self._rl_rollout(encoder_output, proprio, actions, tokens_list)
        return self._rl_train_step(encoder_output, proprio)

    @torch.no_grad()
    def _rl_rollout(
        self,
        encoder_output: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        tokens_list: List[str],
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        loss_dict: Dict[str, torch.Tensor] = {}

        G = self.group_size
        B = encoder_output.shape[0]
        device = encoder_output.device

        encoder_rep = encoder_output[:, None].repeat_interleave(G, 1)
        proprio_rep = proprio[:, None].repeat_interleave(G, 1)
        action_rep = action[:, None].repeat_interleave(G, 1)

        # Pre-load metric caches once per unique token, share via Ray object store.
        unique_tokens = set(tokens_list)
        metric_cache: Dict[str, Any] = {}
        for token in unique_tokens:
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, "rb") as f:
                metric_cache[token] = pickle.load(f)
        metric_cache_ref = ray.put(metric_cache)

        reward_record: Optional[torch.Tensor] = None
        details_record: Optional[Dict[str, torch.Tensor]] = None
        remaining_token: List[tuple] = list(enumerate(tokens_list))
        num_iter = 0

        while remaining_token and num_iter < self.max_rollout_iter and len(remaining_token) >= B:
            proc_index = [it[0] for it in remaining_token]
            proc_tokens = [it[1] for it in remaining_token]

            enc_remain = encoder_rep[proc_index].flatten(0, 1)
            prop_remain = proprio_rep[proc_index].flatten(0, 1)
            act_remain = action_rep[proc_index].flatten(0, 1)

            rollout_action = self.model.generate(
                diffusion_sde=self.diffusion_sde,
                encoder_hidden_states=enc_remain,
                proprio=prop_remain,
                steps=self.rollout_steps,
            )
            rollout_action = model_action_to_waypoint(
                rollout_action, self.model.config.kinematic_type,
            )

            if self.current_epoch < 5:
                rollout_action = augment_trajectory_batch(rollout_action)

            # Convert (cos_yaw, sin_yaw) back to a heading for PDM scoring.
            rollout_traj_xyh = torch.cat(
                [rollout_action[..., :2],
                 torch.atan2(rollout_action[..., 3:4], rollout_action[..., 2:3])],
                dim=-1,
            )
            tokens_list_rep = [t for t in proc_tokens for _ in range(G)]
            rewards, details = self._reward_fn(
                rollout_traj_xyh, tokens_list_rep, metric_cache_ref,
            )

            rollout_action = rollout_action.unflatten(0, (-1, G))
            rewards = rewards.unflatten(0, (-1, G))
            gt_action = act_remain.unflatten(0, (-1, G))

            filter_mask = (rewards > -1.) & ~rewards.isnan().any(dim=1)[:, None]
            filter_idx = torch.where(filter_mask)[0]
            filter_abs_idx = [proc_index[i] for i in filter_idx]

            remaining_token = [
                it for it in remaining_token if (filter_idx != it[0]).all()
            ]

            if reward_record is None:
                reward_record = rewards
            elif len(filter_idx) > 0:
                reward_record[filter_abs_idx] = rewards[filter_idx]

            if details_record is None:
                details_record = details
            elif len(filter_idx) > 0:
                for k, v in details.items():
                    details_record[k][filter_abs_idx] = v[filter_idx]

            if len(filter_idx) > 0:
                for token, rollout, gt, reward in zip(
                    [proc_tokens[i] for i in filter_idx],
                    rollout_action[filter_idx],
                    gt_action[filter_idx],
                    rewards[filter_idx],
                ):
                    self.replay_buffer.put(token, rollout, gt, reward)

            num_iter += 1

        loss_dict["Reward/group_max_reward"] = reward_record.max(dim=1).values.mean().to(device)
        loss_dict["Reward/group_min_reward"] = reward_record.min(dim=1).values.mean().to(device)
        loss_dict["Reward/group_rand_reward"] = reward_record[:, 0].mean().to(device)
        loss_dict["Reward/reward"] = reward_record.mean().to(device)
        loss_dict["Metric/Buffer_Size"] = torch.tensor(len(self.replay_buffer), device=device)
        loss_dict["Metric/retry"] = torch.tensor(num_iter, device=device)

        if details_record is not None and "ego_progress" in details_record:
            self.avg_ep += float(torch.sum(details_record["ego_progress"]).item())
        self.total_num += B

        if details_record is not None:
            for k, v in details_record.items():
                if k in METRIC_NAME_LABEL:
                    loss_dict[f"Metric/{METRIC_NAME_LABEL[k]}"] = v.mean()

        return loss_dict

    def _rl_train_step(
        self,
        encoder_output: torch.Tensor,
        proprio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.train()
        loss_dict: Dict[str, torch.Tensor] = {}
        loss_dict["Metric/Buffer_Size"] = torch.tensor(
            len(self.replay_buffer), device=encoder_output.device,
        )

        B = encoder_output.shape[0]
        G = self.group_size
        K = self.diffusion_repeat_size

        scene_tokens, rollout_actions, _, reward_abs = self.replay_buffer.get(B)
        rollout_actions = torch.cat(rollout_actions, dim=0)
        reward_abs = torch.stack(reward_abs, dim=0)

        encoder_outs, propris = self.cache_extractor.get_feature_from_tokens(scene_tokens)
        encoder_output = (
            torch.cat(encoder_outs, dim=0)
            .to(rollout_actions.device)
            .repeat_interleave(G, 0)
        )
        proprio = (
            torch.cat(propris, dim=0)
            .to(rollout_actions.device)
            .repeat_interleave(G, 0)
        )

        # Group-relative reward normalisation.
        rewards = (reward_abs - reward_abs.mean(dim=1, keepdim=True)) / (
            reward_abs.std(dim=1, keepdim=True) + 1e-6
        )
        rewards = rewards.reshape(-1)

        encoder_output = encoder_output.repeat_interleave(K, 0)
        proprio = proprio.repeat_interleave(K, 0)
        rollout_actions = rollout_actions.repeat_interleave(K, 0)
        rewards = rewards.repeat_interleave(K, 0)
        attention_mask = torch.ones(
            encoder_output.shape[:2],
            dtype=torch.bool, device=encoder_output.device,
        )

        # ---- Reward-weighted diffusion loss: exp(reward) * MSE ----
        model_type = self.model.config.model_type
        kinematic_type = self.model.config.kinematic_type
        model_actions = waypoint_to_model_action(rollout_actions, kinematic_type)
        action_with_noise, time, target = self.diffusion_sde.sample(model_actions)
        pred = self.model.decode(
            action_with_noise, time, proprio, encoder_output, attention_mask,
        )
        target_prediction = prediction_target(
            model_type, model_actions, time, target, self.diffusion_sde,
        )
        per_sample_mse = F.mse_loss(
            pred, target_prediction, reduction="none",
        ).mean(dim=[-1, -2])
        weights = torch.exp(rewards)
        reward_weighted_mse = (per_sample_mse * weights).mean()
        loss_dict["loss/reward_weighted_mse"] = reward_weighted_mse

        hybrid_loss_weight = float(_get(self.config, "hybrid_loss_weight", 0.0))
        if should_compute_hybrid_loss(kinematic_type, hybrid_loss_weight):
            pred_x_start = prediction_to_x_start(
                model_type, pred, action_with_noise, time, self.diffusion_sde,
            )
            detach_window_size = int(_get(self.config, "detach_window_size", 1))
            pred_xy = detached_integral(
                pred_x_start[..., :2],
                detach_window_size=max(detach_window_size, 1),
            )
            per_sample_waypoint_mse = F.mse_loss(
                pred_xy, rollout_actions[..., :2], reduction="none",
            ).mean(dim=[-1, -2])
            reward_weighted_waypoint_mse = (per_sample_waypoint_mse * weights).mean()
        else:
            reward_weighted_waypoint_mse = pred.new_zeros(())

        loss_dict["loss/reward_weighted_waypoint_mse"] = reward_weighted_waypoint_mse
        loss_dict["loss"] = (
            reward_weighted_mse
            + hybrid_loss_weight * reward_weighted_waypoint_mse
        )

        self._activate_training_parameters()
        return loss_dict

    def _reward_fn(
        self,
        pred_traj: torch.Tensor,
        tokens_list: List[str],
        cache_dict_ref,
    ):
        rewards, details = parallel_pdm_scores(
            pred_traj, tokens_list, cache_dict_ref,
            self.train_simulator_ref, self.train_scorer_ref,
        )

        details_dicts = [d.__dict__ for d in details]
        result_details = {k: [d[k] for d in details_dicts] for k in details_dicts[0].keys()}
        for r in details_dicts:
            for k, v in r.items():
                result_details[k].append(v)

        result_details = {
            k: torch.tensor(v, device=pred_traj.device) for k, v in result_details.items()
        }
        return (
            torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach(),
            result_details,
        )

    # ---------------------------------------------------------------- lifecycle

    def on_train_epoch_start(self, epoch: int) -> None:
        self.current_epoch = epoch
        if epoch % self.replay_buffer_update_epoch == 0:
            self.replay_buffer.clear()
            self.only_ep = self.avg_ep / max(self.total_num, 1) < 0.95
            self.avg_ep = 0.0
            self.total_num = 0

    def on_train_epoch_end(self, epoch: int) -> None:
        pass

    # ---------------------------------------------------------------- validation

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        try:
            self.model.eval()
            features, targets, token_list = batch
            encoder_output = features["encoder_output"]
            proprio = features["history"]

            with torch.no_grad():
                pred_action = self.model.generate(
                    diffusion_sde=self.diffusion_sde,
                    encoder_hidden_states=encoder_output,
                    proprio=proprio,
                )
                pred_action = model_action_to_waypoint(
                    pred_action, self.model.config.kinematic_type,
                )
                predictions = pred_action.detach().cpu().numpy()

            heading = np.arctan2(predictions[..., 3], predictions[..., 2])[..., None]
            poses = np.concatenate([predictions[..., :2], heading], axis=-1)

            unique_tokens = set(token_list)
            cache_dict: Dict[str, Any] = {}
            for token in unique_tokens:
                path = self.test_metric_cache_loader.metric_cache_paths[token]
                with lzma.open(path, "rb") as f:
                    cache_dict[token] = pickle.load(f)

            scores, results = parallel_pdm_scores(
                torch.tensor(poses),
                token_list,
                cache_dict,
                self.train_simulator_ref,
                self.train_scorer_ref,
            )

            df = pd.concat(results, ignore_index=True).drop(
                columns=["weighted_metrics", "weighted_metrics_array", "pdm_score"]
            ).to_dict(orient="list")
            df = {f"Metric/{k}": torch.tensor(v, device=encoder_output.device).mean()
                  for k, v in df.items()}
            df["Metric/PDMS"] = torch.tensor(scores, device=encoder_output.device).mean()
            return df
        except Exception as exc:
            logger.warning("validation_step failed: %s", exc)
            return {}

    # ---------------------------------------------------------------- optim

    def get_optimizers(self):
        return optim.AdamW(self.optimizer_params)

    def get_scheduler(self, optimizer, total_steps):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def get_training_callbacks(self, save_epochs: int = 10) -> List[pl.Callback]:
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
