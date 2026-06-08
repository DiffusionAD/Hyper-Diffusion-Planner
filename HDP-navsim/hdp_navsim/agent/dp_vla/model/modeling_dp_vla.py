"""Dp-VLA backbone -- pure architecture, transformers-style.

The model is intentionally **stateless w.r.t. training**: no losses, no replay
buffers, no Ray, no scorers. It exposes four primitives, mirroring the
encoder-decoder pattern used elsewhere in the ``transformers`` library:

- :meth:`encode`   -- multi-view vision + text  -> encoder hidden states
- :meth:`decode`   -- single-step noise prediction (optionally through LoRA)
- :meth:`forward`  -- ``encode`` then ``decode`` in one call
- :meth:`generate` -- iterative denoising via the supplied diffusion SDE

All training procedures live in
``hdp_navsim/agent/dp_vla/{dp_vla_agent,dp_vla_rl_agent}.py``.

The submodule names (``encoder``, ``decoder``, ``lora_decoder``) are part of
the public checkpoint contract; renaming them would break loading of any
state-dict saved before this refactor.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from transformers import AutoModelForCausalLM

from .configuration_dp_vla import DpVlaConfig
from .decoder import CustomDiT
from .diffusion_utils.dpm_solver_pytorch import model_wrapper

logger = logging.getLogger(__name__)


# Filenames follow the HuggingFace conventions so any external tool
# (``transformers``, ``peft``) can also pick up these dumps.
WEIGHTS_NAME = "model.safetensors"
LORA_SUBDIR = "lora"


def _log_load_diff(source: str, missing, unexpected) -> None:
    """Uniform "missing/unexpected" logging shared by the loaders."""
    missing = list(missing)
    unexpected = list(unexpected)
    if missing:
        logger.warning(
            "load_state_dict from %s: %d missing key(s), e.g. %s",
            source, len(missing), missing[:3],
        )
    if unexpected:
        logger.warning(
            "load_state_dict from %s: %d unexpected key(s), e.g. %s",
            source, len(unexpected), unexpected[:3],
        )


# ---------------------------------------------------------------------------
# Output dataclasses (transformers-style ModelOutput is overkill here)
# ---------------------------------------------------------------------------


@dataclass
class DpVlaEncoderOutput:
    """Result of :meth:`DpVlaModel.encode`."""

    last_hidden_state: torch.FloatTensor
    attention_mask: torch.LongTensor


@dataclass
class DpVlaModelOutput:
    """Result of :meth:`DpVlaModel.forward`."""

    prediction: torch.FloatTensor

    @property
    def noise_pred(self) -> torch.FloatTensor:
        return self.prediction

    encoder_hidden_states: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.LongTensor] = None


# ---------------------------------------------------------------------------
# LoRA target regex -- matches every projection inside a ``CustomDiT`` block.
# Pinned as a module constant so the agent never has to know the regex.
# ---------------------------------------------------------------------------
_LORA_TARGET_MODULES = (
    r"blocks\.\d+\.attn\.qkv"
    r"|blocks\.\d+\.attn\.proj"
    r"|blocks\.\d+\.mlp\.fc1"
    r"|blocks\.\d+\.mlp\.fc2"
    r"|blocks\.\d+\.cross_attn\.proj_q"
    r"|blocks\.\d+\.cross_attn\.proj_k"
    r"|blocks\.\d+\.cross_attn\.proj_v"
    r"|blocks\.\d+\.cross_attn\.proj"
    r"|blocks\.\d+\.cross_mlp\.fc1"
    r"|blocks\.\d+\.cross_mlp\.fc2"
)


class DpVlaModel(nn.Module):
    """Pure backbone of the Dp-VLA planner.

    Construction takes a single :class:`DpVlaConfig`; the diffusion SDE
    (``DPM-Solver`` wrapper) is passed at *call time* to :meth:`generate` so
    the model itself stays sampler-agnostic.
    """

    config_class = DpVlaConfig

    def __init__(self, config: DpVlaConfig) -> None:
        super().__init__()
        self.config = config

        if config.with_encoder:
            self.encoder = AutoModelForCausalLM.from_pretrained(
                config.encoder_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            # The Florence-2 language decoder / LM head are unused by Dp-VLA;
            # drop them to save ~hundreds of MB of weights.
            del self.encoder.language_model.model.decoder
            del self.encoder.language_model.lm_head

        self.decoder = CustomDiT(
            num_actions=config.num_actions,
            dim_action=config.dim_action,
            dim_y=config.dim_y,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
        )

    # ------------------------------------------------------------------ LoRA

    def init_lora_adapter(self) -> None:
        """Wrap :attr:`decoder` with PEFT, exposing ``positive`` / ``negative`` adapters.

        Idempotent: calling twice is a no-op so that ``initialize_training``
        followed by ``initialize`` (or vice versa) does not fail.
        """
        if getattr(self, "lora_decoder", None) is not None:
            return
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=_LORA_TARGET_MODULES,
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        self.lora_decoder = get_peft_model(
            self.decoder, lora_config, adapter_name="positive"
        )
        self.lora_decoder.add_adapter("negative", lora_config)

    # ---------------------------------------------------------- HF-style I/O

    def _backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return ``encoder.*`` + ``decoder.*`` weights (no PEFT wrapper).

        ``lora_decoder`` shares parameter storage with ``decoder`` (PEFT moves
        the original module under ``lora_decoder.base_model.model.*``), so we
        intentionally skip those keys to keep the ``model.safetensors`` file
        free of duplicates. Each surviving tensor is ``clone()``-d to break
        any remaining storage sharing (e.g. tied embeddings inside Florence-2)
        which the safetensors backend refuses to serialise.
        """
        full = self.state_dict()
        return {
            k: v.detach().cpu().clone().contiguous()
            for k, v in full.items()
            if not k.startswith("lora_decoder.")
        }

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        *,
        save_lora_only: bool = False,
        safe_serialization: bool = True,
    ) -> None:
        """Dump the model in a HuggingFace-style directory.

        ``save_lora_only=False`` (default, used for pretraining) writes
        ``config.json`` + ``model.safetensors`` (encoder + decoder weights).
        ``save_lora_only=True`` (used for RL finetuning) writes only
        ``config.json`` + a ``lora/`` subdirectory containing PEFT-style
        adapter dumps for every active LoRA adapter.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        self.config.save_pretrained(save_directory)

        if save_lora_only:
            if getattr(self, "lora_decoder", None) is None:
                raise RuntimeError(
                    "save_lora_only=True requires init_lora_adapter() first"
                )
            self.save_lora_pretrained(save_directory / LORA_SUBDIR)
            return

        state_dict = self._backbone_state_dict()
        if safe_serialization:
            safetensors_save_file(
                state_dict,
                str(save_directory / WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
        else:
            torch.save(state_dict, save_directory / "pytorch_model.bin")

        if getattr(self, "lora_decoder", None) is not None:
            # Convenient: the LoRA adapters travel alongside the full
            # backbone so a single directory is fully self-contained.
            self.save_lora_pretrained(save_directory / LORA_SUBDIR)

        logger.info("DpVlaModel saved to %s", save_directory)

    def save_lora_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
    ) -> None:
        """PEFT-style dump of every LoRA adapter currently registered.

        Produces ``save_directory/<adapter_name>/{adapter_config.json,
        adapter_model.safetensors}`` for each adapter (e.g. ``positive`` /
        ``negative``).
        """
        if getattr(self, "lora_decoder", None) is None:
            raise RuntimeError(
                "save_lora_pretrained requires init_lora_adapter() first"
            )
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        adapters = list(self.lora_decoder.peft_config.keys())
        previous = getattr(self.lora_decoder, "active_adapter", None)
        for name in adapters:
            self.lora_decoder.set_adapter(name)
            self.lora_decoder.save_pretrained(
                str(save_directory),
                selected_adapters=[name],
                safe_serialization=True,
            )
        if previous is not None:
            self.lora_decoder.set_adapter(previous)
        logger.info(
            "DpVlaModel LoRA adapters %s saved to %s", adapters, save_directory,
        )

    def load_lora_pretrained(
        self,
        load_directory: Union[str, os.PathLike],
    ) -> None:
        """Load LoRA adapter weights from a directory created by
        :meth:`save_lora_pretrained`.

        Calls :meth:`init_lora_adapter` first if needed (so freshly built
        models are usable). Loads every subdirectory whose name starts with a
        plausible adapter prefix; any subdirectory containing
        ``adapter_config.json`` is treated as a single adapter named after the
        folder.
        """
        load_directory = Path(load_directory)
        if not load_directory.is_dir():
            raise FileNotFoundError(f"LoRA directory not found: {load_directory}")

        if getattr(self, "lora_decoder", None) is None:
            self.init_lora_adapter()

        loaded: list[str] = []
        for sub in sorted(load_directory.iterdir()):
            if not sub.is_dir():
                continue
            if not (sub / "adapter_config.json").exists():
                continue
            adapter_name = sub.name
            self.lora_decoder.load_adapter(
                str(sub), adapter_name=adapter_name, is_trainable=True,
            )
            loaded.append(adapter_name)
        if not loaded:
            raise RuntimeError(
                f"No PEFT adapter subdirectories found under {load_directory}"
            )
        logger.info("Loaded LoRA adapters %s from %s", loaded, load_directory)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: Union[str, os.PathLike],
        *,
        with_encoder: Optional[bool] = None,
        lora_path: Optional[Union[str, os.PathLike]] = None,
        strict: bool = False,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "DpVlaModel":
        """Build a :class:`DpVlaModel` from a directory written by
        :meth:`save_pretrained` (or by an HF export callback).

        Parameters
        ----------
        load_directory:
            Directory containing ``config.json`` and either
            ``model.safetensors`` (full backbone, pretrain output) or only a
            ``lora/`` subfolder (RL finetune output).
        with_encoder:
            Override ``config.with_encoder``. ``None`` keeps whatever the
            saved config said. Setting ``False`` is handy on small machines:
            the Florence-2 vision tower is skipped entirely.
        lora_path:
            Optional separate directory holding LoRA adapters. When
            ``load_directory`` itself contains a ``lora/`` subfolder, it is
            picked up automatically; ``lora_path`` lets you point at an
            adapter dump produced from a different RL run.
        strict:
            Forwarded to :meth:`torch.nn.Module.load_state_dict`.
        """
        load_directory = Path(load_directory)
        if not load_directory.is_dir():
            raise FileNotFoundError(f"Not a directory: {load_directory}")

        config = DpVlaConfig.from_pretrained(load_directory)
        if with_encoder is not None:
            config.with_encoder = with_encoder

        model = cls(config)

        weights_safetensors = load_directory / WEIGHTS_NAME
        weights_pt = load_directory / "pytorch_model.bin"
        if weights_safetensors.exists():
            state_dict = safetensors_load_file(
                str(weights_safetensors), device=str(map_location),
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            _log_load_diff("safetensors", missing, unexpected)
        elif weights_pt.exists():
            state_dict = torch.load(weights_pt, map_location=map_location)
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            _log_load_diff("pytorch_model.bin", missing, unexpected)
        else:
            logger.info(
                "No backbone weights file in %s -- model will only carry the "
                "Florence-2 init plus any LoRA dump", load_directory,
            )

        # Auto-pick up co-located LoRA dump.
        embedded_lora = load_directory / LORA_SUBDIR
        if lora_path is None and embedded_lora.is_dir():
            lora_path = embedded_lora
        if lora_path is not None:
            model.load_lora_pretrained(lora_path)

        logger.info("DpVlaModel loaded from %s", load_directory)
        return model

    # -------------------------------------------------------------- encode

    def encode(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> DpVlaEncoderOutput:
        """Multi-view vision + text -> encoder hidden states.

        ``pixel_values`` is shaped ``(B, V, F, C, H, W)`` (``V`` cameras,
        ``F`` history frames). The Florence-2 vision tower is run on every
        ``(view, frame)`` slot, and the resulting tokens are merged into the
        BART encoder's input embeddings via ``_merge_input_ids_with_image_features``.
        """
        if not hasattr(self, "encoder"):
            raise RuntimeError("DpVlaModel was built with with_encoder=False")
        B, V, F = pixel_values.shape[:3]
        inputs_embeds = self.encoder.get_input_embeddings()(input_ids)
        image_features = self.encoder._encode_image(pixel_values.flatten(0, 2))
        N, C = image_features.shape[1:]
        inputs_embeds, attention_mask = (
            self.encoder._merge_input_ids_with_image_features(
                image_features.view(B, N * V * F, C), inputs_embeds,
            )
        )
        encoder_outputs = self.encoder.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        return DpVlaEncoderOutput(
            last_hidden_state=encoder_outputs[0],
            attention_mask=attention_mask,
        )

    # -------------------------------------------------------------- decode

    def decode(
        self,
        action_with_noise: torch.Tensor,
        time: torch.Tensor,
        proprio: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        adapter: Optional[str] = None,
    ) -> torch.Tensor:
        """Single-step noise prediction.

        ``adapter`` selects the parameters used for the decode call:

        - ``None`` or ``"base"`` -- raw backbone decoder. ``"base"`` requires
          :meth:`init_lora_adapter` to have been called; ``None`` works in
          either mode.
        - ``"positive"`` / ``"negative"`` -- the corresponding LoRA adapter.
        """
        if adapter == "base":
            if not hasattr(self, "lora_decoder"):
                raise RuntimeError(
                    'decode(adapter="base") requires init_lora_adapter() first'
                )
            with self.lora_decoder.disable_adapter():
                return self.lora_decoder(
                    action_with_noise, time,
                    y=proprio, c=encoder_hidden_states, c_mask=attention_mask,
                )
        if adapter is None or not hasattr(self, "lora_decoder"):
            return self.decoder(
                action_with_noise, time, proprio,
                encoder_hidden_states, attention_mask,
            )
        self.lora_decoder.set_adapter(adapter)
        return self.lora_decoder(
            action_with_noise, time,
            y=proprio, c=encoder_hidden_states, c_mask=attention_mask,
        )

    # -------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_obs: Optional[torch.FloatTensor] = None,
        action_with_noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        adapter: Optional[str] = None,
    ) -> DpVlaModelOutput:
        """One-shot encode + decode.

        Either ``(input_ids, image_obs)`` *or* ``encoder_hidden_states`` must be
        supplied. ``action_with_noise``, ``time`` and ``proprio`` are always
        required.
        """
        if encoder_hidden_states is None:
            encoder_outputs = self.encode(input_ids, image_obs)
            encoder_hidden_states = encoder_outputs.last_hidden_state
            attention_mask = encoder_outputs.attention_mask
        elif attention_mask is None:
            attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.bool, device=encoder_hidden_states.device,
            )

        prediction = self.decode(
            action_with_noise, time, proprio,
            encoder_hidden_states, attention_mask,
            adapter=adapter,
        )
        return DpVlaModelOutput(
            prediction=prediction,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

    # -------------------------------------------------------------- generate

    @torch.no_grad()
    def generate(
        self,
        *,
        diffusion_sde,
        encoder_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_actions: Optional[int] = None,
        dim_action: Optional[int] = None,
        steps: int = 10,
        sample_temperature: float = 0.5,
        cfg_scale: Optional[float] = None,
        use_base: bool = False,
    ) -> torch.Tensor:
        """Sample an action trajectory via iterative denoising.

        When ``use_base`` is true (or :meth:`init_lora_adapter` has never been
        called), the raw backbone decoder is used. Otherwise classifier-free
        guidance combines the ``positive`` and ``negative`` adapters with
        ``cfg_scale`` (defaulting to :attr:`config.cfg_scale`).
        """
        if attention_mask is None:
            attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.bool, device=encoder_hidden_states.device,
            )
        num_actions = num_actions or self.config.num_actions
        dim_action = dim_action or self.config.dim_action
        cfg_scale = (
            cfg_scale if cfg_scale is not None else self.config.cfg_scale
        )

        if use_base or not hasattr(self, "lora_decoder"):
            def decoder_fn(x, t, **kwargs):
                return self.decode(
                    x, t, kwargs["y"], kwargs["c"], kwargs["c_mask"],
                    adapter=None,
                )
        else:
            def decoder_fn(x, t, **kwargs):
                pos = self.decode(
                    x, t, kwargs["y"], kwargs["c"], kwargs["c_mask"],
                    adapter="positive",
                )
                neg = self.decode(
                    x, t, kwargs["y"], kwargs["c"], kwargs["c_mask"],
                    adapter="negative",
                )
                return (1.0 + cfg_scale) * pos - cfg_scale * neg

        wrapped = model_wrapper(
            decoder_fn,
            diffusion_sde.sde,
            model_type=self.config.model_type,
            guidance_type="uncond",
            model_kwargs={
                "y": proprio,
                "c": encoder_hidden_states,
                "c_mask": attention_mask,
            },
        )

        B = encoder_hidden_states.shape[0]
        x_init = torch.randn(
            B, num_actions, dim_action, device=encoder_hidden_states.device,
        ) * sample_temperature
        return diffusion_sde.generate(x_init, wrapped, steps)
