"""Configuration class for :class:`DpVlaModel` (transformers-style).

Acts as the **single source of truth** for backbone hyper-parameters; the
matching ``model:`` block in the agent yaml only overrides these defaults, and
no other module duplicates them.
"""

from __future__ import annotations

from transformers import PretrainedConfig


DEFAULT_MODEL_CONFIG: dict = {
    "encoder_name": "microsoft/Florence-2-large",
    "with_encoder": True,
    "hidden_size": 1024,
    "depth": 12,
    "num_heads": 16,
    "num_actions": 8,
    "dim_action": 4,
    "dim_y": 12,
    "mlp_ratio": 4.0,
    "cfg_scale": 1.0,
    "model_type": "noise",
    "kinematic_type": "waypoint",
    "lora_r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
}


class DpVlaConfig(PretrainedConfig):
    """Hyper-parameters of the Dp-VLA backbone."""

    model_type = "dp_vla"

    def __init__(
        self,
        encoder_name: str = DEFAULT_MODEL_CONFIG["encoder_name"],
        with_encoder: bool = DEFAULT_MODEL_CONFIG["with_encoder"],
        hidden_size: int = DEFAULT_MODEL_CONFIG["hidden_size"],
        depth: int = DEFAULT_MODEL_CONFIG["depth"],
        num_heads: int = DEFAULT_MODEL_CONFIG["num_heads"],
        num_actions: int = DEFAULT_MODEL_CONFIG["num_actions"],
        dim_action: int = DEFAULT_MODEL_CONFIG["dim_action"],
        dim_y: int = DEFAULT_MODEL_CONFIG["dim_y"],
        mlp_ratio: float = DEFAULT_MODEL_CONFIG["mlp_ratio"],
        cfg_scale: float = DEFAULT_MODEL_CONFIG["cfg_scale"],
        model_type: str = DEFAULT_MODEL_CONFIG["model_type"],
        kinematic_type: str = DEFAULT_MODEL_CONFIG["kinematic_type"],
        lora_r: int = DEFAULT_MODEL_CONFIG["lora_r"],
        lora_alpha: int = DEFAULT_MODEL_CONFIG["lora_alpha"],
        lora_dropout: float = DEFAULT_MODEL_CONFIG["lora_dropout"],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if model_type not in {"noise", "score", "x_start"}:
            raise ValueError(
                "DpVlaConfig.model_type must be one of "
                "{\"noise\", \"score\", \"x_start\"}, got "
                f"{model_type!r}"
            )
        if kinematic_type not in {"diff", "waypoint"}:
            raise ValueError(
                "DpVlaConfig.kinematic_type must be one of "
                "{\"diff\", \"waypoint\"}, got "
                f"{kinematic_type!r}"
            )
        self.encoder_name = encoder_name
        self.with_encoder = with_encoder
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_actions = num_actions
        self.dim_action = dim_action
        self.dim_y = dim_y
        self.mlp_ratio = mlp_ratio
        self.cfg_scale = cfg_scale
        self.model_type = model_type
        self.kinematic_type = kinematic_type
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
