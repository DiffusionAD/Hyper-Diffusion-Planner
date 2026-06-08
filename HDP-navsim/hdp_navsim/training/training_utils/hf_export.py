"""HuggingFace-style export callback for PyTorch Lightning training.

The agent's :class:`pytorch_lightning.callbacks.ModelCheckpoint` handles
*resume*-friendly snapshots (full Lightning blob with optimiser state). This
callback complements it by writing a portable HuggingFace-style directory
(``config.json`` + ``model.safetensors`` and/or ``lora/`` adapter dump) every
``every_n_epochs`` epochs, plus a final ``last`` snapshot at training end.

Two flavours, picked at construction:

- ``mode="full"``   -- :meth:`DpVlaModel.save_pretrained` (pretraining).
- ``mode="lora"``   -- :meth:`DpVlaModel.save_pretrained` with
                       ``save_lora_only=True`` (RL finetune).

The callback is intentionally agnostic to the rest of the pipeline: it only
needs ``trainer.lightning_module.agent.model`` to be a :class:`DpVlaModel`
instance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class HFExportCheckpoint(pl.Callback):
    """Write a HuggingFace-style snapshot of the underlying ``DpVlaModel``."""

    VALID_MODES = ("full", "lora")

    def __init__(
        self,
        export_dirname: str = "hf_checkpoints",
        every_n_epochs: int = 10,
        save_last: bool = True,
        mode: str = "full",
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"HFExportCheckpoint mode must be one of {self.VALID_MODES}, "
                f"got {mode!r}"
            )
        self.export_dirname = export_dirname
        self.every_n_epochs = max(int(every_n_epochs), 1)
        self.save_last = save_last
        self.mode = mode

    # ------------------------------------------------------------------ utils

    def _resolve_root(self, trainer: pl.Trainer) -> Path:
        """Pick a stable export root regardless of Lightning's checkpoint
        configuration."""
        root = trainer.default_root_dir or trainer.log_dir or "."
        return Path(root) / self.export_dirname

    def _save_to(self, trainer: pl.Trainer, subdir: str) -> Optional[Path]:
        if trainer.global_rank != 0:
            return None  # only rank-0 writes to disk
        agent = getattr(trainer.lightning_module, "agent", None)
        model = getattr(agent, "model", None)
        if model is None:
            logger.warning(
                "HFExportCheckpoint: agent.model is None at save time; skipping",
            )
            return None
        out_dir = self._resolve_root(trainer) / subdir
        if self.mode == "full":
            model.save_pretrained(out_dir)
        else:
            model.save_pretrained(out_dir, save_lora_only=True)
        return out_dir

    # ----------------------------------------------------------- lifecycle

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        out = self._save_to(trainer, f"epoch_{epoch:02d}")
        if out is not None:
            logger.info("HFExportCheckpoint[%s] -> %s", self.mode, out)

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        if not self.save_last:
            return
        out = self._save_to(trainer, "last")
        if out is not None:
            logger.info("HFExportCheckpoint[%s] (last) -> %s", self.mode, out)
