from typing import Dict, Tuple, Optional
from omegaconf import DictConfig
import math
from torch import Tensor
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ConstantLR, LambdaLR
import pytorch_lightning as pl
from timm.utils import ModelEma
from matplotlib import pyplot as plt

from navsim.agents.abstract_agent import AbstractAgent

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(
        self, 
        agent: AbstractAgent, 
        lr: float,
        total_epochs: int, 
        warmup_epochs: int = 5, 
        use_ema: bool = True,
    ):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        :param lr: learning rate
        :param total_epochs: number of epochs for training
        :param warmup_epochs: number of warm up epochs
        :param use_ema: ema training
        """
        super().__init__()
        
        self.agent = agent
        self.lr = lr

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        self.use_ema = use_ema


    def initialize(self, cfg: DictConfig, dataset_len: int, batch_size: int) -> None:
        """Inherited, see superclass."""

        self.cfg = cfg
        
        self.batch_size = batch_size
        self.dataset_len = dataset_len

        self.save_hyperparameters(cfg)

    def setup(self, stage=None):
        """Setup method to compute steps after Trainer is initialized."""

        total_batches = self.dataset_len // (self.batch_size * self.trainer.world_size)
        if self.dataset_len % (self.batch_size * self.trainer.world_size) != 0:
            total_batches += 1
        
        self.total_steps = self.total_epochs * total_batches
        self.warmup_steps = self.warmup_epochs * total_batches

        self.agent.initialize_training(self.cfg.lightning_agent.params.lr)

        # if self.use_ema and self.global_rank == 0:
        #     self.agent_ema = ModelEma(
        #         self.agent.model,
        #         decay=0.999,
        #         device='cuda' if torch.cuda.is_available() else 'cpu'
        #     )

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """

        features, targets, token_list = batch

        batch_size = features['history'].shape[0]

        loss = self.agent.compute_loss(features, targets, token_list)

        for key, value in loss.items():
            if key == 'loss':
                self.log("train-loss/total", loss["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=False, batch_size=batch_size)
            else:
                self.log(f"train-{key}", value, on_step=True, on_epoch=True, prog_bar=False, sync_dist=False, batch_size=batch_size)

        if 'loss' not in loss.keys():
            return None
        else:
            return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):

        if self.use_ema and self.global_rank == 0:
            self.agent_ema.update(self.agent.model)

    def on_train_epoch_start(self):
        self.agent.on_train_epoch_start(self.current_epoch)

    def on_train_epoch_end(self):
        self.agent.on_train_epoch_end(self.current_epoch)

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """

        # if hasattr(self.agent, "validation_step"):
        #     fig = self.agent.validation_step(batch, batch_idx)

        #     self.logger.experiment.add_figure(
        #         tag="val/visualization",
        #         figure=fig,
        #         global_step=self.global_step
        #     )

        #     plt.close(fig)
        #     return None
        # else:
        #     features, targets = batch

        #     batch_size = features['image_obs'].shape[0]

        #     loss = self.agent.compute_loss(features, targets)

        #     for key, value in loss.items():
        #         if key == 'loss':
        #             self.log("val/loss", loss["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=False, batch_size=batch_size)
        #         else:
        #             self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False, batch_size=batch_size)

        #     return loss

        batch_size = batch[0]["encoder_output"].shape[0]
        results = self.agent.validation_step(batch, batch_idx)

        for key, value in results.items():
            if value.dim() == 0:  # 标量，记录为 plot（scalar）
                self.log(f"val-{key}", value.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
            elif value.dim() == 1:  # 一维向量，记录为直方图
                self.logger.experiment.add_histogram(
                    f"val-{key}", value.detach(), global_step=self.current_epoch
                )
            else:
                # 如果是更高维的 tensor，可以选择忽略或展平后记录直方图
                self.logger.experiment.add_histogram(
                    f"val-{key}", value.detach().flatten(), global_step=self.current_epoch
                )

        return results

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema and self.global_rank == 0:
            checkpoint["ema_state_dict"] = self.agent_ema.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):

        if "ema_state_dict" in checkpoint and self.use_ema and self.global_rank == 0:
            self.agent_ema.ema.load_state_dict(checkpoint['ema_state_dict'])


    def configure_optimizers(self):
        optimizer = self.agent.get_optimizers()

        total_steps = self.total_steps

        # def lr_lambda(current_step):
        #     if current_step < self.warmup_steps:
        #         return 0.1 + 0.9 * (current_step + 1) / self.warmup_steps
        #     cosine_decay = 0.1 + 0.45 * (
        #         1 + math.cos(math.pi * (current_step - self.warmup_steps) / (total_steps - self.warmup_steps))
        #     )
        #     return cosine_decay

        # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = self.agent.get_scheduler(optimizer, total_steps)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
