import logging
import os
import hydra
import glob
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from navsim.agents.abstract_agent import AbstractAgent

from hdp_navsim.paths import tensorboard_log_path
from hdp_navsim.training.training_utils.dataset import build_datasets, build_datasets_use_cache
from hdp_navsim.training.agent_lightning_module import AgentLightningModule

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.abspath("hdp_navsim/config/training")
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
        **cfg.lightning_agent.params,
    )

    if cfg.resume_checkpoint_path is not None:
        cfg.trainer.params.default_root_dir = cfg.resume_checkpoint_path
        checkpoint_files = glob.glob(os.path.join(cfg.resume_checkpoint_path, "**/checkpoints/last.ckpt"), recursive=True)

    logger.info(f"Path where all results are stored: {cfg.trainer.params.default_root_dir}")
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TORCH_LOGS_DIR"] = cfg.trainer.params.default_root_dir

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data, val_data = build_datasets_use_cache(cfg, agent)
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data) * cfg.trainer.params.limit_train_batches)
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data) * cfg.trainer.params.limit_val_batches)

    lightning_module.initialize(cfg, len(train_data) * cfg.trainer.params.limit_train_batches, cfg.dataloader.params.batch_size)

    # tensorboard_logger = TensorBoardLogger(cfg.trainer.params.default_root_dir)
    tensorboard_logger = TensorBoardLogger(str(tensorboard_log_path()))

    logger.info("Building Trainer")
    cfg.trainer.params.num_nodes = int(cfg.trainer.params.num_nodes)
    trainer = pl.Trainer(
        **cfg.trainer.params,
        logger=[tensorboard_logger],
        callbacks=agent.get_training_callbacks(cfg.save_epoch) + [LearningRateMonitor(logging_interval='epoch')],
    )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=checkpoint_files[0] if cfg.resume_checkpoint_path else None
    )

if __name__ == "__main__":
    main()
