from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import uuid
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

import os

from hdp_navsim.training.training_utils.dataloader import SceneLoader
from hdp_navsim.training.training_utils.dataclasses import SensorConfig
from navsim.agents.abstract_agent import AbstractAgent

from hdp_navsim.training.training_utils.dataset import (
    Dataset,
    SceneFilter,
    resolve_cache_data_list_path,
    write_cache_data_list,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.abspath("hdp_navsim/config/training")
CONFIG_NAME = "default_training"


def get_cache_data_list_path(cfg: DictConfig) -> Path:
    """Returns the JSON path used by cache-only training after cache generation."""
    configured_path = OmegaConf.select(cfg, "cache_data_list_path")
    if configured_path:
        return Path(configured_path)
    return Path(resolve_cache_data_list_path(cfg))


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    Helper function to cache features and targets of learnable agent.
    :param args: arguments for caching
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    agent: AbstractAgent = instantiate(cfg.agent)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    logger.info(f"Extracted {len(scene_loader.tokens)} scenarios for thread_id={thread_id}, node_id={node_id}.")

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=True,
    )
    return []


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for dataset caching script.
    :param cfg: omegaconf dictionary
    """
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info("Building Worker")
    worker: WorkerPool = instantiate(cfg.worker)
    # import ipdb
    # ipdb.set_trace()
    logger.info("Building SceneLoader")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    logger.info(f"Extracted {len(scene_loader)} scenarios for training/validation dataset")

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    _ = worker_map(worker, cache_features, data_points)
    cache_data_list_path = get_cache_data_list_path(cfg)
    write_cache_data_list(scene_loader, cache_data_list_path)
    logger.info(f"Finished caching {len(scene_loader)} scenarios for training/validation dataset")


if __name__ == "__main__":
    main()