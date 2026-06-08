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
import pickle

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from hdp_navsim.training.training_utils.dataloader import SceneLoader
from hdp_navsim.training.training_utils.dataclasses import SensorConfig
from navsim.agents.abstract_agent import AbstractAgent

from hdp_navsim.training.training_utils.dataset import (
    Dataset,
    SceneFilter,
    resolve_cache_data_list_path,
    write_cache_data_list,
)


import os
import torch.distributed as dist
import torch


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
    if not args:
        logger.info("No scenarios assigned to this rank; skipping cache computation.")
        return []

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

def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    """
    Helper function to broadcast an object from the source rank to all other processes.
    :param obj: Object to broadcast.
    :param device: Device to use for tensor operations.
    :param src: Source rank.
    :return: Broadcasted object.
    """
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor)).to(device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0).to(device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(size_tensor.item()).to(device)
        dist.broadcast(tensor, src=src)
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    return obj

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for dataset caching script.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')  

    logger.info("Global Seed set to 0")
    pl.seed_everything(0)

    logger.info("Building Worker")
    worker: WorkerPool = instantiate(cfg.worker)

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
    if rank == 0:
        tokens_to_evaluate = list(set(scene_loader.tokens))
        tokens_to_evaluate = sorted(tokens_to_evaluate) 
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)

    sampler = InferenceSampler(len(tokens_to_evaluate))

    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token]  
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token], 
        })


    _ = cache_features(data_points)
    dist.barrier()
    if rank == 0:
        cache_data_list_path = get_cache_data_list_path(cfg)
        write_cache_data_list(scene_loader, cache_data_list_path)
    dist.barrier()
    logger.info(f"Finished caching {len(scene_loader)} scenarios for training/validation dataset")


if __name__ == "__main__":
    if os.environ.get("HDP_CACHE_DEBUG", "0") == "1":
        import debugpy

        debug_port = int(os.environ.get("HDP_CACHE_DEBUG_PORT", "5678"))
        debugpy.listen(debug_port)
        print(f"Waiting for debugger on port {debug_port}")
        debugpy.wait_for_client()
        print("Debugger attached")

    main()
