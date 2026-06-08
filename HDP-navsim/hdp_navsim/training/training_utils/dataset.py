import os
import json
from pathlib import Path
import pickle
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, Union
from dataclasses import dataclass, asdict
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from hdp_navsim.paths import data_list_path as repo_data_list_path

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from hdp_navsim.training.training_utils.dataloader import SceneLoader
from hdp_navsim.training.training_utils.dataclasses import SceneFilter

import logging
logger = logging.getLogger(__name__)

IMAGE_SIZE = 384

IMAGE_PREPROCESS_TEST = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)])

IMAGE_PREPROCESS_TRAIN = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC), 
                                             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)])

def get_image(meta_images, training=False):

    image_num = len(meta_images)

    pil_images = []

    for i in range(image_num):
        if type(meta_images[i]) is np.ndarray:
            pil_images.append(Image.fromarray(meta_images[i]))
        else:
            pil_images.append(Image.open(meta_images[i]))

    image_obs =  torch.stack([
                    torch.stack([IMAGE_PREPROCESS_TEST(img)]) for img in pil_images
                ])
    return image_obs


def get_language_promt(meta_status):

    current_state = meta_status[-1]
    prev_state = meta_status[-2]

    dt = 0.5

    cur_velocity = current_state[-4]
    angle_diff = current_state[2] - prev_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = angle_diff / dt

    if abs(cur_velocity) < 0.2:
        steering_angle = 0.0
        yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable
    else:
        steering_angle = np.arctan(
            yaw_rate * get_pacifica_parameters().wheel_base / abs(cur_velocity)
        )

    directions = ['left', 'straight', 'right', 'unknown']
    
    driving_onehot = current_state[3:7]
    velocity = current_state[7:9]
    acc = current_state[9:]
    
    direction_idx = int(torch.argmax(driving_onehot).item())
    direction_text = directions[direction_idx]

    vx, vy = velocity[0].item(), velocity[1].item()
    speed = (vx**2 + vy**2) ** 0.5

    accx, accy = acc[0].item(), acc[1].item()
    acceleration = (accx**2 + accy**2) ** 0.5

    prompt = (
        f"- Steering: {steering_angle:.2f} degrees\n"
        f"- Speed: {speed:.2f} m/s\n"
        f"- Acceleration: {acceleration:.2f} m/s^2\n"
        f"- Navigation: {direction_text}\n"
        f"\nWhat trajectory comes next?"
    )
    return prompt


def random_shifts_aug(x: torch.Tensor, pad: int = 30):

    x = x.float()
    c, h, w = x.size()
    assert h == w
    padding = tuple([pad] * 4)
    x = F.pad(x, padding, "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)

    shift = torch.randint(0, 2 * pad + 1, size=(1, 1, 2), device=x.device, dtype=x.dtype)
    shift *= 2.0 / (h + 2 * pad)

    grid = base_grid + shift
    return F.grid_sample(x.unsqueeze(0), grid.unsqueeze(0), padding_mode="zeros", align_corners=False).squeeze(0)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


def write_cache_data_list(scene_loader: SceneLoader, output_path: Union[str, Path]) -> List[str]:
    """
    Writes a cache-only dataset list in the same format as training_utils/navtest.json.

    Each entry is relative to the configured cache root and has the form
    ``<log_name>/<token>``.
    """
    cache_entries: List[str] = []
    for log_name, tokens in scene_loader.get_tokens_list_per_log().items():
        cache_entries.extend([f"{log_name}/{token}" for token in tokens])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(cache_entries, file, indent=4)
        file.write("\n")
    os.replace(tmp_path, output_path)

    logger.info("Wrote %d cache entries to %s", len(cache_entries), output_path)
    return cache_entries


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        data_list: str,
        cache_path: str,
        feature_unique_name: List[AbstractFeatureBuilder],
        target_unique_name: List[AbstractTargetBuilder],
        training: bool = True,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()

        self._data_list = data_list
        self._cache_path = cache_path

        self._feature_unique_name = feature_unique_name
        self._target_unique_name = target_unique_name

        self._training = training

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(os.path.join(self._cache_path, self._data_list[idx]))

    def _load_scene_with_token(self, token_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """
        scene_token_path = token_path.split("/")[-1]

        features: Dict[str, torch.Tensor] = {}
        # for builder in self._feature_builders:
        data_dict_path = os.path.join(token_path, f"{self._feature_unique_name}.gz")
        data_dict = load_feature_target_from_pickle(data_dict_path)
        if 'encoder_output' in data_dict.keys():
            features['encoder_output'] = data_dict['encoder_output'].squeeze(0)
        else:
            data_dict['image_obs'] = get_image(data_dict['meta_images'], self._training)
            data_dict['language'] = get_language_promt(data_dict['meta_status'])
            features['image_obs'] = data_dict['image_obs']
            features['language'] = data_dict['language']
        features['history'] = data_dict['meta_status'][:, :3].reshape(-1)

        targets: Dict[str, torch.Tensor] = {}
        # for builder in self._target_builders:
        data_dict_path = os.path.join(token_path, f"{self._target_unique_name}.gz")
        data_dict = load_feature_target_from_pickle(data_dict_path)
        targets['ego_future_trajectory'] = data_dict['ego_future_trajectory'].to(torch.float32)

        return (features, targets, scene_token_path)



class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features_training(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets_training(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path


    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def load_token_cache(self, token_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """
        scene_token_path = token_path.split("/")[-1]

        features: Dict[str, torch.Tensor] = {}
        # for builder in self._feature_builders:
        data_dict_path = os.path.join(token_path, f"{self._feature_builders[0].get_unique_name()}.gz")
        data_dict = load_feature_target_from_pickle(data_dict_path)
        if 'encoder_output' in data_dict.keys():
            features['encoder_output'] = data_dict['encoder_output'].squeeze(0)
        else:
            data_dict['image_obs'] = get_image(data_dict['meta_images'], self._training)
            data_dict['language'] = get_language_promt(data_dict['meta_status'])
            features['image_obs'] = data_dict['image_obs']
            features['language'] = data_dict['language']
        features['history'] = data_dict['meta_status'][:, :3].reshape(-1)

        targets: Dict[str, torch.Tensor] = {}
        # for builder in self._target_builders:
        data_dict_path = os.path.join(token_path, f"{self._target_builders[0].get_unique_name()}.gz")
        data_dict = load_feature_target_from_pickle(data_dict_path)
        targets['ego_future_trajectory'] = data_dict['ego_future_trajectory'].to(torch.float32)

        return (features, targets, scene_token_path)


def build_datasets(cfg: DictConfig, agent: AbstractAgent):
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    with open(cfg.train_val_test_log_split_path, "r", encoding="utf-8") as file:
        train_val_test_log_split = json.load(file)

    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:

        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in set(train_val_test_log_split['train_logs'])
        ]
    else:
        train_scene_filter.log_names = train_val_test_log_split['train_logs']

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in set(train_val_test_log_split['val_logs'])]
    else:
        val_scene_filter.log_names = train_val_test_log_split['val_logs']

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


def resolve_cache_data_list_path(cfg: DictConfig) -> str:
    """Resolve the token-list JSON for cache-only training.

    Priority:
    1. ``agent.config.rl_config.data_list_path`` (RL agent / shell overrides)
    2. ``paths.data_list_path`` (promoted via agent defaults)
    3. Hydra ``train_test_split`` choice -> ``training_utils/{split}.json``
       (same rule as :func:`run_cache_training.get_cache_data_list_path`)
    """
    rl_path = OmegaConf.select(cfg, "agent.config.rl_config.data_list_path")
    if rl_path:
        return rl_path

    paths_path = OmegaConf.select(cfg, "paths.data_list_path")
    if paths_path:
        return paths_path

    log_names_tokens_path = OmegaConf.select(
        cfg, "train_test_split.scene_filter.log_names_tokens_path"
    )
    if log_names_tokens_path:
        split_name = Path(log_names_tokens_path).stem
    else:
        try:
            split_name = HydraConfig.get().runtime.choices.get("train_test_split")
        except Exception:
            split_name = None
        split_name = split_name or OmegaConf.select(cfg, "split") or "trainval"

    return str(repo_data_list_path(split_name))


def build_datasets_use_cache(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets using cached data.

    Train / val split JSONs are taken from ``cfg.train_data_list_path`` /
    ``cfg.val_data_list_path``. When unset, both fall back to
    :func:`resolve_cache_data_list_path` (honours ``train_test_split=...`` from
    the launch script).

    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    default_list = resolve_cache_data_list_path(cfg)
    train_path = OmegaConf.select(cfg, "train_data_list_path") or default_list
    val_path = OmegaConf.select(cfg, "val_data_list_path") or default_list

    with open(train_path, "r", encoding="utf-8") as file:
        train_data_list = json.load(file)
    with open(val_path, "r", encoding="utf-8") as file:
        val_data_list = json.load(file)

    train_data = CacheOnlyDataset(
        train_data_list,
        cache_path=cfg.cache_path,
        feature_unique_name=agent.get_feature_name(),
        target_unique_name=agent.get_target_name(),
    )
    val_data = CacheOnlyDataset(
        val_data_list,
        cache_path=cfg.cache_path,
        feature_unique_name=agent.get_feature_name(),
        target_unique_name=agent.get_target_name(),
        training=False
    )

    return train_data, val_data
