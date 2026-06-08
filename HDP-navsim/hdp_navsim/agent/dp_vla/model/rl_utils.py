import os
import random
from collections import deque
from typing import List

import torch

from hdp_navsim.training.training_utils.dataset import load_feature_target_from_pickle


class ReplayBuffer:
    def __init__(self, maxsize: int = None):
        self.sample_queue = deque(maxlen=maxsize)
        self.current_epoch = 0

    def full(self):
        return len(self.sample_queue) >= self.sample_queue.maxlen

    def put(self, *items):
        self.sample_queue.append(items)

    def get(self, size):
        if len(self.sample_queue) <= 0:
            raise ValueError("Sample size exceeds queue size")

        item_list = random.choices(tuple(self.sample_queue), k=size)
        return [
            torch.cat(list(row), dim=0) if isinstance(row, torch.Tensor) else list(row)
            for row in zip(*item_list)
        ]

    def clear(self):
        self.sample_queue.clear()

    def __len__(self):
        return len(self.sample_queue)


class CacheExtractor:
    def __init__(
        self,
        data_list: List[str],
        cache_path: str,
        unique_name: str,
    ):
        self.cache_path = cache_path
        self.unique_name = unique_name
        self.data_path_dict = {p.split("/")[-1]: p for p in data_list}

    def get_feature_from_tokens(self, token_list: List[str]):
        feature_eo_list = []
        feature_pr_list = []
        for token in token_list:
            feature = load_feature_target_from_pickle(
                os.path.join(
                    self.cache_path,
                    self.data_path_dict[token],
                    f"{self.unique_name}.gz",
                )
            )
            feature_eo_list.append(feature["encoder_output"])
            feature_pr_list.append(
                feature["meta_status"][:, :3].reshape(-1).unsqueeze(0)
            )
        return feature_eo_list, feature_pr_list
