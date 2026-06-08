# Hyper Diffusion Planner (NAVSIM)

A devkit for diffusion-based VLA on [NAVSIM](https://github.com/autonomousvision/navsim). This repository provides:

- `DpVlaAgent`: supervised diffusion training with a Florence-2 encoder and a
  DiT trajectory decoder.
- `DpVlaRlAgent`: reward-based fine-tuning of a pretrained `DpVlaAgent` using
  NAVSIM's PDM simulator and scorer.

Compared to the upstream NAVSIM devkit, this repository includes the following changes:

- Removed redundant code and reused upstream packages wherever possible.
- Replaced YAML data configs with JSON, which significantly speeds up startup (Hydra config compilation).
- In the data cache, store image paths instead of raw image arrays, and manage entries via JSON for faster caching, simpler dataset handling, and easier data augmentation.

## Pretrained Model

We provide a supervised pretrain checkpoint corresponding to the **base model** in the HDP paper. This checkpoint reaches **88.6 PDMS** on NAVSIM with single-shot inference: one trajectory per scene, without multi-sample selection, goal conditioning, or anchor-based decoding.

| Model | Training Data | PDMS | Description | Checkpoint |
| --- | --- | --- | --- | --- |
| DP-VLA | trainval | 88.6 | HDP base model (supervised pretrain) | [huggingface](https://huggingface.co/ZhengYinan2001/DP-VLA) |

## Table of Contents

- [Pretrained Model](#pretrained-model)
- [Requirements](#requirements)
- [Data And Paths](#data-and-paths)
- [Cache Preparation](#cache-preparation)
  - [Supervised Cache](#supervised-cache)
  - [RL Caches](#rl-caches)
- [Supervised Training](#supervised-training)
- [RL Fine-Tuning](#rl-fine-tuning)
- [Evaluation](#evaluation)

## Requirements

The codebase targets Python 3.9 and has been developed with:

- PyTorch 2.2.2, torchvision 0.17.2, and torchaudio 2.2.2
- transformers 4.49.0
- NAVSIM and nuPlan devkit checkouts
- CUDA GPUs for training and RL feature caching


```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install \
  transformers==4.49.0 accelerate==1.4.0 einops==0.8.0 peft \
  pytorch-lightning hydra-core ray timm safetensors \
  pandas matplotlib pillow tqdm pyquaternion

git clone https://github.com/autonomousvision/navsim.git
git clone https://github.com/motional/nuplan-devkit.git
pip install -e ./navsim
pip install -e ./nuplan-devkit

pip install -e .
```


## Data And Paths

The expected dataset layout under `OPENSCENE_DATA_ROOT` is:

```text
OPENSCENE_DATA_ROOT/
+-- navsim_logs/
|   `-- trainval/
+-- sensor_blobs/
|   `-- trainval/
`-- maps/
```

Copy the environment template, set the local paths, and source it:

```bash
cp env.sh env.local.sh
$EDITOR env.local.sh
source env.local.sh
```

The main variables are:

| Variable | Purpose |
| --- | --- |
| `HDP_NAVSIM_ROOT` | This repository |
| `NAVSIM_DEVKIT_ROOT` | Upstream NAVSIM checkout |
| `OPENSCENE_DATA_ROOT` | NAVSIM/OpenScene dataset root |
| `NUPLAN_MAPS_ROOT` | nuPlan map root |
| `DP_VLA_ENCODER_PATH` | Florence-2 model ID or local checkpoint |
| `NAVSIM_EXP_ROOT` | Experiment output root |
| `HDP_NAVSIM_CACHE_PATH` | Supervised feature/target cache |
| `HDP_RL_CACHE_PATH` | RL encoder-feature cache |
| `NAVSIM_METRIC_CACHE_PATH` | PDM metric cache |
| `TENSORBOARD_LOG_PATH` | TensorBoard log root |

## Cache Preparation

Training defaults to `use_cache_without_dataset=true`, so the corresponding
feature cache and JSON data list must exist before training.

The supervised cache stores image paths and ego metadata rather than embedding
raw image arrays. Images are loaded and transformed by the dataset at training
time. The RL cache additionally runs the pretrained encoder and stores its
hidden states.

### Supervised Cache

```bash
./scripts/training/run_cache_training.sh dp_vla_agent navtrain
```

This writes per-token `.gz` files under `HDP_NAVSIM_CACHE_PATH` and refreshes:

```text
hdp_navsim/training/training_utils/navtrain.json
```

For a small pipeline check, replace `navtrain` with `smoke_test`.

### RL Caches

RL training requires three aligned inputs:

1. A supervised `DpVlaAgent` checkpoint.
2. Encoder features under `HDP_RL_CACHE_PATH`.
3. PDM metric-cache entries under `NAVSIM_METRIC_CACHE_PATH`.

Build the RL feature cache:

```bash
DP_VLA_NPROC=1 ./scripts/training/run_cache_training.sh \
  dp_vla_rl_agent navtrain \
  agent.config.pretrain_config.checkpoint_path=/path/to/pretrained/checkpoint
```

The checkpoint loader accepts either a Lightning `.ckpt` file or a
Hugging Face-style export directory containing `config.json` and
`model.safetensors`.

Build the PDM metric cache for the same split:

```bash
./scripts/evaluation/run_metric_caching.sh navtrain
```

RL feature caching uses NCCL and `torchrun`; increase `DP_VLA_NPROC` to use
multiple local GPUs.

## Supervised Training

The launcher uses single-node `torchrun` by default:

```bash
DP_VLA_NPROC=1 ./scripts/training/run_training.sh \
  train_test_split=navtrain \
  dataloader.params.batch_size=4
```

Useful overrides include:

```bash
./scripts/training/run_training.sh \
  train_test_split=navtrain \
  lightning_agent.params.lr=1e-4 \
  lightning_agent.params.total_epochs=100 \
  dataloader.params.batch_size=16 \
  save_epoch=10
```
To train a base model, override the agent config with `agent=dp_vla_agent_base`; to train an HDP model, override the agent config with `agent=dp_vla_agent_hdp`.

## RL Fine-Tuning

`DpVlaRlAgent` loads the pretrained diffusion model without the Florence
encoder during training because encoder outputs are read from
`HDP_RL_CACHE_PATH`. It periodically samples trajectory groups, scores them
with PDM, stores them in a replay buffer, and optimizes a reward-weighted
diffusion objective.

Override the script's machine-specific checkpoint and data-list defaults:

```bash
DP_VLA_SPLIT=navtrain \
DP_VLA_NPROC=1 \
./scripts/training/run_training_rl.sh \
  agent.config.pretrain_config.checkpoint_path=/path/to/pretrained/checkpoint \
  agent.config.rl_config.data_list_path=${HDP_NAVSIM_ROOT}/hdp_navsim/training/training_utils/navtrain.json \
  dataloader.params.batch_size=40
```

Important RL controls live in
`hdp_navsim/config/agent/dp_vla_rl_agent.yaml`, including:

- `replay_buffer_update_epoch`
- `group_size`
- `rollout_steps`
- PDM reward weights

The metric and feature caches must contain every token listed by
`agent.config.rl_config.data_list_path`.

## Evaluation

First build the PDM metric cache for the evaluation split:

```bash
./scripts/evaluation/run_metric_caching.sh navtest
```

Then set the checkpoint and matching Hydra configuration:

```bash
export DP_VLA_RL_CKPT=/path/to/checkpoint.ckpt
export DP_VLA_RL_HPARAMS=/path/to/hparams.yaml
./scripts/evaluation/run_pdm_score.sh
```
