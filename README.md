<div align="center">

<h3>Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving</h3>

[Yinan Zheng](https://zhengyinan-air.github.io/)\*, [Tianyi Tan\*](https://github.com/0ttwhy4), Bin Huang\*, Enguang Liu, [Ruiming Liang](https://github.com/LRMbbj), Jianlin Zhang, Jianwei Cui, Guang Chen, Kun Ma, Hangjun Ye, [Long Chen](https://long.ooo/), [Ya-Qin Zhang](https://scholar.google.com/citations?user=mDOMfxIAAAAJ&hl=zh-CN&oi=ao), [Xianyuan Zhan](https://zhanzxy5.github.io/zhanxianyuan/), [Jingjing Liu](https://air.tsinghua.edu.cn/en/info/1046/1194.htm)



[**[Arxiv]**](https://arxiv.org/pdf/2602.22801) [**[Project Page]**](https://zhengyinan-air.github.io/Hyper-Diffusion-Planner/)

<sub>Check out the **real-vehicle demo** on our project page</sub>

<img src="./assets/framework.jpg" width=100% style="vertical-align: bottom;">
</div>

The official implementation of **Hyper Diffusion Planner**. Our work demonstrates that diffusion models, when properly designed and trained, can serve as effective and scalable E2E AD planners for complex, real-world autonomous driving tasks.

<div style="display: flex; justify-content: center; align-items: center; gap: 1%;">

  <img src="./assets/001.gif" width="32%" alt="Video 1">

  <img src="./assets/002.gif" width="32%" alt="Video 2">

  <img src="./assets/003.gif" width="32%" alt="Video 3">

</div>

<div align="center">

Real-world urban scenario testing uses model output, with only simple smoothness post-refinement.
</div>

## To Do List

**Note**: In this repository, we will release implementation details from the paper and provide benchmark implementations for NAVSIM and NuPlan to support community research. Since our design is derived from real-world vehicle experiments, its performance on simulated benchmarks may not fully align with its real-world efficacy, which is a known limitation of such benchmarks as discussed in our paper.

- [x] Real Vehicle Demo
- [x] Nuplan Implementation
- [x] NAVSIM Implementation
- [x] initial repo & paper

## NuPlan Implementation

We provide the NuPlan implementation in [`HDP-nuplan/`](./HDP-nuplan), based on [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner/) with several modifications for HDP.

### Main Modifications

- Diffusion loss space:
  support flexible combinations of model prediction target and supervision target via `--diffusion_model_type` and `--diffusion_supervision_type` in `train_predictor.py`.
  The conversion logic is implemented in `hdp_nuplan/model/diffusion_utils/sde.py` and used in `hdp_nuplan/loss.py`.
  Current options: `x_start`, `noise`, `v`, `score`.
- Hybrid planning loss:
  we use
  `L_hybrid = L_velocity + w * L_waypoints`
  where `w` is set by `--planning_hybrid_loss` (default `0.01`).
  The detached integration used by the waypoint term is implemented in `hdp_nuplan/utils/traj_kinematics.py`.

For additional notes, see [`HDP-nuplan/README.md`](./HDP-nuplan/README.md).

## NAVSIM Implementation

We provide the NAVSIM implementation in [`HDP-navsim/`](./HDP-navsim), based on the [NAVSIM](https://github.com/autonomousvision/navsim) devkit with diffusion-based VLA agents for supervised pretraining and reward-based fine-tuning.

### Main Features

- Supervised diffusion VLA:
  `DpVlaAgent` implements the base and HDP models in the paper, using a Florence-2 encoder and a DiT trajectory decoder for supervised diffusion training on NAVSIM.
- Reward-based fine-tuning:
  `DpVlaRlAgent` implements HDP-RL by fine-tuning a pretrained `DpVlaAgent` with NAVSIM's PDM simulator and scorer.
- Pretrained checkpoint:
  we provide the supervised DP-VLA base model checkpoint, which reaches **88.6 PDMS**.
- Devkit and cache updates:
  the implementation reuses upstream NAVSIM packages where possible, replaces YAML data configs with JSON for faster startup, and stores image paths instead of raw image arrays in the data cache for simpler caching and augmentation.

For setup, cache preparation, training, RL fine-tuning, evaluation, and checkpoint links, see [`HDP-navsim/README.md`](./HDP-navsim/README.md).

## Bibtex

If you find our code and paper can help, please cite our paper as:

```
@article{
zheng2026unleash,
title={Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving},
author={Yinan Zheng and Tianyi Tan and Bin Huang and Enguang Liu and Ruiming Liang and Jianlin Zhang and Jianwei Cui and Guang Chen and Kun Ma and Hangjun Ye and Long Chen and Ya-Qin Zhang and Xianyuan Zhan and Jingjing Liu},
journal={arXiv preprint arXiv:2602.22801},
year={2026}
}
```

If you find our diffusion and flow-matching designs useful, please cite our papers as:

```
@inproceedings{
zheng2025diffusionplanner,
title={Diffusion-Based Planning for Autonomous Driving with Flexible Guidance},
author={Yinan Zheng and Ruiming Liang and Kexin ZHENG and Jinliang Zheng and Liyuan Mao and Jianxiong Li and Weihao Gu and Rui Ai and Shengbo Eben Li and Xianyuan Zhan and Jingjing Liu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}

@inproceedings{
tan2025flow,
title={Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling},
author={Tianyi Tan and Yinan Zheng and Ruiming Liang and Zexu Wang and Kexin Zheng and Jinliang Zheng and Jianxiong Li and Xianyuan Zhan and Jingjing Liu},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025}
}
```

If you find our diffusion-based RL designs useful, please cite our papers as:

```
@inproceedings{liang2026dipole,
title={Dichotomous Diffusion Policy Optimization},
author={Ruiming Liang and Yinan Zheng and Kexin Zheng and Tianyi Tan and Jianxiong Li and Liyuan Mao and Zhihao Wang and Guang Chen and Hangjun Ye and Jingjing Liu and Jinqiao Wang and Xianyuan Zhan},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026}
}

@inproceedings{
zheng2024safe,
title={Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model},
author={Yinan Zheng and Jianxiong Li and Dongjie Yu and Yujie Yang and Shengbo Eben Li and Xianyuan Zhan and Jingjing Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
```