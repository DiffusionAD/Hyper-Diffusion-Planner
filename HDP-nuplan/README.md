# Hyper Diffusion Planner (NuPlan)

In this repo, we provide an implementation of our Hyper Diffusion Planner on NuPlan benchmark, based on [*Diffusion Planner*](https://github.com/ZhengYinan-AIR/Diffusion-Planner). One can follow the *Diffusion Planner* for data processing, model training and evaluation.

## Main Modification

### Diffusion Loss Space

**(See more details in [Section 4.1] of the paper)**

We add choices of in diffusion loss space. Specifically, we add diffusion sde transformation in `hdp_nuplan/model/diffusion_utils/sde.py` and choices of supervision in `hdp_nuplan/loss.py`. One can train HDP models with different combination of model prediction and loss function by modifying the `diffusion_model_type` and `diffusion_supervision_type` arguments in `train_predictor.py`. 
```
# HDP-nuplan/torch_run.sh
# The default configuration is x_start model prediction with x_start supervision
sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
--diffusion_model_type "x_start" \
--diffusion_supervision_type "x_start" \
--batch_size 2048

# (e.g.) to use noise model prediction with v supervision
sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
--diffusion_model_type "noise" \
--diffusion_supervision_type "v" \
--batch_size 2048
```
We currently support `x_start`($\tau_0$), `noise`($\epsilon$) and `velocity`($v_t$). The transformation can be found in Table III of the paper.

### Hybrid Loss

**(See more details in [Section 4.2] of the paper)**

We use hybrid loss with velocity prediction in `hdp_nuplan/loss.py`: $$\mathcal{L}_{hybrid} = \mathcal{L}_{velocity} + \omega \cdot \mathcal{L}_{waypoints}$$
where the hybrid loss weight $\omega$ is passed by `planning_hybrid_loss` argument in `train_predictor.py`. The detach integration can be found in `hdp_nuplan/utils/traj_kinematics.py`.
```
def detached_integral(u, detach_window_size):
    # u: (B, T=80, D)
    cum_detach = torch.cumsum(u.detach(), dim=-2)
    cum_normal = torch.cumsum(u, dim=-2)

    # number of gradient from previous timesteps contained in: 
    # shifted: [0, 1, 2, ..., window_size-1, window_size, ...., T] ->
    # shifted: [T-window_size+1, T-window_size+2, ...,T, 0, 1, 2, ...., T - window_size] ->
    # sum_recent: [0, 1, 2, ..., window_size-1, window_size, ...., window_size]
    shifted = torch.roll(cum_normal, shifts=detach_window_size, dims=-2)
    shifted[:, :, :detach_window_size] = 0
    sum_recent = cum_normal - shifted
        
    cum_detach_shifted = torch.roll(cum_detach, shifts=detach_window_size, dims=-2)
    cum_detach_shifted[:, :, :detach_window_size] = 0
        
    cumulative_sum = cum_detach_shifted + sum_recent
    return cumulative_sum
```
We also provide a default normalization compatible with the numerical scale of velocity, specified in `normalizaition.json`.

## Getting Started

- Setup conda environment
```
conda create -n hdp_nuplan python=3.9
conda activate hdp_nuplan

# setup hyper_diffusion_planner
# pwd: */Hyper-Diffusion-Planner/
cd ./HDP-nuplan/
pip install -e .
pip install -r requirements_torch.txt
```
- Setup the nuPlan dependency, prepare the training data, and launch training and evaluation following the guidance in [*Diffusion Planner*](https://github.com/ZhengYinan-AIR/Diffusion-Planner). 