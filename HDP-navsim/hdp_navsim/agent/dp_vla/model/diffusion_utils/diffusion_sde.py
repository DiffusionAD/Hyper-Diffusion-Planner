import torch
from torch import Tensor

from hdp_navsim.agent.dp_vla.model.diffusion_utils.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP


class DiffusionSDE:
    def __init__(self, sde, time_sampler, **sample_params):
        self.sde: NoiseScheduleVP = sde
        self.time_sampler = time_sampler
        self.sample_params = sample_params

    def sample(self, x_data):
        t = self.time_sampler.sample(x_data.shape[0]).to(x_data.device)
        z = torch.randn_like(x_data, device=x_data.device)

        alpha_t, sigma_t = self.sde.marginal_alpha(t), self.sde.marginal_std(t)
        xt = alpha_t[:, None, None] * x_data + sigma_t[:, None, None] * z

        target = {
            'noise': z,
            'score': -z / (self.sde.marginal_std(t)[:, None, None] + 1e-6),
            'x_start': x_data
        }

        return xt, t, target

    def generate(self, x_init, model_fn, sample_steps, **dpm_solver_params):
        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=self.sde,
            algorithm_type='dpmsolver++',
            **dpm_solver_params
        )

        return dpm_solver.sample(
            x_init,
            steps=sample_steps,
            order=self.sample_params['sample_order'],
            skip_type=self.sample_params['sample_skip_type'],
            method=self.sample_params['sample_method'],
            denoise_to_zero=self.sample_params['denoise_to_zero']
        )


class TimeSampler:
    """Sample diffusion timesteps; only ``uniform`` is used in training configs."""

    def __init__(self, sample_method='uniform', **sample_params):
        self.sample_method = sample_method
        self.eps = sample_params['eps']
        self.sample_params = sample_params
        if sample_method != 'uniform':
            raise ValueError(
                f"Unsupported time sampler {sample_method!r}; "
                "only 'uniform' is supported after cleanup"
            )

    def sample(self, shape) -> Tensor:
        # T=0 is data, T=1 is noise (diffusion convention used here).
        return torch.rand(shape) * (1 - self.eps) + self.eps
