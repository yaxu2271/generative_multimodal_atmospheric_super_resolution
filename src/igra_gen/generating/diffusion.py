import os
import numpy as np
import torch
from igra_gen.generating.conditioning_methods import *

class DiffusionSampler:
    def __init__(self, net):
        super().__init__()
        self.net = net

    @torch.no_grad()
    def edm_sampler(
        self,
        latents,
        condition=None,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ):
        """EDM sampler implementation."""
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=torch.float32, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float32) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self.net(x_hat, t_hat, condition).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.net(x_next, t_next, condition).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


class PosteriorDiffusionSampler:
    def __init__(self, net, conditioning_type="sr", in_shape=(48,96), target_shape=(128,256), scale=1.0):
        super().__init__()
        self.net = net
        from igra_gen.generating.conditioning_methods import UnifiedOperator
        self.operator = UnifiedOperator(
            conditioning_type=conditioning_type,
            in_shape=in_shape,
            target_shape=target_shape
        )
        self.cm = ConditioningMethod(self.operator, scale=scale)
        
    def pos_edm_sampler(
        self,
        latents,
        measurement,
        condition=None,
        randn_like=torch.randn_like,
        num_steps=60,
        sigma_min=0.001,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        **conditioning_kwargs,
        
    ):
        """EDM sampler implementation."""
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)
        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=torch.float32, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float32) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next.requires_grad_() 
            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = (x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)).float()
            # Euler step.
            with torch.autocast(
                                latents.device.type, enabled=True, dtype=torch.bfloat16
            ):
                denoised = self.net(x_hat, t_hat, condition)#.float()
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            exp_x_0_x_t = denoised#/t_hat # 1/s^2(x_t + s^2 * sigma^2 * score(x_t))
            likelihood_score, err = self.cm.conditioning(x_prev = x_cur,x_t = x_next,x_0_hat=exp_x_0_x_t,
                                                measurement=measurement,sigma=t_hat, **conditioning_kwargs)
            if os.environ.get("DPS_TRACE", "0") not in ("0", "", "false", "False"):
                print(f"[DPS] step={i:03d} sigma={float(t_hat):.4f} err={err}")
            # Apply 2nd order correction.
            if i < num_steps - 1:
                with torch.no_grad():
                    with torch.autocast(latents.device.type, enabled=True, dtype=torch.bfloat16
                        ):
                        denoised = self.net(x_next.float(), t_next.float(), condition)#.float()
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime - likelihood_score)
        return x_next
    
