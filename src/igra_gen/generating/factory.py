from typing import Callable

import torch

from igra_gen.generating.diffusion import DiffusionSampler,PosteriorDiffusionSampler
from igra_gen.generating.solver import ODESolver


def sampler_factory(mode: str, net: torch.nn.Module, conditioning_type="sr", in_shape=(48,96), target_shape=(128,256), scale=1.0) -> Callable[..., torch.Tensor]:
    """Factory to return a sampler function based on the mode.

    Args:
        mode: Sampling mode ("edm", "edm_pos_sample", "edm_cfg_sample", "cfm")
        net: Main network model
        net_cfg: CFG network model (only for edm_cfg_sample mode)
        conditioning_type: Type of conditioning ("unconditional", "sr", "igra", "sr_igra")
        in_shape: Input shape for SR operations (only used for conditional sampling)
        target_shape: Target shape for SR operations (only used for conditional sampling)
        scale: Conditioning scale (for conditional sampling)

    usage: sampler = sampler_factory("edm_pos_sample", net, conditioning_type="sr")
    """
    # TODO: currently using sampler defaults.
    if mode == "edm":
        O = DiffusionSampler(net)

        def sampler(
            X, generator: torch.Generator, condition=None, device=None, in_shape = None, *args, **kwargs
        ) -> torch.Tensor:
            if in_shape is None:
                in_shape = X.shape
            if device is None:
                device=X.device    
            return O.edm_sampler(
                latents=torch.randn(in_shape, generator=generator, device=device),
                condition=condition,
                *args,
                **kwargs
            )

        return sampler

    elif mode == "cfm":
        O = ODESolver(net)

        def cfm_sampler(X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return O.sample(x_init=X, step_size=0.1,*args,**kwargs)

        return cfm_sampler
    
    elif mode == "edm_pos_sample":
        O = PosteriorDiffusionSampler(net, conditioning_type=conditioning_type, in_shape=in_shape, target_shape=target_shape, scale=scale)

        def pos_sampler(
            X, generator, condition=None, device=None, in_shape = None, *args, **kwargs
        ) -> torch.Tensor:
            if in_shape==None:
                in_shape = X.shape
            if device==None:
                device=X.device
            return O.pos_edm_sampler(
                latents=torch.randn(in_shape, generator=generator, device=device),
                condition=condition,
                measurement=X,
                *args, **kwargs
            )

        return pos_sampler
    

    else:

        def sampler(X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return net(X)

        return sampler
