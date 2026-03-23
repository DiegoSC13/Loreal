import torch
import deepinv
from deepinv.loss.sure import SureGaussianLoss, SurePoissonLoss, SurePGLoss
from deepinv.loss.r2r import R2RLoss


def get_loss(loss_name,
             device,
             sigma=None,
             gamma=None,
             alpha=0.15,
             step_size = (1e-4, 1e-4),  #UNSURE y PGUSRE
             momentum  = (0.9, 0.9),   #UNSURE y PGURE
             #mc_iter=1,    
             **kwargs):
    """
    Factory function to create deepinv loss objects.

    Parameters
    ----------
    loss_name : str
        Name of the loss to use.
    device : torch.device
        Device where the loss should live.
    sigma : float, optional
        Gaussian noise std (for SURE Gaussian).
    alpha : float, optional
        Poisson scaling parameter.
    mc_iter : int
        Monte Carlo iterations (for SURE-based losses).
    kwargs : dict
        Extra arguments forwarded to deepinv loss constructors.

    Returns
    -------
    loss_fn : nn.Module
    """

    if loss_name == "sure":
        if sigma is None:
            raise ValueError("sigma must be provided for sure_gaussian")

        loss_fn = SureGaussianLoss(
            sigma=sigma,
            #mc_iter=mc_iter,
            **kwargs
        )

    elif loss_name == "pure":
        if gamma is None:
            raise ValueError("alpha must be provided for sure_poisson")

        loss_fn = SurePoissonLoss(
            gain=gamma,
            #mc_iter=mc_iter,
            **kwargs
        )

    elif loss_name == "pgure":
        if sigma is None or gamma is None:
            raise ValueError("alpha must be provided for sure_poisson")

        loss_fn = SurePGLoss(
            sigma=sigma,
            gain=gamma,
            #mc_iter=mc_iter,
            **kwargs
        )

    elif loss_name == "unsure":
        loss_fn = SureGaussianLoss(
            sigma=sigma,
            #mc_iter=mc_iter,
            unsure=True,
            step_size = step_size,
            momentum  = momentum,
            **kwargs
        )

    elif loss_name == "unpgure":

        loss_fn = SurePGLoss(
            sigma=sigma,
            gain=gamma,
            #mc_iter=mc_iter,
            unsure=True,
            step_size = step_size,
            momentum  = momentum,
            **kwargs
        )

    elif loss_name == "r2r_g":
        if sigma is None:
            raise ValueError("sigma must be provided for r2r_gaussian")

        loss_fn = R2RLoss(
            noise_model=deepinv.physics.GaussianNoise(sigma=sigma),
            alpha=alpha,
            **kwargs
        )

    elif loss_name == "r2r_p":
        if gamma is None:
            raise ValueError("alpha must be provided for r2r_poisson")

        loss_fn = R2RLoss(
            noise_model=deepinv.physics.PoissonNoise(gain=gamma),
            alpha=alpha,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    # Mover la loss al mismo device que el modelo
    loss_fn = loss_fn.to(device)

    return loss_fn