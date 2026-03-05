import deepinv as dinv


def get_physics(loss_name,
                sigma=None,
                gamma=None,
                device=None,
                **kwargs):
    """
    Factory function to create deepinv physics objects
    consistent with the selected loss.

    Parameters
    ----------
    loss_name : str
        Name of the loss being used.
    sigma : float, optional
        Gaussian noise std.
    gamma : float, optional
        Poisson gain parameter.
    device : torch.device, optional
        Device where physics should live.
    kwargs : dict
        Extra arguments forwarded to noise model constructors.

    Returns
    -------
    physics : dinv.physics.Physics
    """

    # ---------------------------
    # Gaussian noise
    # ---------------------------
    if loss_name in ["sure", "unsure", "r2r_g"]:

        if sigma is None:
            raise ValueError("sigma must be provided for Gaussian denoising.")

        noise_model = dinv.physics.GaussianNoise(
            sigma=sigma,
            **kwargs
        )

    # ---------------------------
    # Poisson noise
    # ---------------------------
    elif loss_name in ["pure", "r2r_p"]:

        if gamma is None:
            raise ValueError("gamma must be provided for Poisson denoising.")

        noise_model = dinv.physics.PoissonNoise(
            gain=gamma,
            **kwargs
        )

    # ---------------------------
    # Poisson-Gaussian noise
    # ---------------------------
    elif loss_name in ["pgure", "unpgure"]:

        if sigma is None or gamma is None:
            raise ValueError("sigma and gamma must be provided for PG denoising.")

        noise_model = dinv.physics.PoissonGaussianNoise(
            sigma=sigma,
            gain=gamma,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown loss (cannot infer physics): {loss_name}")

    physics = dinv.physics.Denoising(
        noise_model=noise_model
    )

    if device is not None:
        physics = physics.to(device)

    return physics