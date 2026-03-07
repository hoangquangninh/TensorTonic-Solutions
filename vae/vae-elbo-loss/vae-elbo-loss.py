import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    """
    # Your implementation here
    recon_loss = np.sum((x - x_recon) ** 2)
    kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    total_loss = recon_loss + kl_loss

    return {
        "total": float(total_loss),
        "recon": float(recon_loss),
        "kl": float(kl_loss)
    }