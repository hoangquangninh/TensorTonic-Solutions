import numpy as np

def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    """
    # Your implementation here
    n_rows, n_cols = z.shape
    W = np.random.randn(n_cols, output_dim)
    b = np.zeros(output_dim)

    h = z @ W + b
    x_hat = 1 / (1 + np.exp(-h))   # sigmoid

    return x_hat