import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    """
    # Your implementation here
    n_rows, n_cols = mu.shape
    rng = np.random.default_rng()
    epsilon = rng.standard_normal(size=(n_rows, n_cols))

    return mu + np.exp(log_var)*epsilon