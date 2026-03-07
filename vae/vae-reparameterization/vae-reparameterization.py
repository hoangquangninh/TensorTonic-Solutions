import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    """
    n_rows, n_cols = mu.shape
    # Your implementation here
    rng = np.random.default_rng()
    
    # Generate a single random number from N(0, 1)
    epsilon = rng.standard_normal(size=(n_rows, n_cols))

    return mu + np.exp(log_var)*epsilon