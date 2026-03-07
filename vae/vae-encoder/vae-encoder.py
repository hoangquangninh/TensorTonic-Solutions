import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters.
    """
    # Your implementation here
    batch_size, input_dim = x.shape
    hidden_dim = 128

    W1 = np.random.uniform(size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)

    W_mu = np.random.uniform(size=(hidden_dim, latent_dim))
    b_mu = np.zeros(latent_dim)

    W_logvar = np.random.uniform(size=(hidden_dim, latent_dim))
    b_logvar = np.zeros(latent_dim)

    # Hidden layer
    h = x @ W1 + b1
    h = np.tanh(h)

    # Output layers
    mu = h @ W_mu + b_mu
    logvar = h @ W_logvar + b_logvar

    return mu, logvar