import numpy as np

class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 128

        # Encoder weights
        self.W1 = np.random.randn(input_dim, self.hidden_dim) * 0.01
        self.b1 = np.zeros(self.hidden_dim)

        self.W_mu = np.random.randn(self.hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)

        self.W_logvar = np.random.randn(self.hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

        # Decoder weights
        self.W_dec = np.random.randn(latent_dim, input_dim) * 0.01
        self.b_dec = np.zeros(input_dim)

    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass through VAE.
        """
        # Encode
        h = np.tanh(x @ self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar

        # Reparameterization trick
        eps = np.random.randn(*mu.shape)
        z = mu + np.exp(0.5 * log_var) * eps

        # Decode
        x_hat = 1 / (1 + np.exp(-(z @ self.W_dec + self.b_dec)))

        return x_hat, mu, log_var

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples from prior.
        """
        z = np.random.randn(n_samples, self.latent_dim)
        x_hat = 1 / (1 + np.exp(-(z @ self.W_dec + self.b_dec)))
        return x_hat