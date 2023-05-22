import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):

    def __init__(self, encoding_layers, penalty=0, norm_type="l2"):
        super().__init__()
        self.penalty = penalty
        self.norm_type = norm_type
        latent_dim = encoding_layers[-1]

        # Encoding layers
        encoding = []
        for i in range(len(encoding_layers) - 1):
            encoding.append(nn.Linear(encoding_layers[i], encoding_layers[i + 1]))
            encoding.append(nn.ReLU(inplace=True))
            encoding.append(nn.BatchNorm1d(encoding_layers[i + 1]))

        self.encoder = nn.Sequential(*encoding)
        self.fc_mu = nn.Linear(encoding_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoding_layers[-1], latent_dim)

        # Decoding layers
        decoding_layers = encoding_layers[::-1]
        decoding = []
        for i in range(len(decoding_layers) - 1):
            decoding.append(nn.Linear(decoding_layers[i], decoding_layers[i + 1]))
            decoding.append(nn.ReLU(inplace=True))
            decoding.append(nn.BatchNorm1d(decoding_layers[i + 1]))

        self.decoder = nn.Sequential(*decoding)
        self.fc_output = nn.Linear(decoding_layers[-1], encoding_layers[0])

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z):
        h = self.decoder(z)
        x_reconstructed = self.fc_output(h)
        return x_reconstructed

    def forward(self, x):
        mu, log_var = self.encode(x)
        x_encoded = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(x_encoded)
        return x_encoded, x_reconstructed

    def loss_function(self, x_encoded, x_reconstructed, original, mu, log_var):
        reconstruction_loss = F.mse_loss(x_reconstructed, original, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Compute penalty term
        dot_product = torch.matmul(x_encoded.T, x_encoded)
        if self.norm_type == 'frobenius':
            penalty_loss = torch.norm(dot_product - torch.eye(dot_product.shape[0]), p='fro') ** 2
        elif self.norm_type == 'l2':
            penalty_loss = torch.norm(dot_product - torch.eye(dot_product.shape[0]), p=2) ** 2
        else:
            raise ValueError("Invalid norm type: {}, use either frobenius or l2 norms".format(self.norm_type))
        return reconstruction_loss + kl_divergence + self.penalty * penalty_loss
