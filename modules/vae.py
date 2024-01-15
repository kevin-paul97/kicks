import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_size, hidden_dim, mu_size, logvar_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.mu_size = mu_size
        self.logvar_size = logvar_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 50),
            nn.ReLU(),
            nn.Linear(input_size // 50, hidden_dim),
            nn.ReLU()
        )
        # Separate layers for mu and logvar with their respective sizes
        self.fc_mu = nn.Linear(hidden_dim, mu_size)
        self.fc_logvar = nn.Linear(hidden_dim, logvar_size)

        # Decoder - assuming the latent space is combined mu and logvar sizes
        self.decoder = nn.Sequential(
            nn.Linear(mu_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_size // 50),
            nn.Tanh(),
            nn.Linear(input_size // 50, input_size)
        )

    def encode(self, x):
        h1 = self.encoder(x)
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
