import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, env, stack, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = np.array(env.observation_space(env.agents[0]).shape).prod() * stack
        self.latent_dim = latent_dim
        
        # Assuming the input can be reshaped suitably for a Conv1D input
        # Example: Input dimension is (Channels, Length)
        self.channels = 1  # You might need to adjust this based on your specific data
        self.length = self.input_dim // self.channels
        
        # Encoder
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, self.input_dim)

    def encode(self, x):
        x = x.view(-1, self.channels, self.length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        return torch.tanh(self.fc_mean(x)), torch.tanh(self.fc_log_var(x))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc1(z)
        z = z.view(-1, 32, self.length // 2)
        z = F.relu(self.deconv1(z))
        z = self.deconv2(z)
        return z.view(-1, self.input_dim)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var



def vae_loss_function(recon_x, x, mu, log_var, input_dim=None, beta=1.0, kld_weight=0.5):
    """
    Compute the VAE loss function.

    Parameters:
    - recon_x: Reconstructed outputs (from the decoder).
    - x: Original inputs.
    - mu: Mean from the latent space.
    - log_var: Log variance from the latent space.
    - input_dim: Total number of features in the input. If None, assumes `x` is already flattened.
    - beta: Weighting factor for the KL divergence, commonly used in beta-VAE.

    Returns:
    - Combined loss as a scalar.
    """
    # Flatten x if input_dim is provided
    if input_dim is not None:
        x = x.view(-1, input_dim)
    
    # Reconstruction loss (binary cross-entropy)
    BCE = ((x - recon_x) ** 2).sum()
    
    # KL divergence
    KL = -1 * kld_weight * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss with possible weighting on the KL term
    return BCE + beta * KL, BCE, KL