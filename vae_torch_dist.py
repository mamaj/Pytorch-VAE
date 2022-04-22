import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributions as td
from torch.distributions.kl import kl_divergence as kl
import numpy as np
from tqdm.notebook import tqdm, trange

from utils import load_mnist, display


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class VAE(nn.Module):
    def __init__(self, latent_dim=2) -> None:
        super().__init__()
        
        self.d = latent_dim
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.ReLU(),
            nn.Conv2d(10, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(320, latent_dim * 2) # mu, log_var
        )
        
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 784),
        )

        self.prior = td.Independent(
            td.Normal(loc=torch.zeros(latent_dim),
                      scale=torch.ones(latent_dim)),
            reinterpreted_batch_ndims=1
        )
        
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.prior.base_dist.loc = self.prior.base_dist.loc.to(*args, **kwargs)
        self.prior.base_dist.scale = self.prior.base_dist.scale.to(*args, **kwargs)
        
        
    def encode(self, x):
        mean_logstd = self._encoder(x)
        mu, log_std = mean_logstd[:, :self.d], mean_logstd[:, self.d:]
        return td.Independent(
            td.Normal(loc=mu, scale=torch.exp(log_std)),
            reinterpreted_batch_ndims=1
        )
    
    
    def decode(self, z):
        logits = self._decoder(z).view(-1, 1, 28, 28)
        return td.Independent(
            td.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=2
        )
        
    
    def forward(self, x, deterministic=False):
        if deterministic:
            z = self.encode(x).base_dist.loc
            x_hat = self.decode(z).base_dist.probs
        else:
            z = self.encode(x).sample()
            x_hat = self.decode(z).sample()
        return z, x_hat
    

mnist_train = load_mnist()
train_loader = DataLoader(mnist_train, shuffle=True, batch_size=64)

vae = VAE(latent_dim=2)
vae.to(DEVICE)

optim = torch.optim.Adam(lr=0.01, params=vae.parameters())
epochs = 5

train_loss = []
for epoch in trange(epochs):
    for x, _ in tqdm(train_loader):
        x = x.to(DEVICE)
        optim.zero_grad()
        
        posterior = vae.encode(x)
        z = posterior.rsample()
        
        elbo = vae.decode(z).log_prob(x) - kl(posterior, vae.prior)
        loss = -1 * elbo.mean()
        
        train_loss.append(loss.item())
        loss.backward()
        optim.step()
        
        
_, x_hat = vae(x, deterministic=True)
display(x)
display(x_hat)