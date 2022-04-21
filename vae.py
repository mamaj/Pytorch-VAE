from calendar import c
from pathlib import Path
import sys
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.distributions as td
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange

DATAPATH = Path.home() / '.datasets'


class VAE(nn.Module):
    def __init__(self, latent_dim=2) -> None:
        super().__init__()
        
        self.d = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.ReLU(),
            nn.Conv2d(10, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(320, latent_dim * 2) # mu, log_var
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 784),
        )
    
    
    def encode(self, x):
        mean_logstd = self.encoder(x)
        mu, log_std = mean_logstd[:, :self.d], mean_logstd[:, self.d:]
        return mu, log_std
    
    
    def decode(self, z):
        x_hat = self.decoder(z)
        return F.sigmoid(x_hat).view(-1, 28, 28)
        
    
    def forward(self, x):
        mu, log_std = self.encode(x)
        
        if self.training:
            z = self.sample_latent(mu, log_std)
        else:
            z = mu
        
        x_hat = self.decode(z)
        
        return mu, log_std, z, x_hat
        
            
    def sample_latent(self, mu, log_std): #TODO sample K latent points from posterior
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_std) * eps
    
    
    @classmethod
    def log_gauss(cls, x, mu, log_std):
        return (- .5 * torch.log(2 * torch.pi) - log_std 
                - .5 * (x - mu).pow(2) / torch.exp(log_std).pow(2) 
                ).sum(dim=-1)
    

    def log_prior(self, z):
        return self.log_gauss(z, 0, 0)
    
    
    def log_posterior(self, z, mu, log_std):
        return self.log_gauss(z, mu, log_std)

        
    
    def loss(self, x, z, x_hat, mu, log_std):
        """
        x: (B, C, W, H)
        z: (B, d) 
        x_hat: ()
        """
        pass

        
    
        
def display(x):
    """
    x (Tensor): B, C, W, H
    """
    x = x.detach().cpu()
    x = make_grid(x).permute(1, 2, 0)
    plt.imshow(x)


train_ds = datasets.MNIST(root=DATAPATH,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             lambda x: (x>0.5).to(x.dtype)
                             ])
                         )

train_loader = DataLoader(train_ds, shuffle=True, batch_size=32)

vae = VAE(latent_dim=2)

epochs = 1

for epoch in range(epochs):
    for x, _ in train_loader:
        z, x_hat = vae(x)
        