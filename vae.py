import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    
    
    def encode(self, x):
        mean_logstd = self._encoder(x)
        mu, log_std = mean_logstd[:, :self.d], mean_logstd[:, self.d:]
        return mu, log_std
    
    
    def decode(self, z):
        logits = self._decoder(z)
        return logits.view(z.shape[0], 1, 28, 28)
        
    
    def forward(self, x):
        mu, log_std = self.encode(x)
        
        if self.training:
            z = self.sample_latent(mu, log_std)
        else:
            z = mu
        logits = self.decode(z)
        return mu, log_std, z, logits
        
            
    def sample_latent(self, mu, log_std): #TODO sample K latent points from posterior
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_std) * eps
    
    
    @classmethod
    def log_gauss(cls, x, mu, log_std):
        return (- .5 * torch.log(torch.tensor(2. * torch.pi)) - log_std 
                - .5 * (x - mu).pow(2.) / torch.exp(log_std).pow(2.) 
                ).sum(dim=-1)
    

    def log_prior(self, z):
        return self.log_gauss(z, torch.tensor(0), torch.tensor(0))
    
    
    def log_posterior(self, z, mu, log_std):
        return self.log_gauss(z, mu, log_std)


    def log_likelihood(self, x, logits):
        p = F.binary_cross_entropy_with_logits(x, logits, reduce=False)
        return p.flatten(start_dim=1).sum(-1)
        
    
    def loss(self, x, z, logits, mu, log_std):
        """
        x: (B, C, W, H)
        z: (B, d) #TODO Handle multiple samples
        logit: (B, C, W, H)
        mu: (B, d)
        log_std: (B, d)
        """
        return (- self.log_likelihood(x, logits)
                + self.log_posterior(z, mu, log_std)
                - self.log_prior(z)).mean()
        


mnist_train = load_mnist()
train_loader = DataLoader(mnist_train, shuffle=True, batch_size=32)


vae = VAE(latent_dim=2)
vae.to(DEVICE)


optim = torch.optim.Adam(lr=0.01, params=vae.parameters())
epochs = 2

train_loss = []
for epoch in trange(epochs):
    for x, _ in tqdm(train_loader):
        x = x.to(DEVICE)
        optim.zero_grad()
        
        mu, log_std, z, logits = vae(x)
        loss = vae.loss(x, z, logits, mu, log_std)
        
        train_loss.append(loss.item())
        print(loss.item())
        loss.backward()
        optim.step()