import torch
import torch.nn as nn
import torch.distributions as td

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((-1, *self.shape))
    
    
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.d = latent_dim
        
        self._encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2) # mu, log_var
        )

        self._decoder = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 64*6*6),
            nn.ReLU(),
            Reshape(64, 6, 6),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, output_padding=1),
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
        mu_logstd = self._encoder(x)
        mu, log_std = mu_logstd[:, :self.d], mu_logstd[:, self.d:]
        return td.Independent(
            td.Normal(loc=mu, scale=torch.exp(log_std)),
            reinterpreted_batch_ndims=1
        )
    
    
    def decode(self, z):
        logits = self._decoder(z).view(-1, 1, 28, 28)
        return td.Independent(
            td.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=3 
        )
        
    
    def forward(self, x, deterministic=False):
        if deterministic:
            z = self.encode(x).base_dist.loc
            x_hat = self.decode(z).base_dist.probs
        else:
            z = self.encode(x).sample()
            x_hat = self.decode(z).sample()
        return z, x_hat
    
