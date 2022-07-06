from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import torch.distributions as td

DATAPATH = Path.home() / '.datasets'

def display(x):
    """
    x (Tensor): B, C, W, H
    """
    x = x.detach().cpu()
    x = make_grid(x).permute(1, 2, 0)
    plt.imshow(x)
    

def load_mnist():
    train_ds = datasets.MNIST(
        root=DATAPATH,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            lambda x: (x>0.5).to(x.dtype)
            ])
        )
    
    test_ds = datasets.MNIST(
        root=DATAPATH,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            lambda x: (x>0.5).to(x.dtype)
            ])
        )
    
    return train_ds, test_ds
    
    
def visualize_latent(test_loader, vae):
    latents = []
    labels = []
    for x, y in test_loader:
        z = vae.encode(x).base_dist.loc
        latents.append(z.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    latents = np.vstack(latents)
    labels = np.hstack(labels)
    
    plt.scatter(latents[:, 0], latents[:, 1], c=labels)


def plot_latent_images(model, n, digit_size=28, ax=None):
    """Plots n x n digit images decoded from the latent space."""
    
    norm = td.Normal(0, 1)
    grid_x = norm.icdf(torch.linspace(0.05, 0.95, n))
    grid_y = norm.icdf(torch.linspace(0.05, 0.95, n))
    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = torch.tensor([[xi, yi]])
            x_decoded = model.decode(z).base_dist.probs
            digit = torch.reshape(x_decoded[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='Greys_r')
    ax.axis('Off')
    return im