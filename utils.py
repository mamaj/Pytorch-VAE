from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import datasets, transforms

DATAPATH = Path.home() / '.datasets'

def display(x):
    """
    x (Tensor): B, C, W, H
    """
    x = x.detach().cpu()
    x = make_grid(x).permute(1, 2, 0)
    plt.imshow(x)
    

def load_mnist():
    return datasets.MNIST(root=DATAPATH,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             lambda x: (x>0.5).to(x.dtype)
                             ])
                         )