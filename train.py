import matplotlib.pyplot as plt
from celluloid import Camera
import torch
from torch.distributions.kl import kl_divergence as kl
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from utils import display, load_mnist, plot_latent_images, visualize_latent
from vae_torch_dist import VAE


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
animate = False

def main():
    mnist_train, mnist_test = load_mnist(binary=True)
    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=64)
    # test_loader = DataLoader(mnist_test, shuffle=False, batch_size=64)
    
    vae = VAE(latent_dim=2)
    vae.to(DEVICE)

    optim = torch.optim.Adam(lr=0.005, params=vae.parameters())
    epochs = 5

    train_loss = []
    
    # setup animation
    if animate:
        fig, ax = plt.subplots(figsize=(10, 10))
        camera = Camera(fig)
    
    for _ in trange(epochs):
        for i, (x, _) in enumerate(tqdm(train_loader)):
            optim.zero_grad()
            x = x.to(DEVICE)
            
            loss = -1 * vae.elbo(x, n_samples=1).mean()
            
            train_loss.append(loss.item())
            loss.backward()
            optim.step()
            
            # capture animation frame
            if animate and i % 200 == 0:
                plot_latent_images(vae, 20, ax=ax)
                camera.snap()
    
    # save animation
    if animate:
        animation = camera.animate()
        animation.save('animation2.gif')
        animation.save('animation2.mp4')
    
    # display reconstruction
    z = vae.encode(x).sample()
    x_hat = vae.decode(z).base_dist.probs
    display(x)
    display(x_hat)    
    
    
    
if __name__ == '__main__':
    main()
