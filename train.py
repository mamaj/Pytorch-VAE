import matplotlib.pyplot as plt
from celluloid import Camera
import torch
from torch.distributions.kl import kl_divergence as kl
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange

from utils import display, load_mnist, plot_latent_images, visualize_latent
from vae_torch_dist import VAE


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    mnist_train, mnist_test = load_mnist()
    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=64)
    # test_loader = DataLoader(mnist_test, shuffle=False, batch_size=64)
    
    vae = VAE(latent_dim=2)
    vae.to(DEVICE)

    optim = torch.optim.Adam(lr=0.001, params=vae.parameters())
    epochs = 10

    train_loss = []
    
    # setup animation
    fig, ax = plt.subplots(figsize=(10, 10))
    camera = Camera(fig)
    
    for _ in trange(epochs):
        for i, (x, _) in tqdm(enumerate(train_loader)):
            optim.zero_grad()
            x = x.to(DEVICE)
            
            posterior = vae.encode(x)
            z = posterior.rsample()   
            elbo = vae.decode(z).log_prob(x) - kl(posterior, vae.prior)
            loss = -1 * elbo.mean()
            
            train_loss.append(loss.item())
            loss.backward()
            optim.step()
            
            # capture animation frame
            if i % 10 == 0:
                plot_latent_images(vae, 20, ax=ax)
                camera.snap()
    
    # save animation
    animation = camera.animate()
    animation.save('animation.gif')
    
    # display reconstruction
    z = vae.encode(x).sample()
    x_hat = vae.decode(z).base_dist.probs
    display(x)
    display(x_hat)    
    
    
    
if __name__ == '__main__':
    main()
