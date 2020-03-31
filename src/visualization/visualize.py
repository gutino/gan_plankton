import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def print_samples(dataloader, gpu=True):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=2, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )


def plot_loss(G_losses, D_losses, fig_size=(10, 5), notebook=False, img_path=None):
    plt.figure(figsize=fig_size)
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    if notebook:
        plt.show()
    else:
        plt.savefig(img_path)


def generator_progress(fake_images, notebook=False, gif_path=None):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in fake_images]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True
    )

    if notebook:
        HTML(ani.to_jshtml())
    else:
        ani.save(gif_path, writer="imagemagick", fps=60)


def compare_fake_real(dataloader, device, fake_images, notebook=False, img_path=None):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=5, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(fake_images[-1], (1, 2, 0)))

    if notebook:
        plt.show()
    else:
        plt.savefig(img_path)
