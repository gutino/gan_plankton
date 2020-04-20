import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def print_samples(dataloader, gpu=True):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    print_batch_images(next(iter(dataloader)), device)


def print_batch_images(imgs_batch, device, num_imgs=64):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                imgs_batch[0].to(device)[:num_imgs], padding=2, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )


def plot_loss(G_losses, D_losses, img_path=None, notebook=True):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    if img_path:
        plt.savefig(img_path)
    if notebook:
        plt.show()
    else:
        plt.close()


def generator_progress(fake_images, gif_path=None, notebook=True):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in fake_images]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True
    )

    if gif_path:
        ani.save(gif_path, writer="imagemagick", fps=60)
    if notebook:
        HTML(ani.to_jshtml())
    else:
        plt.close(fig)


def compare_fake_real(dataloader, device, fake_images, img_path=None, notebook=True):
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

    if img_path:
        plt.savefig(img_path)
    if notebook:
        plt.show()
    else:
        plt.close()
