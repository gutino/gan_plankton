import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils


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
