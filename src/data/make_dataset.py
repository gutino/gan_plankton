# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def default_transformations(image_size):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def make_dataloader(dataroot, image_size=64, batch_size=128, workers=2):
    transformations = default_transformations(image_size)

    dataset = ImageFolder(root=dataroot, transform=transformations)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    return dataloader


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    pass


if __name__ == "__main__":
    main()
