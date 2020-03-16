# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def make_dataloader(dataroot, transforms, batch_size, workers=6):
    dataset = ImageFolder(root=dataroot, transform=transforms)

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
