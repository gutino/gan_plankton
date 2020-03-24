import torch
import random

import torch.device as torch_device
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel

random.seed(100)
torch.manual_seed(100)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_size, feature_map_size, channel_size, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_size, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(
                feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(feature_map_size, channel_size, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, feature_map_size, channel_size, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channel_size, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class dcgan:
    def __init__(self, discriminator, generator, learning_rate, beta1, ngpu):
        # Define device to run model
        self.device = torch_device(
            "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
        )

        # Create the Generator
        self.generator = generator.to(self.device)
        # Create the Discriminator
        self.discriminator = discriminator.to(self.device)

        # Handle multi-gpu if desired
        if (self.device.type == "cuda") and (ngpu > 1):
            self.generator = nn.DataParallel(self.generator, list(range(ngpu)))
            netD = nn.DataParallel(netD, list(range(ngpu)))

        # init weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # Set optimizers
        self.criterion = nn.BCELoss()

        self.optimizerG = optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
        )
        self.optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
        )

    def train(self, dataloader, num_epochs):
        pass