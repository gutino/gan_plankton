import numpy as np
import torch
import random

import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


random.seed(100)
torch.manual_seed(100)


class DCGAN_Generator(nn.Module):
    def __init__(self, latent_size, feature_map_size, channel_size, ngpu=1):
        super(DCGAN_Generator, self).__init__()
        self.latent_size = latent_size
        self.ngpu = ngpu

        # Parametrizar tamanho dos filtros ou deixar claro que os parâmetros vieram do paper

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                self.latent_size, feature_map_size * 8, 4, 1, 0, bias=False
            ),
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


class DCGAN_Discriminator(nn.Module):
    def __init__(self, feature_map_size, channel_size, ngpu=1):
        super(DCGAN_Discriminator, self).__init__()
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


class GAN:
    def __init__(self, discriminator, generator, learning_rate, beta1, ngpu=1):
        # Define device to run model
        self.device = torch.device(
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
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

        # Set optimizers
        self.loss_func = nn.BCELoss()

        self.optimizerG = optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
        )
        self.optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
        )

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, dataloader, num_epochs):
        fixed_noise = torch.randn(
            64, self.generator.latent_size, 1, 1, device=self.device
        )

        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch

        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.discriminator.zero_grad()
                # Format batch

                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), 1, device=self.device)
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.loss_func(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(
                    b_size, self.generator.latent_size, 1, 1, device=self.device
                )
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(0)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.loss_func(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################self.latent_size
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.zero_grad()
                label.fill_(1)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.loss_func(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                        % (
                            epoch,
                            num_epochs,
                            i,
                            len(dataloader),
                            errD.item(),
                            errG.item(),
                            D_x,
                            D_G_z1,
                            D_G_z2,
                        )
                    )

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # checar diferença entre pixel das imagens
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or (
                    (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    self.img_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                    )

                iters += 1

    def generate(self, latent_tensors):
        with torch.no_grad():
            return self.generator(latent_tensors).detach().cpu()

    def predict_discriminator(self, images):
        dev_images = images.to(self.device)
        # Forward pass real batch through D
        with torch.no_grad():
            return self.discriminator(dev_images).view(-1)
