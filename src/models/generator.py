# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author        Jonathon Gibbs
# Email         pszjg@nottingham.ac.uk
# Website       https://www.jonathongibbs.com
# Github        https://github.com/DrJonoG/
# StomataHub    https://www.stomatahub.com
#----------------------------------------------------------------------------

import torch.nn as nn

#----------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(
        self: nn.Module, 
        gen_input: int,
        gen_features: int,
        channels: int
    ) -> None:

        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(gen_input, gen_features * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_features * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gen_features*32) x 4 x 4
            nn.ConvTranspose2d(gen_features * 32, gen_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gen_features*16) x 4 x 4
            nn.ConvTranspose2d(gen_features * 16, gen_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gen_features*8) x 8 x 8
            nn.ConvTranspose2d(gen_features * 8, gen_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gen_features*4) x 16 x 16
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gen_features*2) x 32 x 32
            nn.ConvTranspose2d(gen_features * 2,     gen_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (gen_features) x 64 x 64
            nn.ConvTranspose2d(    gen_features,      channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)