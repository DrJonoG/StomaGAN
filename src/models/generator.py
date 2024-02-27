# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author            Jonathon Gibbs
# Acknowledgements  Based on original work on Pytorch
# Email             pszjg@nottingham.ac.uk
# Website           https://www.jonathongibbs.com
# Github            https://github.com/DrJonoG/
# StomataHub        https://www.stomatahub.com
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
            self._block(gen_input, gen_features * 32, 4, 1, 0, bias=False),
            self._block(gen_features * 32, gen_features * 16, 4, 2, 1, bias=False),
            self._block(gen_features * 16, gen_features * 8, 4, 2, 1, bias=False),
            self._block(gen_features * 8, gen_features * 4, 4, 2, 1, bias=False),
            self._block(gen_features * 4, gen_features * 2, 4, 2, 1, bias=False),
            self._block(gen_features * 2,     gen_features, 4, 2, 1, bias=False),
            
            nn.ConvTranspose2d(    gen_features,      channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )



    def _block (
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding,
        bias=False
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)