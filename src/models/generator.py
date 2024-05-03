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
from math import log

#----------------------------------------------------------------------------
    
class Generator_(nn.Module):
    def __init__(
        self: nn.Module, 
        gen_input: int = 128,
        gen_features: int = 64,
        channels: int = 1,
        dimensions: int = 256
    ) -> None:

        super(Generator_, self).__init__()

        self.kwargs = {
            'gen_input': gen_input, 
            'gen_features': gen_features, 
            'channels': channels,
            'dimensions': dimensions
        }

        #--------------------------------------------------------------------
        ## Adjustments to Generator to easily adapt to changes in image dimensions
        #--------------------------------------------------------------------



        n_layers = 4 #int(log(dimensions)/log(2)) - 2

        conv_blocks = []

        conv_blocks.append(self._block(gen_input, int(gen_features * (2**n_layers)), 4, 1, 0, layer = "input"))

        for i in range(n_layers, 0, -1):
            conv_blocks.append(self._block((gen_features * (2**i)), (gen_features * (2**(i-1))),  4, 2, 1))

        # Output layer
        conv_blocks.append(self._block(gen_features, channels, 4, 2, 1, layer="final"))

        self.main = nn.Sequential(*conv_blocks)

    #------------------------------------------------------------------------

    def _block (
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding,
        layer="hidden"
    ):
        if layer == "final":
            return nn.Sequential (
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Tanh()
            )
        elif layer == "hidden" or layer == "input":    
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                #nn.LeakyReLU(0.2, inplace=True)
            )

    #------------------------------------------------------------------------

    def forward(self, input):
        return self.main(input)

#----------------------------------------------------------------------------