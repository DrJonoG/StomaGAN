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

class Discriminator(nn.Module):
    def __init__(
        self: nn.Module, 
        dis_features: int,
        channels: int,
        dimensions: int = 256
    ) -> None:

        super(Discriminator, self).__init__()

        self.kwargs = {
            'dis_features': dis_features, 
            'channels': channels,
            'dimensions': dimensions
        }

        #--------------------------------------------------------------------
        ## Adjustments to Discriminator to easily adapt to changes in image dimensions
        #--------------------------------------------------------------------

        n_layers = 4 #int(log(dimensions)/log(2)) - 2
        
        conv_blocks = []

        conv_blocks.append(self._block(channels, dis_features,  layer="input"))

        for i in range(0, n_layers):
            conv_blocks.append(self._block((dis_features * (2**i)), (dis_features * (2**(i+1)))))

        conv_blocks.append(self._block(dis_features * (2**n_layers), 1,  layer="final"))

        self.main = nn.Sequential(*conv_blocks)


    #------------------------------------------------------------------------

    def _block (
        self, 
        in_channels, 
        out_channels, 
        layer="hidden"
    ):
        if layer == "final":
            return nn.Sequential (
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2),
                nn.Sigmoid()
            )
        elif layer == "input":
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.PReLU(),
                nn.Dropout(0.5)
            )
        elif layer == "hidden":
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),                
                nn.PReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),                
                nn.PReLU(),
                nn.Dropout(0.5)
            )

    #------------------------------------------------------------------------

    def forward(self, input):
        return self.main(input)

#----------------------------------------------------------------------------