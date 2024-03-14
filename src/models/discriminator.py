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
            'channels': channels
        }

        #--------------------------------------------------------------------
        ## Adjustments to Discriminator to easily adapt to changes in image dimensions
        #--------------------------------------------------------------------

        n = int(dimensions / 8)
        n_layers = int(log(n, 2))
        
        conv_blocks = []
        
        # Input layer
        conv_blocks.append(
            nn.Sequential(
                nn.Conv2d(channels, dis_features, 4, stride=2, padding=1, bias=False),
                nn.PReLU(),
                nn.Dropout(0.4)
            )
        )

        for i in range(0, n_layers):
            conv_blocks.append(self._block((dis_features * (2**i)), (dis_features * (2**(i+1))), 4, stride=2, padding=1))

        conv_blocks.append(self._block(dis_features * n, 1, 4, 1, 0, final_layer=True))

        self.main = nn.Sequential(*conv_blocks)


    #------------------------------------------------------------------------

    def _block (
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride,  
        padding,
        final_layer=False
    ):
        if final_layer:
            return nn.Sequential (
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride)),
                nn.Sigmoid()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),                
                nn.PReLU(),
                nn.Dropout(0.4)
            )

    #------------------------------------------------------------------------

    def forward(self, input):
        return self.main(input)

#----------------------------------------------------------------------------