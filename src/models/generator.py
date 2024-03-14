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

        n = int(dimensions / 8)

        conv_blocks = []

        # Input layer
        conv_blocks.append(self._block(gen_input, int(gen_features * n), 4, 1, 0))

        # Hidden layers
        while (n % 2) == 0:
            conv_blocks.append(self._block(int(gen_features * n), int(gen_features * (n / 2)), 4, 2, 1))
            n //= 2

        # Output layer
        conv_blocks.append(self._block(gen_features, channels, 4, 2, 1, final_layer=True))

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
                nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)),
                nn.Tanh()
            )
        else:    
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.PReLU()
                #nn.LeakyReLU(0.2, inplace=True)
            )

    #------------------------------------------------------------------------

    def forward(self, input):
        return self.main(input)

#----------------------------------------------------------------------------