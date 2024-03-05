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
        channels: int = 1
    ) -> None:

        super(Generator_, self).__init__()

        self.main = nn.Sequential(
            self._block(gen_input, gen_features * 32, 4, 1, 0),
            self._block(gen_features * 32, gen_features * 16, 4, 2, 1),
            self._block(gen_features * 16, gen_features * 8, 4, 2, 1),
            self._block(gen_features * 8, gen_features * 4, 4, 2, 1),
            self._block(gen_features * 4, gen_features * 2, 4, 2, 1),
            self._block(gen_features * 2,     gen_features, 4, 2, 1),
            self._block(gen_features, channels, 4, 2, 1, final_layer=True)
        )



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

    def forward(self, input):
        return self.main(input)