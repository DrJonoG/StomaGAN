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

class Discriminator(nn.Module):
    def __init__(
        self: nn.Module, 
        dis_features: int,
        channels: int
    ) -> None:
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(channels, dis_features, 4, stride=2, padding=1, bias=False),
            #nn.LeakyReLU(0.4, inplace=True),
            nn.PReLU(),
            nn.Dropout(0.4),

            self._block(dis_features, dis_features * 2, 4, stride=2, padding=1),
            self._block(dis_features * 2, dis_features * 4, 4, stride=2, padding=1),
            self._block(dis_features * 4, dis_features * 8, 4, stride=2, padding=1),
            self._block(dis_features * 8, dis_features * 16, 4, stride=2, padding=1),
            self._block(dis_features * 16, dis_features * 32, 4, stride=2, padding=1),
            self._block(dis_features * 32, 1, 4, 1, 0, final_layer=True)
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
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride)),
                nn.Sigmoid()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                #nn.LeakyReLU(0.4, inplace=True),
                
                nn.PReLU(),
                nn.Dropout(0.4)
            )

    def forward(self, input):
        return self.main(input)