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

class Discriminator(nn.Module):
    def __init__(
        self: nn.Module, 
        dis_features: int,
        channels: int
    ) -> None:
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Dropout(0.4),
            # input is (channels) x 128 x 128
            nn.Conv2d(channels, dis_features, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dis_features) x 64 x 64
            nn.Conv2d(dis_features, dis_features * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dis_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dis_features*2) x 32 x 32
            nn.Conv2d(dis_features * 2, dis_features * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dis_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dis_features*4) x 16 x 16
            nn.Conv2d(dis_features * 4, dis_features * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dis_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dis_features*8) x 8 x 8
            nn.Conv2d(dis_features * 8, dis_features * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dis_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dis_features*8) x 8 x 8
            nn.Conv2d(dis_features * 16, dis_features * 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dis_features * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dis_features*16) x 4 x 4
            nn.Conv2d(dis_features * 32, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        return self.main(input)