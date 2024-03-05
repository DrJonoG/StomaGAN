# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author            Jonathon Gibbs
# Email             pszjg@nottingham.ac.uk
# Website           https://www.jonathongibbs.com
# Github            https://github.com/DrJonoG/
# StomataHub        https://www.stomatahub.com
#----------------------------------------------------------------------------

import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from utils.helpers import printer, dotdict
from torchvision.utils import save_image
from argparse import ArgumentParser
from models import Generator_

#----------------------------------------------------------------------------

def main (
	config: object
) -> None:
    gpus = general.gpus.split(",")
	# set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load generator model
    netG = Generator_(config.gen_input, config.gen_features, config.channels).to(device)

    # If trained on multiple GPUs this must also be loaded into DataParallel
    if (device.type == 'cuda') and (len(gpus) > 1):
        netG = nn.DataParallel(netG, list(int(d) for d in gpus))

    netG.load_state_dict(torch.load(config.model_path))

    printer("Generating fake images.... this may take some time.")
    # create random noise
    noise = torch.randn(config.qty, config.gen_input, 1, 1, device=device)
    fake = netG(noise).detach().cpu()

    printer(f"Saving {config.qty} fake images to {config.destination}.")   
    for image in fake:   
        save_image(image, os.path.join(config.destination, str(int(time.time()*1000.0))) + ".jpg")

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-config",  type=str,   help="Path to config",  		default="./config.json")

    args = parser.parse_args()

    parser_error = False

    if not os.path.isfile(args.config):
        printer(f"Error -> configuration file {args.config} not found")
        parser_error = True

    with open(args.config) as config_file:
        try:
            config_raw = json.load(config_file)
        except Exception as e:
            printer(f"Error reading configuration file {args.config}")
            printer(str(e))
            parser_error = True

    if parser_error:
        exit(0)

    # Covert accesibility to dot
    config = dotdict(config_raw['generate'])
    general = dotdict(config_raw['general'])

    if not os.path.isfile(config.model_path):
        printer(f"Error -> The model file {config.model_path} cannot be found.")
        parser_error = True


    if not os.path.exists(config.destination):
        try:
            os.makedirs(config.destination)
        except Exception as e:
            printer(f"Error creating folder {config.destination}.")
            print(str(e))
            parser_error = True

    if config.qty < 0:
        printer(f"Error -> Quantity to generate should be greater than 0, not {config.qty}")
        parser_error = True

    if config.channels < 0:
        printer(f"Error -> Image channels should be greater than 0, not {config.channels}")
        parser_error = True

    
    if parser_error:
        exit(0)

    main(config)