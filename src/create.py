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
import torch


from argparse import ArgumentParser
from utils.helpers import printer, dotdict, parse_int
from models import Generator

#----------------------------------------------------------------------------

def create_from_model (
	netG: Generator,
	destination: str,
	quantity: int,
	dimensions: [int, int],
	arrangement: str,
	spacing: str,
	magnification: int,
	gen_input: int,
	device: object
) -> None:
	# create random noise
    noise = torch.randn(quantity, gen_input, 1, 1, device=device)
    fake = netG(noise).detach().cpu()

    printer(f"Saving {config.qty} fake images to {config.destination}.")   
    for image in fake:   
        print(image)
        #save_image(image, os.path.join(config.destination, str(int(time.time()*1000.0))) + ".jpg")

#----------------------------------------------------------------------------


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument("-config",  type=str,   help="Path to config",  default="./config.json")
	args = parser.parse_args()

	with open(args.config) as config_file:
		try:
			config_raw = json.load(config_file)
		except Exception as e:
			printer(f"Error reading configuration file {args.config}")
			printer(str(e))
			exit(0)

	config_error = False
	# Covert accesibility to dot
	config = dotdict(config_raw['create'])
	general = dotdict(config_raw['general'])

	#------------------------------------------------------------------------
	## Config validation 
	#------------------------------------------------------------------------

	if not os.path.exists(config.destination):
		try:
			os.path.makedirs(config.destination)
		except Exception as e:
			printer(f"Error creating folder {config.destination}")
			print(e)
			config_error = True


	try:
		config.dimensions = (int(x) for x in config.dimensions.split("x"))
	except Exception as e:
		printer("Error processing dimensions. Please ensure the format is wxh")
		config_error = True

	if config.arrangement not in ['line','random']:
		printer("Error processing arrangement please use either line or random")
		config_error = True

	if config.spacing not in ['sparse','dense','random']:
		printer("Error processing spacing, please use either sparse, dense or random")
		config_error = True

	config_error = (True if parse_int('quantity', config.quantity, 1, 10000) else config_error)
	config_error = (True if parse_int('magnification', config.magnification, 100, 1000) else config_error)
	config_error = (True if parse_int('gen_input', general.gen_input, 1, 1024) else config_error)
	config_error = (True if parse_int('channels', general.channels, 1, 4) else config_error)
	config_error = (True if parse_int('gen_features', general.gen_features, 1, 1024) else config_error)

	try: 
		general.gpus = general.gpus.split(",")
	except Exception as e:
		printer("Error unable to parse gpus, please ensure format is \"int,int,etc.\".")

	from_model = False
	# Load from model if file
	if os.path.isfile(config.source) and not config_error:
		try: 	
			# set up device
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

			# load generator model
			netG = Generator(general.gen_input, general.gen_features, general.channels).to(device)

			# If trained on multiple GPUs this must also be loaded into DataParallel
			if (device.type == 'cuda') and (len(general.gpus) > 1):
				netG = nn.DataParallel(netG, list(int(d) for d in general.gpus))

			netG.load_state_dict(torch.load(config.model_path))

			from_model = True
		except Exception as e:
			printer("Error -> Unable to load model file")
			print(e)
			config_error = True
	elif not os.path.isdir(config.source):
		printer("Error processing source. Please enter a valid model or directory")
		config_error = True

	if config_error:
		exit(0)

	if from_model:
		create_from_model(
			netG, 
			config.destination, 
			config.quantity, 
			config.dimensions, 
			config.arrangement, 
			config.spacing, 
			config.magnification,
			config.gen_input,
			device
		)





