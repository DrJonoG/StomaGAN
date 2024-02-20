# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author        Jonathon Gibbs
# Email         pszjg@nottingham.ac.uk
# Website       https://www.jonathongibbs.com
# Github        https://github.com/DrJonoG/
# StomataHub    https://www.stomatahub.com
#----------------------------------------------------------------------------

# Import comet before other packages
from comet_ml import Experiment, init
from comet_ml.integration.pytorch import log_model, watch

import os
import json
import torch
import random
import datetime
import numpy as np

import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt

from models import Generator, Discriminator
from utils.helpers import dotdict, printer, printer_config
from argparse import ArgumentParser

#----------------------------------------------------------------------------

def weights_init(m):
    # custom weights initialization called on ``netG`` and ``netD``

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#----------------------------------------------------------------------------

def run (
    train: type[dotdict], 
    gen: type[dotdict], 
    dis: type[dotdict], 
    experiment: object
):

    # Generate folders
    folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    train.savepath = os.path.join(train.savepath, folder)

    experiment.set_name(folder)

    if not os.path.exists(train.savepath):
        os.mkdir(train.savepath)

    train.epochpath = os.path.join(train.savepath, "epochs")
    if not os.path.exists(train.epochpath):
        os.mkdir(train.epochpath)
            
    printer(f"Training data saved to {train.savepath}")

    random.seed(train.seed)
    torch.manual_seed(train.seed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results

    printer("Creating dataset", end="")
    # Create the dataset
    dataset = dset.ImageFolder(
        root=train.dataroot,
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=train.channels),
            transforms.Resize((train.resize[0], train.resize[1])),
            transforms.CenterCrop((train.resize[0], train.resize[1])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
         ])
    )
    print(f".... {str(len(dataset))} images available.")

    printer("Creating dataloader", end="")
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=train.batch,
        shuffle=True, 
        num_workers=train.workers
    )
    print(f".... batch size {train.batch} .... {train.workers} workers")

    # Specify GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Plot some training images
    printer("Plotting training images")
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:(8*8)], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.gcf().set_size_inches(10, 5)
    plt.savefig(os.path.join(train.savepath,'real_images.png'), dpi=200)

    printer(f"Saving training images to {os.path.join(train.savepath,'real_images.png')}")
    
    experiment.log_image(os.path.join(train.savepath,'real_images.png'), name="real_images")


    #----------- Model generation -----------------------------------------------

    netG = Generator(train.gen_input, train.gen_features, train.channels).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (len(train.gpus) > 1):
        netG = nn.DataParallel(netG, list(int(d) for d in train.gpus))

    # randomly initialize all weights to mean=0, stdev=0.02.
    printer("Initialising random weights for Generator model")
    netG.apply(weights_init)
    
    if train.debug:
        print(netG)

    printer(f"Generator model saved to {os.path.join(train.savepath,'netG_architecture.pt')}")
    netG_script = torch.jit.script(netG) # Export to TorchScript
    netG_script.save(os.path.join(train.savepath,'netG_architecture.pt')) # Save
    experiment.log_model('netG', os.path.join(train.savepath,'netG_architecture.pt'))

    # Create the Discriminator
    netD = Discriminator(train.dis_features, train.channels).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (len(train.gpus) > 1):
        netD = nn.DataParallel(netD, list(int(d) for d in train.gpus))

    # randomly initialize all weights to mean=0, stdev=0.02.
    printer("Initialising random weights for Discriminator model")
    netD.apply(weights_init)

    if train.debug:
        print(netD)

    printer(f"Discriminator model saved to {os.path.join(train.savepath,'netD_architecture.pt')}")
    netD_script = torch.jit.script(netD) # Export to TorchScript
    netD_script.save(os.path.join(train.savepath,'netD_architecture.pt')) # Save
    experiment.log_model('netD', os.path.join(train.savepath,'netD_architecture.pt'))

    #----------------------------------------------------------------------------

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, train.gen_input, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    if dis.optimizer == "Adam":
        optimizerD = optim.Adam(netD.parameters(), lr=dis.lr, betas=(dis.beta1, 0.999)) # should use SGD

    if gen.optimizer == "Adam":
        optimizerG = optim.Adam(netG.parameters(), lr=gen.lr, betas=(gen.beta1, 0.999))

    #optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.9)


    #----------------------------------------------------------------------------
    # Training Loop
    #----------------------------------------------------------------------------
    with experiment.train():

        watch(netG)
        watch(netD)

        step = 0

        # Lists to keep track of progress
        G_losses = []
        D_losses = []

        printer("Starting Training Loop...")
        # For each epoch
        for epoch in range(train.epochs):
            experiment.log_current_epoch(epoch)
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                #--------------------------------------------------------------------
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                #--------------------------------------------------------------------

                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, train.gen_input, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()
                
                #--------------------------------------------------------------------
                # (2) Update G network: maximize log(D(G(z)))           
                #--------------------------------------------------------------------
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    printer('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, train.epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end="\r")
      
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                experiment.log_metric("errG", errG.item(), step=step)
                experiment.log_metric("errD", errD.item(), step=step)

                experiment.log_metric("D_x", D_x, step=step)
                experiment.log_metric("D_G_z1", D_G_z1, step=step)
                experiment.log_metric("D_G_z2", D_G_z2, step=step)

                step += 1


            # Generate fake image and save grid
            fake = netG(fixed_noise).detach().cpu()
            fake_grid = vutils.make_grid(fake, padding=2, normalize=True)

            vutils.save_image(fake_grid, os.path.join(train.epochpath, f"fake_epoch_{epoch}.jpg"))            
            experiment.log_image(os.path.join(train.epochpath, f"fake_epoch_{epoch}.jpg"), name=f"fake_epoch_{epoch}")

            # Save state to allow resumption if failure
            gen_state = {
                'epoch': epoch,
                'state_dict': netG.module.state_dict(),
                'optimizer': optimizerG.state_dict(),
            }

            torch.save(gen_state,os.path.join(train.savepath, 'gen_checkpoint.t7'))
            experiment.log_model(experiment, gen_state, model_name="gen_checkpoint")

            dis_state = {
                'epoch': epoch,
                'state_dict': netD.module.state_dict(),
                'optimizer': optimizerD.state_dict(),
            }

            torch.save(dis_state,os.path.join(train.savepath, 'dis_checkpoint.t7'))
            experiment.log_model(experiment, dis_state, model_name="dis_checkpoint")

    #----------------------------------------------------------------------------
    # End Training Loop
    #----------------------------------------------------------------------------

    # Save final models
    torch.save(netG, os.path.join(train.savepath, 'netg_final.pth'))
    torch.save(netD, os.path.join(train.savepath, 'netd_final.pth'))

    # Training loss
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.savefig(os.path.join(train.savepath,'training_loss.png'), dpi=200)

    experiment.end()

#----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-config",  type=str,   help="Path to config",  default="./config.json")

    args = parser.parse_args()


    if not os.path.isfile(args.config):
        printer("Error configuration file not found")
        exit(0)


    with open(args.config) as config_file:
        try:
            config_raw = json.load(config_file)
        except Exception as e:
            printer(f"Error reading configuration file {args.config}")
            printer(str(e))
            exit(0)



    # Covert accesibility to dot
    config          = dotdict(config_raw)
    config.comet    = dotdict(config.comet_dl)
    config.train    = dotdict(config.train)
    config.gen      = dotdict(config.generator)
    config.dis      = dotdict(config.discriminator)
    config_errors = False

    printer("Processing config file. \n\nGeneral model configuration\n")

    #----------------------------------------------------------------------------

    if not os.path.isdir(config.train.dataroot):
        printer("Error invalid data path specified in config file")
        config_errors = True
    else:
        printer_config("dataroot", config.train.dataroot)

    #----------------------------------------------------------------------------

    if config.train.seed: 
        config.train.seed = int(config.train.seed)
    else:
        config.train.seed = random.randint(1, 1000)
        config_raw['train']['seed'] = config.train.seed 

    printer_config("seed", config.train.seed)

    #----------------------------------------------------------------------------

    if not config.train.cache:
        config.train.cache = "cache.pt"

    printer_config("cache", config.train.cache)

    #----------------------------------------------------------------------------

    if not config.train.resize:
        config.train.resize = [128, 128]
    else:
        try:
            resize = config.train.resize.split("x")
            if len(resize) == 1:
                config.train.resize = [int(resize[0]), int(resize[0])]
            else:
                config.train.resize = [int(resize[0]), int(resize[1])]

            printer_config("resize",f"{config.train.resize[0]}x{config.train.resize[1]}")

        except Exception as e:
            printer("Error reading resize. Please ensure format is \"widthxheight\"")
            config_errors = True

    #----------------------------------------------------------------------------

    if config.train.gpus:
        try: 
            config.train.gpus = config.train.gpus.split(",")
            
            printer_config("gpus", config.train.gpus)

        except Exception as e:
            printer("Error unable to parse gpus, please ensure format is \"int,int,etc.\".")
            config_errors = True

    #----------------------------------------------------------------------------

    if config.train.channels:
        try: 
            config.train.channels = int(config.train.channels)
            if config.train.channels < 0: 
                raise TypeError()

            printer_config("channels", config.train.channels)

        except Exception as e:
            printer("Error unable to parse channels, please enter a positive integer.")
            config_errors = True

    #----------------------------------------------------------------------------

    if config.train.workers:
        try: 
            config.train.workers = int(config.train.workers)
            if config.train.workers < 0: 
                raise TypeError()

            printer_config("workers", config.train.workers)

        except Exception as e:
            printer("Error unable to parse workers, please enter a positive integer.")
            config_errors = True

    #----------------------------------------------------------------------------

    if config.train.batch:
        try: 
            config.train.batch = int(config.train.batch)
            if config.train.batch < 0: 
                raise TypeError()

            printer_config("batch", config.train.batch)

        except Exception as e:
            printer("Error unable to parse batch, please enter a positive integer.")
            config_errors = True

    #----------------------------------------------------------------------------

    if config.train.epochs:
        try: 
            config.train.epochs = int(config.train.epochs)
            if config.train.epochs < 0: 
                raise TypeError()

            printer_config("epochs", config.train.epochs)

        except Exception as e:
            printer("Error unable to parse epochs, please enter a positive integer.")
            config_errors = True

    #----------------------------------------------------------------------------

    if config.train.gen_input:
        try: 
            config.train.gen_input = int(config.train.gen_input)
            # Verify number is a power of 2
            if config.train.gen_input < 0 and not ((config.train.gen_input != 0) and (config.train.gen_input & (config.train.gen_input-1) == 0)): 
                raise TypeError()

            printer_config("gen_input", config.train.gen_input)

        except Exception as e:
            printer("Error unable to parse gen_input, please enter a integer to the power of 2.")
            printer(str(e))
            config_errors = True

    #----------------------------------------------------------------------------

    if config.train.gen_features:
        try: 
            config.train.gen_features = int(config.train.gen_features)
            # Verify number is a power of 2
            if config.train.gen_features < 0 and not  ((config.train.gen_features != 0) and (config.train.gen_features & (config.train.gen_features-1) == 0)): 
                raise TypeError()

            printer_config("gen_features", config.train.gen_input)
            
        except Exception as e:
            printer("Error unable to parse gen_features, please enter a integer to the power of 2.")
            printer(str(e))
            config_errors = True

    #----------------------------------------------------------------------------

    if config.train.dis_features:
        try: 
            config.train.dis_features = int(config.train.dis_features)
            # Verify number is a power of 2
            if config.train.dis_features < 0 and not  ((config.train.dis_features != 0) and (config.train.dis_features & (config.train.dis_features-1) == 0)):  
                raise TypeError()

            printer_config("dis_features", config.train.dis_features)
            
        except Exception as e:
            printer("Error unable to parse dis_features, please enter a integer to the power of 2.")
            printer(str(e))
            config_errors = True

    #----------------------------------------------------------------------------


    if (not os.path.isfile(config.comet.path)):
        printer("Error invalid comet path specified in config file.") 
        config_errors = True
    else:   
        printer_config("comet", config.comet.path)

        with open(config.comet.path) as comet_file:
            try:
                comet = json.load(comet_file)
                config.comet = dotdict(comet)
            except Exception as e:
                printer(f"Error reading configuration file {config.comet.path}")
                printer(str(e))
                config_errors = True

    #----------------------------------------------------------------------------

    if config_errors:
        exit(0)
    else:
        # setup the experiement        
        experiment = Experiment (
            api_key = config.comet.api_key,
            project_name = config.comet.project_name,
            workspace = config.comet.workspace,
            log_env_details = False
        )


        parameters = {
            "train": config_raw['train'],
            "generator": config_raw['generator'],
            "discriminator": config_raw['discriminator']
        }

        experiment.log_parameters(parameters)


        # Train
        run(config.train, config.gen, config.dis, experiment)

