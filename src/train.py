# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author            Jonathon Gibbs
# Acknowledgements  Based on original work on Pytorch
# Email             pszjg@nottingham.ac.uk
# Website           https://www.jonathongibbs.com
# Github            https://github.com/DrJonoG/
# StomataHub        https://www.stomatahub.com
#----------------------------------------------------------------------------

# Import comet before other packages
from comet_ml import Experiment, init
from comet_ml.integration.pytorch import log_model, watch

import os
import copy
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
from utils.helpers import dotdict, printer, printer_config, print_progress_bar
from argparse import ArgumentParser

#----------------------------------------------------------------------------

def calculate_gradient_penalty (
    model, 
    real_images, 
    fake_images, 
    device
):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

#----------------------------------------------------------------------------

def get_real_labels(batch_size, device):
    #labels = np.random.uniform(0.7, 1.0, size=(batch_size,))
    return torch.full((batch_size,), 1.0, dtype=torch.float, device=device)   
    #return torch.from_numpy(labels).float().to(device)

#----------------------------------------------------------------------------

def get_fake_labels(batch_size, device):
    #labels = np.random.uniform(0.0, 0.3, size=(batch_size,))
    return torch.full((batch_size,), 0.0, dtype=torch.float, device=device)    
    #return torch.from_numpy(labels).float().to(device)

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
    #----------------------------------------------------------------------------
    ## Dataset intialisation
    #----------------------------------------------------------------------------

    folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    train.savepath = os.path.join(train.savepath, folder)

    experiment.set_name(folder)

    # Log implementation of model
    experiment.log_code(file_name='./src/models/discriminator.py')
    experiment.log_code(file_name='./src/models/generator.py')

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
            #transforms.GaussianBlur(kernel_size=(5)),
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
    real_grid = vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True)
    real_grid = transforms.ToPILImage()(real_grid) 

    experiment.log_image(real_grid, name="Real images")
    

    #----------------------------------------------------------------------------
    ## Model intialisation 
    #----------------------------------------------------------------------------

    netG = Generator(train.gen_input, train.gen_features, train.channels).to(device)

    printer(f"Generator model saved to {os.path.join(train.savepath,'netG_architecture.pt')}")
    #netG_script = torch.jit.script(netG) # Export to TorchScript
    #netG_script.save(os.path.join(train.savepath,'netG_architecture.pt')) # Save
    experiment.log_model('netG', os.path.join(train.savepath,'netG_architecture.pt'))

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (len(train.gpus) > 1):
        netG = nn.DataParallel(netG, list(int(d) for d in train.gpus))

    # randomly initialize all weights to mean=0, stdev=0.02.
    printer("Initialising random weights for Generator model")
    netG.apply(weights_init)
    
    if train.debug:
        print(netG)    
 
    # Create the Discriminator
    netD = Discriminator(train.dis_features, train.channels).to(device)

    printer(f"Discriminator model saved to {os.path.join(train.savepath,'netD_architecture.pt')}")
    #netD_script = torch.jit.script(netD) # Export to TorchScript
    #netD_script.save(os.path.join(train.savepath,'netD_architecture.pt')) # Save
    experiment.log_model('netD', os.path.join(train.savepath,'netD_architecture.pt'))

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (len(train.gpus) > 1):
        netD = nn.DataParallel(netD, list(int(d) for d in train.gpus))

    # randomly initialize all weights to mean=0, stdev=0.02.
    printer("Initialising random weights for Discriminator model")
    netD.apply(weights_init)

    if train.debug:
        print(netD)  

    #----------------------------------------------------------------------------
    ## Loss and Optimizer initialisation
    #----------------------------------------------------------------------------

    # Initialize the ``BCELoss`` function
    criterionD = nn.BCELoss()
    criterionG = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, train.gen_input, 1, 1, device=device)

    if dis.optimizer == "Adam":
        optimizerD = optim.Adam(netD.parameters(), lr=dis.lr, betas=(dis.beta1, 0.999)) # should use SGD
    elif dis.optimizer == "SGD":
        optimizerD = optim.SGD(netD.parameters(), lr=dis.lr, momentum=0.9)

    if gen.optimizer == "Adam":
        optimizerG = optim.Adam(netG.parameters(), lr=gen.lr, betas=(gen.beta1, 0.999))


    #----------------------------------------------------------------------------
    ## Training Loop
    #----------------------------------------------------------------------------
    with experiment.train():

        watch(netG)
        watch(netD)

        step = 0
        totalSteps = train.epochs * len(dataloader)
        top_k = train.batch
        min_k = int(train.batch * 0.3)

        # Variables for saving models
        dis_lowest_loss = float("inf")
        gen_lowest_loss = float("inf")
        D_G_z2_highest = 0

        printer("Starting Training Loop...")
        # For each epoch
        for epoch in range(train.epochs):

            # Store for plotting
            epoch_G_losses = []
            epoch_D_losses = []
            epoch_D_G_z1 = []
            epoch_D_G_z2 = []
            epoch_D_x = []
            epoch_grad_penalty = []

            

            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                netD.zero_grad()
                netG.zero_grad()
                optimizerD.zero_grad()
                optimizerG.zero_grad()


                #--------------------------------------------------------------------
                ## (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                #--------------------------------------------------------------------

                ## 1.a --------- Train with all-real batch
                # Format batch
                real_images = data[0].to(device)
                batch_size = real_images.size(0)

                # Adjust the top elements to select
                if epoch > train.warmup_epochs:
                    top_k = batch_size - epoch
                    if top_k < min_k:
                        top_k = min_k
                        
                # Adjust qunatity to be filtered in Generator training
                if top_k > batch_size: top_k = batch_size 
                # Forward pass real batch through D
                real_output = netD(real_images).view(-1)
                real_labels = get_real_labels(batch_size, device)
                # Calculate loss on all-real batch
                loss_D_real = criterionD(real_output, real_labels)
                # Calculate gradients for D in backward pass
                loss_D_real.backward()
                D_x = real_output.mean().item()

                ## 1.b --------- Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, train.gen_input, 1, 1, device=device)
                # Generate fake image batch with G
                fake_images = netG(noise)            
                # Classify all fake batch with D and Get labels
                fake_output = netD(fake_images.detach()).view(-1)                
                fake_output, _ = torch.topk(fake_output, top_k)
                fake_labels = get_fake_labels(top_k, device)
                # Calculate D's loss on the all-fake batch
                loss_D_fake = criterionD(fake_output, fake_labels)
                # Calculate the gradients for this batch,
                loss_D_fake.backward()                
                D_G_z1 = fake_output.mean().item()
                # Update D
                optimizerD.step()
                #Compute error of D as sum over the fake and the real batches
                loss_D = loss_D_real + loss_D_fake        

                # Adversarial loss
                adversarial_loss = 0 #-D_x + D_G_z1 + lambda_gp * gradient_penalty    

                
                #--------------------------------------------------------------------
                ## (2) Update G network: maximize log(D(G(z)))           
                #--------------------------------------------------------------------                
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake_images).view(-1)
                output, _ = torch.topk(output, top_k)
                labels = get_real_labels(top_k, device)
                # Calculate G's loss based on this output
                loss_G = criterionG(output, labels)
                # Calculate gradients for G
                loss_G.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                #--------------------------------------------------------------------
                ## (3) Output training stats:
                #--------------------------------------------------------------------

                print_progress_bar(
                        i+1, 
                        len(dataloader), 
                        prefix = f"Epoch {epoch+1} of {train.epochs}. Progress:", 
                        suffix = f"Complete. Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}, Adversarial Loss: {adversarial_loss:.4f}",
                        decimals = 1,
                        length = 30,
                )

                # Save epoch results for averagers
                epoch_G_losses.append(loss_G.item())
                epoch_D_losses.append(loss_D.item())
                epoch_D_G_z1.append(D_G_z1)
                epoch_D_G_z2.append(D_G_z2)
                epoch_D_x.append(D_x)
                epoch_grad_penalty.append(adversarial_loss)

                # Log
                step_logger = {
                    "Gen_Loss": loss_G.item(),
                    "Dis_Loss": loss_D.item(),
                    "D_G_z1": D_G_z1,
                    "D_G_z2": D_G_z2,
                    "Avg. Output": D_x,
                    "Adversarial_Loss": adversarial_loss
                }
                experiment.log_metrics(step_logger, step=step)

                step += 1


            #------------------------------------------------------------------------
            ## Log data (end of epoch)
            #------------------------------------------------------------------------

            epoch_G_losses = np.average(epoch_G_losses)
            epoch_D_losses = np.average(epoch_D_losses)
            epoch_D_G_z1 = np.average(epoch_D_G_z1)
            epoch_D_G_z2 = np.average(epoch_D_G_z2)
            epoch_D_x = np.average(epoch_D_x)
            epoch_grad_penalty = np.average(epoch_grad_penalty)

            epoch_logger = {
                "Epoch_Gen_Loss": epoch_G_losses, 
                "Epoch_Dis_Loss": epoch_D_losses,
                "Epoch_D_G_z1": epoch_D_G_z1, 
                "Epoch_D_G_z2": epoch_D_G_z2,
                "Epoch_D_x": epoch_D_x,
                "Epoch_grad_penalty": epoch_grad_penalty
            }

            experiment.log_metrics(epoch_logger, epoch=epoch)

            # Output training stats
            print_progress_bar(
                    len(dataloader), 
                    len(dataloader), 
                    prefix = f"Epoch {epoch+1} of {train.epochs}. Progress:", 
                    suffix = f"Complete. Loss_D: {epoch_D_losses:.4f}, Loss_G: {epoch_G_losses:.4f}, D(x): {epoch_D_x:.4f}, D(G(z)): {epoch_D_G_z1:.4f} / {epoch_D_G_z2:.4f}",
                    decimals = 1,
                    length = 30,
                    end = "\r\n"
            )

            # Generate fake image and save grid
            fake = netG(fixed_noise).detach().cpu()
            fake_grid = vutils.make_grid(fake, padding=2, normalize=True)
            fake_grid = transforms.ToPILImage()(fake_grid) 
            
            experiment.log_image(fake_grid, name=f"{str(epoch).zfill(8)}")

            # Save current state for resumption
            try:
                # Save state to allow resumption if failure
                gen_state = {
                    'epoch': epoch,
                    'state_dict': netG.state_dict(),
                    'optimizer': optimizerG.state_dict(),
                }
                #with open(os.path.join(train.savepath, 'gen_checkpoint.t7'), 'wb') as f:
                #    torch.save(gen_state, f)

                dis_state = {
                    'epoch': epoch,
                    'state_dict': netD.state_dict(),
                    'optimizer': optimizerD.state_dict(),
                }

                #with open(os.path.join(train.savepath, 'dis_checkpoint.t7'), 'wb') as f:
                #    torch.save(dis_state, f)
            except Exception as e:
                printer(f"Error {str(e)}")

    #----------------------------------------------------------------------------
    ## End of Training Loop
    #----------------------------------------------------------------------------

    # Save final models
    torch.save(netG.state_dict(), os.path.join(train.savepath, 'netg_final.pth'))
    torch.save(netD.state_dict(), os.path.join(train.savepath, 'netd_final.pth'))

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

