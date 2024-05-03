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
import math
import copy
import time
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

from models import Generator_, Discriminator
from utils.helpers import dotdict, printer, printer_config, print_progress_bar

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from argparse import ArgumentParser

#----------------------------------------------------------------------------

def get_real_labels(batch_size, device):
    labels = np.random.uniform(0.9, 1.0, size=(batch_size,))  
    return torch.from_numpy(labels).float().to(device)

#----------------------------------------------------------------------------

def get_fake_labels(batch_size, device):
    labels = np.random.uniform(0.0, 0.1, size=(batch_size,))   
    return torch.from_numpy(labels).float().to(device)

#----------------------------------------------------------------------------

def get_top_k(output, temperature, batch_size, warmup):
    if warmup:
        return batch_size
    else:
        mean = torch.mean(output)
        base_batch_size = batch_size * temperature
        top_k = base_batch_size + (mean * (batch_size - base_batch_size))
        return int(math.ceil(top_k))

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

class Train:
    def __init__(
        self,
        train: type[dotdict], 
        gen: type[dotdict], 
        dis: type[dotdict], 
        general: type[dotdict],
        experiment: object
    ) -> None:
        #----------------------------------------------------------------------------
        ## Dataset intialisation
        #----------------------------------------------------------------------------

        folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        train.savepath = os.path.join(train.savepath, folder)

        if not os.path.exists(train.savepath):
            os.mkdir(train.savepath)

        train.epochpath = os.path.join(train.savepath, "epochs")
        if not os.path.exists(train.epochpath):
            os.mkdir(train.epochpath)
                
        printer(f"Training data saved to {train.savepath}")

        random.seed(train.seed)
        torch.manual_seed(train.seed)
        #torch.use_deterministic_algorithms(True) # Needed for reproducible results

        printer("Creating dataset", end="")
        # Create the dataset
        dataset = dset.ImageFolder(
            root=train.dataroot,
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
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
        self.dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=train.batch,
            shuffle=True, 
            num_workers=train.workers
        )
        print(f".... batch size {train.batch} .... {train.workers} workers")

        # Specify GPUs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        


        #----------------------------------------------------------------------------
        ## Model intialisation 
        #----------------------------------------------------------------------------

        self.netG = Generator_(train.gen_input, train.gen_features, train.channels, train.resize[0]).to(self.device)

        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (len(general.gpus) > 1):
            self.netG = nn.DataParallel(self.netG, list(int(d) for d in general.gpus))

        # randomly initialize all weights to mean=0, stdev=0.02.
        printer("Initialising random weights for Generator model")
        self.netG.apply(weights_init)
        print(self.netG) 
     
        # Create the Discriminator
        self.netD = Discriminator(train.dis_features, train.channels, train.resize[0]).to(self.device)

        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (len(general.gpus) > 1):
            self.netD = nn.DataParallel(self.netD, list(int(d) for d in general.gpus))

        # randomly initialize all weights to mean=0, stdev=0.02.
        printer("Initialising random weights for Discriminator model")
        self.netD.apply(weights_init)
        print(self.netD)  

        #----------------------------------------------------------------------------
        ## Loss and Optimizer initialisation
        #----------------------------------------------------------------------------

        # Initialize the ``BCELoss`` function
        self.criterionD = nn.BCELoss()
        self.criterionG = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        self.fixed_noise = torch.randn(64, train.gen_input, 1, 1, device=self.device)
        torch.save(self.fixed_noise, os.path.join(train.savepath, 'fixed_noise.pt'))

        # Initialise Adam optimisers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=dis.lr, betas=(dis.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=gen.lr, betas=(gen.beta1, 0.999))

        # Initiailise variables
        self.num_gpus = len(general.gpus)
        self.gen_input = train.gen_input
        self.savepath = train.savepath
        self.epochs = train.epochs
        self.warmup = train.warmup_epochs
        self.min_temp = train.min_temp
        self.temperature = train.temperature
        self.total_steps = self.epochs * len(self.dataloader)
        self.temp_step = (self.temperature - self.min_temp) / (self.total_steps * 0.5)
        self.fid_score = 9999
        self.d_loss = 9999
        self.g_loss = 9999

        self.experiment = experiment
        # Run with experiement if true
        if self.experiment:
            self.experiment.set_name(folder)

            # Log training images
            real_batch = next(iter(self.dataloader))
            real_grid = vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True)
            real_grid = transforms.ToPILImage()(real_grid) 
            self.experiment.log_image(real_grid, name="Real images")

            self.experiment.log_model('netG', os.path.join(train.savepath,'netG_architecture.pt'))
            self.experiment.log_model('netD', os.path.join(train.savepath,'netD_architecture.pt'))

            with self.experiment.train():

                watch(self.netG)
                watch(self.netD)

                self.training_loop()

            self.experiment.end()
        else:
            self.training_loop()

#----------------------------------------------------------------------------

    def get_state_dict(self):
        # Module is only used with multi GPU 
        if self.num_gpus > 1:
            netG_state_dict = self.netG.module.state_dict()
            netD_state_dict = self.netD.module.state_dict()
        else:
            netG_state_dict = self.netG.state_dict()
            netD_state_dict = self.netD.state_dict()

        return (netG_state_dict, netD_state_dict)

#----------------------------------------------------------------------------

    def save_state(self, epoch, filename="checkpoint"): 
        netG_state_dict, netD_state_dict  = self.get_state_dict()

        # Save current state for resumption        
        try:
            # Save state to allow resumption if failure
            gen_state = {
                'epoch': epoch,
                'state_dict': netG_state_dict,
                'optimizer': self.optimizerG.state_dict(),
            }
            with open(os.path.join(self.savepath, 'gen_' + str(filename) + '.t7'), 'wb') as f:
                torch.save([self.netG.kwargs, gen_state], f)

            dis_state = {
                'epoch': epoch,
                'state_dict': netD_state_dict,
                'optimizer': self.optimizerD.state_dict(),
            }
            with open(os.path.join(self.savepath, 'dis_' + str(filename) + '.t7'), 'wb') as f:
                torch.save([self.netD.kwargs, dis_state], f)
        except Exception as e:
            printer(f"Error {str(e)}")

#----------------------------------------------------------------------------

    def epoch_eval(self, epoch, runtime):
        if (epoch % 10) == 0:                
            # Get and expand images
            real1, _ = next(iter(self.dataloader))
            real1 = torch.repeat_interleave(real1, 3, dim=1)

            real2, _ = next(iter(self.dataloader))
            real2 = torch.repeat_interleave(real2, 3, dim=1)

            real = torch.cat((real1, real2)) 
            feature_num = max(real.shape[2], real.shape[3])
            feature_num = 64

            # Generate fake image
            with torch.no_grad():
                fake_test = self.netG(self.fixed_noise).detach().cpu()

            fake_test = torch.repeat_interleave(fake_test, 3, dim=1)
            fake_test = fake_test[0:real.shape[0]]

            # https://pytorch.org/torcheval/main/generated/torcheval.metrics.FrechetInceptionDistance.html
            # https://github.com/mseitzer/pytorch-fid
            fid = FrechetInceptionDistance(feature=feature_num, normalize=True)
            fid.update(real, real=True)
            fid.update(fake_test, real=False)
            fid_score2048 = fid.compute()

            if fid_score2048 < self.fid_score:
                netG_state_dict, netD_state_dict  = self.get_state_dict()
                torch.save([self.netD.kwargs, netD_state_dict], os.path.join(self.savepath, 'lowest_fid_loss_D.pth'))
                torch.save([self.netG.kwargs, netG_state_dict], os.path.join(self.savepath, 'lowest_fid_loss_G.pth'))

                self.fid_score = fid_score2048

        epoch_G_losses = np.average(self.epoch_G_losses)
        epoch_D_losses = np.average(self.epoch_D_losses)
        epoch_D_G_z1 = np.average(self.epoch_D_G_z1)
        epoch_D_G_z2 = np.average(self.epoch_D_G_z2)
        epoch_D_x = np.average(self.epoch_D_x)
        epoch_top_k = np.average(self.epoch_top_k)

        if epoch_D_losses < self.d_loss:
            netG_state_dict, netD_state_dict  = self.get_state_dict()
            torch.save([self.netD.kwargs, netD_state_dict], os.path.join(self.savepath, 'lowest_d_loss_D.pth'))
            torch.save([self.netG.kwargs, netG_state_dict], os.path.join(self.savepath, 'lowest_d_loss_G.pth'))

            self.d_loss = epoch_D_losses

        if epoch_G_losses < self.g_loss:
            netG_state_dict, netD_state_dict  = self.get_state_dict()
            torch.save([self.netD.kwargs, netD_state_dict], os.path.join(self.savepath, 'lowest_g_loss_D.pth'))
            torch.save([self.netG.kwargs, netG_state_dict], os.path.join(self.savepath, 'lowest_g_loss_G.pth'))

            self.g_loss = epoch_G_losses    

        if (epoch % 20) == 0:
            netG_state_dict, netD_state_dict  = self.get_state_dict()
            torch.save([self.netD.kwargs, netD_state_dict], os.path.join(self.savepath, ('D_epcoh_' + str(epoch) +'.pth')))
            torch.save([self.netG.kwargs, netG_state_dict], os.path.join(self.savepath, ('G_epcoh_' + str(epoch) +'.pth')))   

        if self.experiment:
            epoch_logger = {
                "Epoch_Gen_Loss": epoch_G_losses, 
                "Epoch_Dis_Loss": epoch_D_losses,
                "Epoch_D_G_z1": epoch_D_G_z1,  
                "Epoch_D_G_z2": epoch_D_G_z2,
                "Epoch_D_x": epoch_D_x,
                "Epoch_Top_K": epoch_top_k,
                "FID_2048": self.fid_score,
                "Epoch_Runtime": runtime,
            }
            self.experiment.log_metrics(epoch_logger, epoch=epoch)

            if (epoch % 10) == 0:
                with torch.no_grad():
                    fake_test = self.netG(self.fixed_noise).detach().cpu()
                fake_grid = vutils.make_grid(fake_test, padding=2, normalize=True)
                fake_grid = transforms.ToPILImage()(fake_grid)  
                self.experiment.log_image(fake_grid, name=f"Fake_Images")   

        # Output training stats
        print_progress_bar(
                len(self.dataloader), 
                len(self.dataloader), 
                prefix = f"Epoch {epoch+1} of {self.epochs}. Progress:", 
                suffix = f"Complete. Loss_D: {epoch_D_losses:.4f}, Loss_G: {epoch_G_losses:.4f}, D(x): {epoch_D_x:.4f}, D(G(z)): {epoch_D_G_z1:.4f} / {epoch_D_G_z2:.4f}, FID_2048: {self.fid_score:.4f}",
                decimals = 1,
                length = 30,
                end = "\r\n\n"
        )

#----------------------------------------------------------------------------

    def training_loop (self):
        #--------------------------------------------------------------------
        ## Training Loop
        #--------------------------------------------------------------------
        self.netG.train()
        self.netD.train()
        step = 0
        printer("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.epochs):
            # Epoch time
            epoch_timer = time.time()
            # Store for plotting
            self.epoch_G_losses = []
            self.epoch_D_losses = []
            self.epoch_D_G_z1 = []
            self.epoch_D_G_z2 = []
            self.epoch_D_x = []
            self.epoch_top_k = []         

            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                self.netD.zero_grad()
                self.netG.zero_grad()
                self.optimizerD.zero_grad()
                self.optimizerG.zero_grad()

                # Adjust the top elements to select
                if epoch > self.warmup:                    
                    self.temperature = (self.temperature - self.temp_step if self.temperature > self.min_temp else self.min_temp)

                #--------------------------------------------------------------------
                ## (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                #--------------------------------------------------------------------

                ## 1.a --------- Train with all-real batch
                # Format batch
                real_images = data[0].to(self.device)
                batch_size = real_images.size(0)

                # Forward pass real batch through D
                real_output = self.netD(real_images).view(-1)
                real_labels = get_real_labels(batch_size, self.device)
                # Calculate loss on all-real batch
                loss_D_real = self.criterionD(real_output, real_labels)
                # Calculate gradients for D in backward pass
                loss_D_real.backward()
                D_x = real_output.mean().item()


                ## 1.b --------- Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, self.gen_input, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake_images = self.netG(noise)          
                # Classify all fake batch with D and Get labels
                fake_output = self.netD(fake_images.detach()).view(-1)                   
                # Calculate top_k based on (norm(x) - (1-T))*B             
                top_k = get_top_k(fake_output, self.temperature, batch_size, (epoch < self.warmup))   
                # Filter            
                fake_output, _ = torch.topk(fake_output, top_k)
                fake_labels = get_fake_labels(top_k, self.device)
                # Calculate D's loss on the all-fake batch
                loss_D_fake = self.criterionD(fake_output, fake_labels)
                # Calculate the gradients for this batch,
                loss_D_fake.backward()                
                D_G_z1 = fake_output.mean().item()
                # Update D
                self.optimizerD.step()
                #Compute error of D as sum over the fake and the real batches
                loss_D = (loss_D_real + loss_D_fake)

                self.epoch_D_losses.append(loss_D.item())
                self.epoch_D_G_z1.append(D_G_z1)
                self.epoch_D_x.append(D_x)

                #--------------------------------------------------------------------
                ## (2) Update G network: maximize log(D(G(z)))           
                #--------------------------------------------------------------------        

                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake_images).view(-1)
                # Calculate top_k based on (norm(x) - (1-T))*B
                top_k = get_top_k(output, self.temperature, batch_size, (epoch < self.warmup))
                # Filter 
                output, _ = torch.topk(output, top_k)
                labels = get_real_labels(top_k, self.device)
                # Calculate G's loss based on this output
                loss_G = self.criterionG(output, labels)                                
                # Calculate gradients for G
                loss_G.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                self.epoch_top_k.append(top_k / batch_size)
                self.epoch_G_losses.append(loss_G.item())
                self.epoch_D_G_z2.append(D_G_z2)

                #--------------------------------------------------------------------
                ## (3) Output training stats:
                #--------------------------------------------------------------------


                # Flare
                if self.temperature < 0.5 and random.uniform(0,1) > 0.9:
                    self.temperature += 0.3

                print_progress_bar(
                    i+1, 
                    len(self.dataloader), 
                    prefix = f"Epoch {epoch+1} of {self.epochs}. Progress:", 
                    suffix = f"Complete. Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}, Top_K: {top_k} / {batch_size} ({(top_k / batch_size):.4f}), Temp: {self.temperature:.5f}",
                    decimals = 1,
                    length = 30,
                )

                if self.experiment:
                    # Log
                    step_logger = {
                        "Gen_Loss": loss_G.item(),
                        "Dis_Loss": loss_D.item(),
                        "D_G_z1": D_G_z1,
                        "D_G_z2": D_G_z2,
                        "Avg. Output": D_x,
                        "Top_K": (top_k / batch_size)
                    }
                    self.experiment.log_metrics(step_logger, step=step)

                step += 1

            # Run time
            elapsed_time = time.time() - epoch_timer

            #------------------------------------------------------------------------
            ## Log data (end of epoch)
            #------------------------------------------------------------------------            
            self.epoch_eval(epoch, elapsed_time)

            

        #----------------------------------------------------------------------------
        ## End of Training Loop
        #----------------------------------------------------------------------------

        # Save final models
        netG_state_dict, netD_state_dict  = self.get_state_dict()
        torch.save([self.netG.kwargs, netG_state_dict], os.path.join(self.savepath, 'netg_final.pth'))
        torch.save([self.netD.kwargs, netD_state_dict], os.path.join(self.savepath, 'netd_final.pth'))

#----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-config",  type=str,   help="Path to config",  default="./config.json")

    args = parser.parse_args()

    #----------------------------------------------------------------------------

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

    #----------------------------------------------------------------------------

    # Covert accesibility to dot
    config          = dotdict(config_raw)
    config.comet_dl = dotdict(config.comet_dl)
    config.train    = dotdict(config.train)
    config.general  = dotdict(config.general)
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

    if not config.train.resize:
        config.train.resize = [256, 256]
    else:
        try:
            resize = config.train.resize.split("x")
            if len(resize) == 1:
                config.train.resize = [int(resize[0]), int(resize[0])]
            else:
                config.train.resize = [int(resize[0]), int(resize[1])]

            if not (config.train.resize[0] > 0 and (config.train.resize[0] & (config.train.resize[0]-1) == 0)):
                raise TypeError()

            printer_config("resize",f"{config.train.resize[0]}x{config.train.resize[1]}")

        except Exception as e:
            printer("Error processing resize. Please ensure format is \"widthxheight\" and a power of 2")
            config_errors = True

    #----------------------------------------------------------------------------

    if config.general.gpus:
        try: 
            config.general.gpus = config.general.gpus.split(",")
            
            printer_config("gpus", config.general.gpus)

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

            printer_config("gen_features", config.train.gen_features)
            
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

    if config.comet_dl.log:
        try: 
            config.comet_dl.log = int(config.comet_dl.log)   

            if config.comet_dl.log == 1:
                if (not os.path.isfile(config.comet_dl.path)):
                    printer("Error invalid comet path specified in config file.") 
                    config_errors = True
                else:
                    with open(config.comet_dl.path) as comet_file:
                        try:
                            comet = json.load(comet_file)
                            config.comet = dotdict(comet)
                        except Exception as e:
                            printer(f"Error reading configuration file {config.comet_dl.path}")
                            printer(str(e))
                            config_errors = True
        except Exception as e:
            printer("Error unable to parse comet_dl.log, please enter integer 1 (True) or 0 (False).")
            printer(str(e))
            config_errors = True


    
    #----------------------------------------------------------------------------

    if config_errors:
        exit(0)
    else: 
        if config.comet_dl.log:
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
                "discriminator": config_raw['discriminator'],
                "general": config_raw['general']
            }

            experiment.log_parameters(parameters)
        else:
            experiment = None

        # Train
        Train(config.train, config.gen, config.dis, config.general, experiment)
