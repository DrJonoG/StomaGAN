# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author            Jonathon Gibbs
# Email             pszjg@nottingham.ac.uk
# Website           https://www.jonathongibbs.com
# Github            https://github.com/DrJonoG/
# StomataHub        https://www.stomatahub.com
#----------------------------------------------------------------------------

import os
import cv2
import json
import math
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from random import randrange, uniform, sample, randint, choice
from scipy import ndimage
from utils.helpers import printer, dotdict
from utils.image_helpers import resize, rotate
from torchvision.utils import save_image
from argparse import ArgumentParser
from models import Generator_
from PIL import Image

#----------------------------------------------------------------------------

def create_fake (
	config: object
) -> None:
    gpus = config.gpus.split(",")
	# set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs, state = torch.load(config.model_path)
    netG = Generator_(**kwargs).to(device)

    # If trained on multiple GPUs this must also be loaded into DataParallel
    if (device.type == 'cuda') and (len(gpus) > 1):
        netG = nn.DataParallel(netG, list(int(d) for d in gpus))

    #netG.load_state_dict(torch.load(config.model_path))
    netG.load_state_dict(state)

    printer("Generating fake images.... this may take some time.")
    for i in range(0, config.batches):
        printer(f"Generating batch {i} of {config.batches}", end="\r")
        # create random noise
        noise = torch.randn(config.qty, config.gen_input, 1, 1, device=device)
        fake = netG(noise).detach().cpu()
        
        for count, image in enumerate(fake):   
            printer(f"Processing image {count + 1} of {config.qty}", end="\r")
            filepath = os.path.join(config.destination, str(int(time.time()*1000.0))) + ".jpg"

            # Convert from 0..1 to 0..255
            imin = image.min()
            imax = image.max()

            a = 255 / (imax - imin)
            b = 255 - a * imax
            image = (a * image + b)

            # re-shape to numpy image from tensor image
            image = np.array(image.permute(1, 2, 0), dtype=np.uint8)

            h_, w_ = image.shape[:2]
            # Apply border to form gap between edge preventing entire image being categorised into a single contour
            image = cv2.copyMakeBorder(image,20,20,20,20,cv2.BORDER_CONSTANT,value=(255))

            # Apply blurring to gray image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(18,18))
            res = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

            # get color bounds of white background
            lower = (180,180,180) # lower bound for each channel
            upper = (255,255,255) # upper bound for each channel

            # create the mask
            mask = cv2.inRange(res, lower, upper)
            mask = cv2.bitwise_not(mask)

            # get the largest contour
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            # Select largest contour
            big_contour = max(contours, key=cv2.contourArea)

            # Draw contours and bounding boxes
            if config.draw_contours:
                cv2.drawContours(image, [big_contour], 0, (0,255,0), 3)
                cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (0, 0, 255), 1)
            
            mask_value = 255
            stencil  = np.zeros(image.shape[:2]).astype(np.uint8)
            cv2.fillPoly(stencil, [big_contour], mask_value)

            # Select everything that is not mask_value and fill it
            sel = stencil != mask_value 
            image[sel] = [255]

            # Crop out the feature of interest to save as a new image
            x,y,w,h = cv2.boundingRect(big_contour)
            image = image[y:y+h, x:x+w]

            # Apply CLAHE
            grid = max((min((int(w_*0.01),int(h_*0.01))), 8))
            clahe = cv2.createCLAHE(clipLimit=10, tileGridSize= (grid,grid))
            image = clahe.apply(image)

            cv2.imwrite(filepath, image)

#----------------------------------------------------------------------------

def random_coordinates(
    width: int, 
    height: int, 
    boundary: int, 
    existing_coordinates, 
    image
) -> (int, int, int, int):
    """Generate random x, y coordinates within the given dimensions,
    making sure they do not overlap with any existing coordinates."""
    img_height, img_width  = image.shape[:2]
    x, y = randint(boundary, width-(img_width+boundary)), randint(boundary, height-(img_height+boundary))

    error_counter = 0

    while any(x <= x2+w and x+img_width >= x2 and y <= y2+h and y+img_height >= y2 for x2, y2, w, h in existing_coordinates):
        if error_counter > 50: return (0,0,0,0)
        x, y = randint(boundary, width-(img_width+boundary)), randint(boundary, height-(img_height+boundary))
        error_counter += 1
    return (x, y, img_width, img_height)

#----------------------------------------------------------------------------

def blur (
    image: np.ndarray,
    blur_type: str = "mean",
    fsize: int = 9
) -> np.ndarray:
    fsize = choice([3,5,9])
    if blur_type == "mean":
        return cv2.blur(image,(fsize,fsize))
    elif blur_type == "gaussian":
        return cv2.GaussianBlur(image, (fsize, fsize), 0)
    elif blur_type == "median":
        return cv2.medianBlur(image, fsize)
    else:
        return image

#----------------------------------------------------------------------------

def sharpen (
    image: np.ndarray
) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    return cv2.filter2D(image, -1, kernel) 

#----------------------------------------------------------------------------

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

#----------------------------------------------------------------------------    

def generate_whole(
    config: object,
) -> None:
    file_list = []
    background_list = []
    image_extensions = {".jpg",".jpeg",".png",".tif"}



    # Get list of all samples
    for filename in os.listdir(config.source):
        if any(filename.lower().endswith(ext) for ext in image_extensions): # Verify the file is an image
            file_list.append(os.path.join(config.source, filename))

    # Get list of all samples
    if config.background_source:
        for filename in os.listdir(config.background_source):
            if any(filename.lower().endswith(ext) for ext in image_extensions): # Verify the file is an image
                background_list.append(os.path.join(config.background_source, filename))

    train_images = int((1 - config.test_split) * config.image_qty)
    test_images = int(config.image_qty - train_images)

    # Create images
    for i in range(0, config.image_qty):
        # Empty images
        img = np.full((config.size,config.size, 3), 255)
        img_mask = np.full((config.size,config.size), 255)

        # To store existing coordinates to prevent overlapping
        coordinates = []

        # Select random samples for image creation
        samples = randint(config.min_samples,config.max_samples)
        image_samples = sample(file_list, samples)


        printer(f"Generating image {i} of {config.img_qty} processing sample {samples} of {samples}.. Complete.. Performing post processing...", end="\r")

        # Image creation
        if len(background_list) > 0:

            image_temp = background_list[randint(0, len(background_list)-1)]
            image_temp = cv2.imread(image_temp, cv2.COLOR_BGR2GRAY)

            j = int(config.size / image_temp.shape[0])
            k = int(config.size / image_temp.shape[1])

            for img_j in range(0, j):
                for img_k in range(0, k):
                    random_image = background_list[randint(0, len(background_list)-1)]
                    random_image = cv2.imread(random_image, cv2.COLOR_BGR2GRAY)

                    img[int(img_j * image_temp.shape[0]):int((img_j + 1) * image_temp.shape[0]), int(img_k * image_temp.shape[1]):int((img_k + 1) * image_temp.shape[1])] = random_image
        elif config.noise:
            img_h,img_w = img.shape[:2]
            noise = np.random.randint(50, 205, (img_w, img_h))
            img = np.where(img > 250, noise, img)

        img = img.astype(np.uint8)
        #img = resize(img, config.size)


        printer(f"Generating image {i+1} of {config.img_qty} processing {samples} samples", end="\r")
        for count, sample_img in enumerate(image_samples):
            sample_img = cv2.imread(sample_img, cv2.COLOR_BGR2GRAY)

            # Sample augmentations            
            scaling_factor = round(uniform(config.min_scale, config.max_scale), 2)            
            sample_img = cv2.resize(sample_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            sample_img = rotate(sample_img, randint(0, 359))
    
            if randint(0, 10) == 1:
                k = choice([3, 5])
                k = 3
                sample_img = cv2.blur(sample_img, (k,k))
            elif randint(0, 10) == 1:
                k = choice([3, 5])
                k = 3
                sample_img = cv2.GaussianBlur(sample_img, (k, k), 0)
            
            if randint(0, 8) == 1:
                sample_img = adjust_gamma(sample_img, uniform(0.5, 0.9)) 

            h, w = sample_img.shape[:2]

            (x, y, w, h) = random_coordinates(config.size, config.size, config.boundary, coordinates, sample_img)
            if w == 0 or h == 0: continue
            coordinates.append((x, y, w, h))

            # Place sample on image
            for x_count, x_iter in enumerate(range(x, x+w)):
                for y_count, y_iter in enumerate(range(y, y+h)):
                    if sample_img[y_count, x_count] > 245: continue                    
                    img[y_iter, x_iter] = sample_img[y_count, x_count]
            img_mask[y:y+h, x:x+w] = sample_img

        # Mask creation
        img_mask = np.where(img_mask < 220, 120, 0)

        # Close holes in mask
        kernel = np.ones((9,9), np.uint8)
        img_mask = cv2.morphologyEx(img_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel)

        # Invert 
        img_mask = np.invert(img_mask)
        img_mask = resize(img_mask, config.size)

        # Soft smoothing
        if randint(0, 30) == 1:
            img = cv2.blur(img, (3,3))

        if randint(0, 15) == 1:
            img = adjust_gamma(img, uniform(0.7, 0.9)) 

        # Apply clahe
        h,w = img.shape[:2]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        grid = max((min((int(w*0.01),int(h*0.01))), 8))
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize= (grid,grid))

        img = clahe.apply(img)

        # Mask creation
        img_mask = np.where(img_mask > 200, 255, 120)
  
        if i < test_images:     
            file = str(int(time.time()*1000.0))

            valid_mask_path = os.path.join(config.destination, "valid_mask", file)
            valid_samples_path = os.path.join(config.destination, "valid_samples", file) 

            cv2.imwrite(valid_samples_path + ".jpg", img)
            cv2.imwrite(valid_mask_path + "_mask.jpg", img_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        else:
            file = str(int(time.time()*1000.0))

            train_mask_path = os.path.join(config.destination, "train_mask", file)
            train_samples_path = os.path.join(config.destination, "train_samples", file) 

            cv2.imwrite(train_samples_path + ".jpg", img)
            cv2.imwrite(train_mask_path + "_mask.jpg", img_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-config",  type=str,   help="Path to config",  		default="./generate.json")
    parser.add_argument("-func",  type=str,   help="Path to config",          choices=["samples","images"])

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

    if args.func == "samples":
        samples = dotdict(config_raw['samples'])

        if not os.path.isfile(samples.model_path):
            printer(f"Error -> The model file {samples.model_path} cannot be found.")
            parser_error = True

        if not os.path.exists(samples.destination):
            try:
                os.makedirs(samples.destination)
            except Exception as e:
                printer(f"Error creating folder {samples.destination}.")
                print(str(e))
                parser_error = True

        if samples.qty < 1:
            printer(f"Error -> Quantity to generate should be greater than 0, not {samples.qty}")
            parser_error = True

        if samples.batches < 1:
            printer(f"Error -> Batches to generate should be greater than 0, not {samples.batches}")
            parser_error = True

        if samples.channels < 0:
            printer(f"Error -> Image channels should be greater than 0, not {samples.channels}")
            parser_error = True
        
        if parser_error:
            exit(0)

        create_fake(samples)
        #pre_process_fake(config)
    elif args.func == "images":
        config = dotdict(config_raw['images'])

        if not os.path.exists(config.source):
            printer(f"Error -> The source folder {config.source} cannot be found.")
            parser_error = True

        if not os.path.exists(config.destination):
            try:
                os.makedirs(config.destination)
            except Exception as e:
                printer(f"Error creating folder {config.destination}.")
                print(str(e))
                parser_error = True

        if config.image_qty < 1:
            printer(f"Error -> Quantity to generate should be greater than 0, not {config.image_qty}")
            parser_error = True

        if config.min_samples < 1:
            printer(f"Error -> Min samples should be greater than 0, not {config.min_samples}")
            parser_error = True

        if config.max_samples > 10000:
            printer(f"Error -> Max samples should not be greater than 10000, not {config.max_samples}")
            parser_error = True

        if config.max_samples < config.min_samples:
            printer(f"Error -> Max samples should be greater than min samples")
            parser_error = True

        if config.size < 16:
            printer(f"Error -> Size should be greater than 16, not {config.size}")
            parser_error = True

        if config.noise not in [0, 1]:
            printer(f"Error -> Noise should be 0 (False) or 1 (True), not {config.noise}")
            parser_error = True

        if config.boundary < 0:
            printer(f"Error -> Boundary should be a positive integer, not {config.boundary}")
            parser_error = True

        if config.min_scale < 0.1:
            printer(f"Error -> min_scale should be greater than 0.1, not {config.min_scale}")
            parser_error = True

        if config.max_scale < 0.1:
            printer(f"Error -> max_scale should be greater than 0.1, not {config.max_scale}")
            parser_error = True

        if config.max_scale < config.min_scale:
            printer(f"Error -> max_scale should be greater min_scale")
            parser_error = True

        if config.draw_contours not in [0, 1]:
            printer(f"Error -> draw_contours should be 0 (False) or 1 (True), not {config.draw_contours}")
            parser_error = True

        if parser_error:
            exit(0)

        # Create folders for data split
        if not os.path.exists(os.path.join(config.destination, "valid_mask")):
            os.mkdir(os.path.join(config.destination, "valid_mask"))

        if not os.path.exists(os.path.join(config.destination, "valid_samples")):
            os.mkdir(os.path.join(config.destination, "valid_samples"))

        if not os.path.exists(os.path.join(config.destination, "train_mask")):
            os.mkdir(os.path.join(config.destination, "train_mask"))

        if not os.path.exists(os.path.join(config.destination, "train_samples")):
            os.mkdir(os.path.join(config.destination, "train_samples"))

        generate_whole(config)
   