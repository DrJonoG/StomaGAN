# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author        Jonathon Gibbs
# Email         pszjg@nottingham.ac.uk
# Website       https://www.jonathongibbs.com
# Github        https://github.com/DrJonoG/
# StomataHub    https://www.stomatahub.com
#----------------------------------------------------------------------------

import os
import shutil
import datetime
from colorama import init, Fore

#----------------------------------------------------------------------------

def printer (
    text: str,                      # String to print to console
    end: str = "\n",                # End delimeter options ["\r","\n",""]
    color: str = "WHITE"     # Specify text color
) -> None:
    now = datetime.datetime.now()

    color = Fore.__getattribute__(color)

    # Replace error keyword with red
    text = text.replace("Error",f"{Fore.RED}Error{color}")

    if end == "\r":
        print(f"\033[2K{color}{now:%Y-%m-%d %H:%M:%S} ==> {text}", end=end)
    else:
        print(f"{color}{now:%Y-%m-%d %H:%M:%S} ==> {text}", end=end)


#----------------------------------------------------------------------------

def printer_config (
    param: str,                      # Parameter string
    value: str,                      # Value string
) -> None:
    now = datetime.datetime.now()

    value = str(value)

    param_len = len(param)
    value_len = len(value)

    print(f"{param}{' ' * (15-param_len)}{value}", end="\n")

#----------------------------------------------------------------------------

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#----------------------------------------------------------------------------

def print_progress_bar (
    iteration: int, 
    total: int, 
    prefix: str = '', 
    suffix: str = '', 
    decimals: int = 1, 
    length: int = 100, 
    fill: str = '█', 
    end: str = "\r"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        end         - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = end)
    # Print New Line on Complete
    if iteration == total: 
        print()

#----------------------------------------------------------------------------

def parse_int (
    parameter: str,
    value: any, 
    min_val: int = 0, 
    max_val: int = 1000000
) -> bool:
    try: 
        value = int(value)
        if value < min_val or value > max_val: 
            raise TypeError()
    except Exception as e:
        printer(f"Error unable to parse {parameter}, please enter a positive integer between {min_val}-{max_val}.")
        return True
