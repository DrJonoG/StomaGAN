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