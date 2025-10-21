__author__ = "Andra Stroe"
__version__ = "0.1"

import glob
import sys
import os

import numpy as np
from astropy.io import fits
from astropy.table import QTable, Column
from astropy import units as u
from colorama import Fore

import gleam.constants as c


def read_lof(file1):
    """For each sample, telescope setup and pointing, it reads the metadata file, 
    which contains a list of the sources and their properties.
    Input:
        file1: metadata file in ascii or fits format
        The format of the head file is the following
        # Setup Pointing SourceNumber Sample Redshift
    Return: 
        Astropy Table with measurements of interest: the source number, the 
        parent sample, the telescope/instrument and pointing and the redshift. 
        Throws error if the file is not of the right type.
    """
    try:
        table = QTable.read(file1, format="fits")
        return table
    except:
        try:
            table = QTable.read(file1, format="ascii.commented_header")
            return table
        except:
            print(Fore.RED + "Cannot find metadata redshift file")
            sys.exit("Error!")


def naming_convention(data_path, sample, source_number, setup, pointing, mod):
    """
    Naming convention for files which starts with type of file and is followed
    by details about the source and setup, in order: parent sample, setup, 
    pointing and source number. 
    """
    return (
        f"{data_path}/{mod}.{sample}.{setup}.{pointing}.{source_number.astype(int):03d}"
    )

# --- FIX: New function to parse filenames instead of reading FITS headers ---
def create_key_from_filename(filepath):
    """
    Creates a unique key by parsing the spectrum's filename.
    Assumes a filename structure like: 'spec1d.SAMPLE.SETUP.POINTING.SRCNUM.fits'
    """
    basename = os.path.basename(filepath)
    parts = basename.split('.')
    
    # Expecting ['spec1d', 'SMACS', 'JWST_NIRSpec', 'SMACS0723', '9922', 'fits']
    if len(parts) < 5:
        raise ValueError(f"Filename '{basename}' does not have the expected structure.")

    sample = parts[1]
    setup = parts[2]
    pointing = parts[3]
    source_number = parts[4]
    
    return f"{sample}{setup}{pointing}{source_number}"