__author__ = "Andra Stroe"
__version__ = "0.1"

import os
import sys
import warnings
import glob
from multiprocessing import Pool
from functools import reduce
from typing import Tuple
import traceback

import numpy as np
from astropy.io import fits
from astropy.table import vstack, QTable, Table
import click
from colorama import Fore

import gleam.main
import gleam.read_files as rf
import gleam.constants as c

warnings.filterwarnings("ignore")


class Targets:
    """
    Read and stack all metadata files found into a single table.
    """
    def __init__(self, filter: str) -> None:
        find_meta = glob.glob(filter, recursive=True)
        if not find_meta:
            sys.exit(Fore.RED + "Error! Cannot find any metadata files.")
        try:
            targets = vstack(
                [rf.read_lof(list_of_targets) for list_of_targets in find_meta]
            )
            
            for col in targets.colnames:
                if targets[col].dtype.kind in ['S', 'a', 'U']:
                    targets[col] = targets[col].astype('U')

            key_cols = [
                targets["Sample"],
                targets["Setup"],
                targets["Pointing"],
                targets["SourceNumber"].astype(str)
            ]
            targets["key"] = reduce(np.char.add, key_cols)

        except Exception as e:
            sys.exit(Fore.RED + f"Error! Cannot read metadata files. Reason: {e}")
        self.targets = targets


def find_source_properties(spectra, targets):
    """
    For a given spectrum, find its corresponding metadata by parsing its filename.
    """
    for spectrum_filepath in spectra:
        try:
            # --- FIX: Generate the key from the filename, not the header ---
            target_key = rf.create_key_from_filename(spectrum_filepath)
            
            # Find the corresponding row in the metadata table
            target_row = targets.targets[targets.targets["key"] == target_key]
            
            if len(target_row) == 0:
                print(Fore.YELLOW + f"Warning: No metadata found for key '{target_key}' from file {os.path.basename(spectrum_filepath)}")
                continue

            yield spectrum_filepath, target_row

        except Exception as e:
            print(Fore.YELLOW + f"Warning: Skipping {os.path.basename(spectrum_filepath)}. Could not match with metadata. Reason: {e}")


def run_main_safely(*args):
    """
    A wrapper for the main fitting function to catch any exceptions.
    """
    filename = os.path.basename(args[0])
    try:
        gleam.main.run_main(*args)
    except Exception as e:
        print(
            Fore.YELLOW
            + f"--> WARNING: Skipping {filename}. Reason: {e}"
        )
        # Uncomment the line below for a full error report
        # print(Fore.RED + traceback.format_exc())

@click.command()
@click.option("--path", default=".", help="Path to the data.")
@click.option("--spectra", default=None, help='Fits file with the 1D spectrum, can contain wildcards.')
@click.option("--config", default="gleamconfig.yaml", help='Configuration file in YAML format.')
@click.option("--plot", is_flag=True, help='Save plots of spectrum with emission lines fits next to the corresponding spectrum file.')
@click.option("--inspect", is_flag=True, help='Show interactive plots.')
@click.option("--verbose", is_flag=True, help='Print full output from LMFIT.')
@click.option("--bin", default=1, help='Bin the spectrum before fitting.')
@click.option("--nproc", default=8, type=int, help='Number of threads.')
def pipeline(path, spectra, config, plot, inspect, verbose, bin, nproc):
    """
    Main pipeline function that orchestrates the data processing.
    """

    print("--- Matching FITS files to metadata by filename ---")
    targets = Targets(f"{path}/**/meta.*")
    if spectra is None:
        spectra = f"{path}/**/spec1d*fits"
    find_spectra = sorted(glob.glob(f"{spectra}", recursive=True))
    
    # Generate the list of sources to process
    sources_to_process = list(find_source_properties(find_spectra, targets))
    
    if not sources_to_process:
        print(Fore.RED + "CRITICAL: No valid sources found after matching metadata. Check filenames or metadata files.")
        return
        
    print(f"--- Found {len(sources_to_process)} valid sources to process ---")

    unique_sources = (
        (*source_tuple, inspect, plot, verbose, bin, config)
        for source_tuple in sources_to_process
    )

    print("\n--- RUNNING IN FORGIVING PARALLEL MODE ---\n")
    if nproc > 1:
        with Pool(processes=nproc) as pool:
            pool.starmap(run_main_safely, unique_sources)
    else:
        # Sequential mode for debugging
        for source_args in unique_sources:
            run_main_safely(*source_args)