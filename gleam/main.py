__author__ = "Andra Stroe"
__version__ = "0.1"

import os, sys
from contextlib import contextmanager

import numpy as np
import astropy
from astropy import units as u
from astropy.table import QTable, Table, Column
from astropy.io import fits
from colorama import Fore
from colorama import init

init(autoreset=True)

import gleam.read_files as rf
import gleam.gaussian_fitting as gf
import gleam.plot_gaussian as pg
import gleam.spectra_operations as so
import gleam.constants as const


def run_main(spectrum_file, target, inspect, plot, verbose, bin1, config_file):
    """
    For a target/galaxy, read the spectrum and perform the line fitting for each
    line within the list of lines.
    """
    config_factory = const.read_config(config_file)
    c = config_factory(
        sample=target["Sample"][0],
        setup_name=target["Setup"][0],
        pointing=target["Pointing"][0],
        source_number=target["SourceNumber"][0]
    )

    data_path = os.path.dirname(spectrum_file)
    print(
        f"Now working in {data_path} "
        + f'on {target["Sample"][0]} in {target["Setup"][0]} + {target["Pointing"][0]} '
        + f'on source {target["SourceNumber"][0]} at z={target["Redshift"][0]:1.3f}.'
    )

    spectrum = QTable.read(spectrum_file)

    try:
        name_map = {'wl': 'wl_obs', 'flux': 'flux_obs', 'stdev': 'stdev_obs'}
        for old, new in name_map.items():
            if old in spectrum.colnames: spectrum.rename_column(old, new)
    except Exception:
        print(Fore.RED + f"CRITICAL: Failed to rename columns for {spectrum_file}. Exiting.")
        sys.exit(1)

    spectrum = so.add_restframe(spectrum, target)
    
    line_list = QTable.read(c.line_list)
    sky_lines = QTable.read(c.sky_lines) if hasattr(c, 'sky_lines') and c.sky_lines else None
    use_sky_mask = hasattr(c.fitting, 'mask_sky') and c.fitting.mask_sky

    lines = so.select_lines_for_fitting(
        line_list, spectrum, target, c.fitting.cont_width,
        sky_lines if use_sky_mask else None,
    )

    if not lines:
        print(Fore.YELLOW + f'Warning: No valid emission lines found for source {target["SourceNumber"][0]}')
        return

    line_groups = so.group_lines(lines, tolerance=c.fitting.tolerance / 2.0)
    tables = []
    successful_fits = [] # List to store successful fits for the overview plot

    spectrum_binned = so.bin_spectrum(spectrum, bin1) if bin1 != 1 else None
    
    for lines_to_fit in line_groups.groups:
        center_constraint = bool(c.fitting.constraints and len(lines_to_fit) > 1)

        try:
            spectrum_fit, spectrum_line = gf.fit_lines(
                selected_lines=lines_to_fit,
                full_line_list=line_list,
                spectrum=spectrum,
                target=target,
                center_constraint=center_constraint,
                verbose=verbose, inspect=inspect,
                cont_width=c.fitting.cont_width,
                rest_spectral_resolution=c.instrument.rest_spectral_resolution,
                sky=sky_lines if use_sky_mask else None,
            )

            if spectrum_fit is not None:
                successful_fits.append(spectrum_fit) # Add to list for overview plot
                for line in spectrum_fit.lines:
                    tables.append(
                        Table(line.as_fits_table(), masked=True, copy=False)
                    )
                    if plot: # Only plot if the flag is set
                        pg.single_line_plot(
                            target=target, data_path=data_path,
                            spectrum=spectrum_line,
                            full_spectrum=spectrum_binned if spectrum_binned is not None else spectrum,
                            line_list=line_list, line_fit=spectrum_fit, line=line
                        )

        except Exception as e:
            print(f"ERROR during fitting: {e}")
            import traceback
            traceback.print_exc()
            
    # --- FIX: Generate the overview plot once at the end with all successful fits ---
    if plot and successful_fits:
        pg.overview_plot(
            target=target, data_path=data_path,
            spectrum=spectrum,
            successful_fits=successful_fits
        )

    if tables:
        try:
            outtable = astropy.table.vstack(tables)
            outtable = Table(outtable, masked=True, copy=False)
            outfile = f"{rf.naming_convention(data_path, target['Sample'][0], target['SourceNumber'][0], target['Setup'][0], target['Pointing'][0], 'linefits')}.fits"
            outtable.write(outfile, overwrite=True)
        except Exception as e:
            print(Fore.RED + f"Error: Could not write FITS file for source {target['SourceNumber'][0]}. Reason: {e}")
    else:
        print(Fore.YELLOW + f"Warning: no emission line fits were successful for source {target['SourceNumber'][0]} at z={target['Redshift'][0]:1.3f}.")