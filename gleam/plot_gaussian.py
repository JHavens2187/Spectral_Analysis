__author__ = "Andra Stroe"
__version__ = "0.1"

from contextlib import contextmanager
import gc

import numpy as np
import gleam.matplotlibparams
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.visualization import quantity_support
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy.table import vstack

import gleam.read_files as rf
import gleam.gaussian_fitting as gf
import gleam.spectra_operations as so

quantity_support()


def overview_plot(target, data_path, spectrum, successful_fits):
    """
    Creates the main overview plot for a source, highlighting regions
    where line fits were successful.
    """
    basename = rf.naming_convention(
        data_path,
        target["Sample"][0],
        target["SourceNumber"][0],
        target["Setup"][0],
        target["Pointing"][0],
        "linefits",
    )
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.plot(spectrum["wl_rest"], spectrum["flux_rest"], drawstyle="steps-mid", color='black', lw=0.7, label='Flux', rasterized=True)
    
    ax.plot(spectrum["wl_rest"], spectrum["flux_rest"] + spectrum["stdev_rest"], drawstyle="steps-mid", color='gray', lw=0.4, alpha=0.8, rasterized=True, label='1$\sigma$ Noise Envelope')
    ax.plot(spectrum["wl_rest"], spectrum["flux_rest"] - spectrum["stdev_rest"], drawstyle="steps-mid", color='gray', lw=0.4, alpha=0.8, rasterized=True)

    for fit in successful_fits:
        if fit.lines:
            wl_min = np.min(fit.model_fit.userkws['x'])
            wl_max = np.max(fit.model_fit.userkws['x'])
            ax.axvspan(wl_min, wl_max, color='cyan', alpha=0.3, zorder=0)

    try:
        mean, median, std = sigma_clipped_stats(spectrum['flux_rest'], sigma=3.0)
        padding = 2 * std
        y_min = median - padding
        y_max = median + 5 * padding
        ax.set_ylim(y_min, y_max)
    except:
        ax.set_ylim(np.nanpercentile(spectrum['flux_rest'], [5, 99]))

    ax.set_xlim(np.nanmin(spectrum["wl_rest"].value), np.nanmax(spectrum["wl_rest"].value))
    ax.set_xlabel(r"Rest-frame Wavelength [$\rm{\AA}$]")
    ax.set_ylabel(r"Flux [erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]")
    ax.minorticks_on()
    ax.legend(loc="upper right")
    
    plot_title = f'{target["Sample"][0]} {target["Setup"][0]} {target["Pointing"][0]} Source \\#{target["SourceNumber"][0]}'
    ax.set_title(plot_title)
    
    filename = f"{basename}.overview.png"
    fig.savefig(filename, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    gc.collect()


def single_line_plot(target, data_path, spectrum, full_spectrum, line_list, line_fit, line):
    name = line.name
    line_wl = line.wl

    basename = rf.naming_convention(
        data_path,
        target["Sample"][0],
        target["SourceNumber"][0],
        target["Setup"][0],
        target["Pointing"][0],
        "linefits",
    )
    filename = f"{basename}.{name}.png"

    try:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(full_spectrum["wl_rest"], full_spectrum["flux_rest"], drawstyle="steps-mid", color='grey', lw=0.5, alpha=0.7, rasterized=True)
        ax.plot(spectrum["wl_rest"], spectrum["flux_rest"], drawstyle="steps-mid", color='black', lw=1.2, label='Spectrum', rasterized=True)
        
        if line_fit:
            wl_model = spectrum["wl_rest"]
            # --- FIX: Plot the attempted fit for non-detections ---
            if line.detection_flag == 1:
                # This is a detection, plot in solid red
                ax.plot(wl_model, line_fit.model_fit.best_fit, color='red', lw=1.5, label='Full Model')
                comps = line_fit.model_fit.eval_components(x=wl_model.value)
                for prefix, component in comps.items():
                    ax.plot(wl_model, component, color='orange', ls='--', lw=1.0)
            else:
                # This is a non-detection, plot the attempted fit in dashed blue
                ax.plot(wl_model, line_fit.model_fit.best_fit, color='blue', ls='--', lw=1.2, label='Attempted Fit')

        
        zoom_window = 20 * u.AA
        mask = (spectrum["wl_rest"] > (line_wl - zoom_window)) & (spectrum["wl_rest"] < (line_wl + zoom_window))
        
        data_min, data_max = 0, 0
        if np.any(mask):
            try:
                mean, median, std = sigma_clipped_stats(spectrum["flux_rest"][mask])
                data_min = median - 2 * std
                data_max = np.max(spectrum['flux_rest'][mask])
            except:
                data_min = np.nanmin(spectrum['flux_rest'][mask].value)
                data_max = np.nanmax(spectrum['flux_rest'][mask].value)

        # Y-axis scaling should now consider the model fit in all cases
        model_max = 0
        if line_fit:
            model_max = np.nanmax(line_fit.model_fit.best_fit)
        
        final_max = max(data_max.value, model_max) if hasattr(data_max, 'value') else max(data_max, model_max)
        ax.set_ylim(data_min, final_max * 1.2)
        ax.set_xlim(line_wl - 50 * u.AA, line_wl + 50 * u.AA)

        ax.legend(loc='upper right')
        ax.set_xlabel(r"Rest-frame Wavelength [$\rm{\AA}$]")
        ax.set_ylabel(r"Flux [erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]")
        
        safe_name = name.replace('#', '\\#').replace('_', '\\_')
        plot_title = f'{target["Sample"][0]} {target["SourceNumber"][0]} | {safe_name} | z={target["Redshift"][0]:.4f}'
        if line.detection_flag == 0:
            plot_title += ' (Non-detection)'
        ax.set_title(plot_title)
        
        fig.savefig(filename, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    except Exception as e:
        print(f"Warning: Failed to create single plot for {name}. Reason: {e}")
        if 'fig' in locals(): 
            plt.close(fig)