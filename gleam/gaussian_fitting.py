__author__ = "Andra Stroe"
__version__ = "0.1"

import sys
from typing import List, Union
from dataclasses import dataclass
import itertools
import os

import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.table import QTable, Column, hstack
from astropy.stats import sigma_clipped_stats
from astropy.units.quantity import Quantity as Qty
from colorama import Fore
from astropy.cosmology import FlatLambdaCDM

from lmfit.models import GaussianModel, ConstantModel
from lmfit.model import ModelResult

import gleam.spectra_operations as so
from gleam.constants import Length
from matplotlib import pyplot as plt


@dataclass
class RandomVariable:
    value: Qty
    error: Qty

    @staticmethod
    def from_param(param, scale=1.0):
        error_val = param.stderr if param.stderr is not None else np.nan
        return RandomVariable(value=param.value * scale, error=error_val * scale)

    def _propagate_multiplication_error(self, other):
        """Helper for propagating error for multiplication (self * other)."""
        if self.error is None or other.error is None or np.isnan(self.error) or np.isnan(other.error):
            return np.nan
        
        val1 = self.value if self.value != 0 else 1e-9
        val2 = other.value if other.value != 0 else 1e-9

        relative_err_sq = (self.error / val1) ** 2 + (other.error / val2) ** 2
        return abs(self.value * other.value) * np.sqrt(relative_err_sq)

    def _propagate_division_error(self, other):
        """Helper for propagating error for division (self / other)."""
        if self.error is None or other.error is None or np.isnan(self.error) or np.isnan(other.error):
            return np.nan

        val1 = self.value if self.value != 0 else 1e-9
        val2 = other.value if other.value != 0 else 1e-9

        relative_err_sq = (self.error / val1) ** 2 + (other.error / val2) ** 2
        return abs(self.value / val2) * np.sqrt(relative_err_sq)

    @property
    def badness(self):
        if self.error is None or self.value == 0: return np.inf
        return self.error / self.value

    @property
    def significance(self):
        return None if self.error is None else abs(self.value / self.error)

    def __mul__(self, other):
        if isinstance(other, RandomVariable):
            new_value = self.value * other.value
            new_error = self._propagate_multiplication_error(other)
            return RandomVariable(value=new_value, error=new_error)
        else:
            return RandomVariable(value=self.value * other, error=self.error * other if self.error is not None else np.nan)

    def __rmul__(self, other):
        return self * other
        
    def __truediv__(self, other):
        if isinstance(other, RandomVariable):
            new_value = self.value / other.value
            new_error = self._propagate_division_error(other)
            return RandomVariable(value=new_value, error=new_error)
        else:
            return RandomVariable(value=self.value / other, error=self.error / other if self.error is not None else np.nan)


@dataclass
class SpectrumFit:
    lines: List['Line']
    model_fit: ModelResult


@dataclass
class Line:
    original_row: QTable
    name: str
    wl: Qty
    continuum: RandomVariable
    center: RandomVariable
    flux: RandomVariable
    eq_width: RandomVariable
    fwhm: RandomVariable
    chi2: float
    chi2_reduced: float
    detection_flag: int

    def as_fits_table(self):
        return hstack([self.original_row, QTable({
            "continuum": [self.continuum.value], "continuum_err": [self.continuum.error],
            "center": [self.center.value], "center_err": [self.center.error],
            "flux": [self.flux.value], "flux_err": [self.flux.error],
            "eq_width": [self.eq_width.value], "eq_width_err": [self.eq_width.error],
            "fwhm": [self.fwhm.value], "fwhm_err": [self.fwhm.error],
            "chi2": [self.chi2], "chi2_reduced": [self.chi2_reduced],
            "detection_flag": [self.detection_flag]
        })])


def guess_parameters(lines, spectrum, rest_spectral_resolution, cont_width):
    wl_col = 'wavelength' if 'wavelength' in lines.colnames else 'wl'
    center = lines[wl_col][0]

    if not isinstance(cont_width, u.Quantity):
        cont_width = cont_width * u.AA

    try:
        _, continuum_guess, _ = sigma_clipped_stats(spectrum["flux_rest"])
    except:
        continuum_guess = np.nanmedian(spectrum["flux_rest"].value) * spectrum["flux_rest"].unit

    continuum_subtracted_flux = spectrum["flux_rest"] - continuum_guess
    
    w_guess = cont_width / 2.0
    select = np.abs(spectrum["wl_rest"] - center) < w_guess

    if np.any(select):
        try:
            _, _, amp_std = sigma_clipped_stats(continuum_subtracted_flux[select], sigma=2.0)
            amp_median = np.nanmedian(continuum_subtracted_flux[select])
            amp = np.nanmax([amp_median + 3 * amp_std, np.nanmax(continuum_subtracted_flux[select])])
        except:
            amp = np.nanmax(continuum_subtracted_flux[select])
    else:
        amp = continuum_guess * 0.1 

    sigma = (center / const.c * rest_spectral_resolution.to(u.m / u.s)).to(u.AA)
    sigma = so.fwhm_to_sigma(sigma)

    guesses = []
    for line in lines:
        guesses.append({
            "amplitude": amp, "center": line[wl_col], "sigma": sigma,
            "continuum": continuum_guess
        })
    return guesses


def model_selection(
    target,
    lines,
    spectrum,
    center_constraint,
    verbose,
    inspect,
    rest_spectral_resolution,
    SN_limit,
    cont_width,
):
    guesses = guess_parameters(lines, spectrum, rest_spectral_resolution, cont_width)
    wl_col = 'wavelength' if 'wavelength' in lines.colnames else 'wl'

    scale = np.nanmedian(spectrum["flux_rest"].value)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0 

    y_data_scaled = spectrum["flux_rest"].value / scale
    stdev_scaled = spectrum["stdev_rest"].value / scale
    weights = 1 / stdev_scaled

    model = ConstantModel()
    for i, line in enumerate(lines):
        model += GaussianModel(prefix=f"g{i}_")

    params = model.make_params()
    params["c"].set(value=guesses[0]['continuum'].value / scale)

    for i, (line, guess) in enumerate(zip(lines, guesses)):
        params[f"g{i}_center"].set(value=guess["center"].value, min=guess["center"].value - 5, max=guess["center"].value + 5)
        params[f"g{i}_amplitude"].set(value=guess["amplitude"].value / scale, min=0)
        params[f"g{i}_sigma"].set(value=guess["sigma"].value, min=so.fwhm_to_sigma(0.5), max=so.fwhm_to_sigma(15))

    if center_constraint:
        for i in range(1, len(lines)):
            params[f"g{i}_center"].set(expr=f'g0_center * {lines[wl_col][i].value / lines[wl_col][0].value:.4f}')

    result = model.fit(
        y_data_scaled, params,
        x=spectrum["wl_rest"].value, weights=weights, method="least_squares",
    )
    
    result.best_fit *= scale
    if result.init_fit is not None: result.init_fit *= scale


    if verbose: print(result.fit_report())
    if inspect:
        plt.figure(); result.plot(); plt.show()

    fitted_lines = []
    for i, line_row in enumerate(lines):
        line_name_col = 'name' if 'name' in line_row.colnames else 'line'
        line_name = line_row[line_name_col] if line_name_col in line_row.colnames else f"line_{line_row[wl_col].value:.2f}"

        if f"g{i}_amplitude" in result.params:
            amp = RandomVariable.from_param(result.params[f"g{i}_amplitude"], scale)
            center = RandomVariable.from_param(result.params[f"g{i}_center"])
            sigma = RandomVariable.from_param(result.params[f"g{i}_sigma"])
            cont = RandomVariable.from_param(result.params["c"], scale)
            fwhm_val = so.sigma_to_fwhm(sigma.value)
            fwhm_err = so.sigma_to_fwhm(sigma.error) if sigma.error else np.nan
            fwhm = RandomVariable(value=fwhm_val, error=fwhm_err)

            # --- FIX: Implement significance check for upper limits ---
            is_significant = amp.significance is not None and amp.significance >= SN_limit
            detection_flag = 1 if is_significant else 0
            
            if is_significant:
                # This is a detection, calculate flux normally
                flux = amp * sigma * np.sqrt(2 * np.pi)
                eq_width = flux / cont
            else:
                # This is a non-detection, calculate upper limit
                # UL = S/N_limit * RMS * sqrt(N_pix) * delta_lambda
                # Approximated as: UL = S/N_limit * Continuum_Error * FWHM
                continuum_noise = np.nanmedian(spectrum["stdev_rest"])
                flux_upper_limit = SN_limit * continuum_noise * fwhm.value
                
                flux = RandomVariable(value=flux_upper_limit, error=np.nan)
                eq_width = RandomVariable(value=flux_upper_limit / cont.value, error=np.nan)

            fitted_lines.append(
                Line(
                    original_row=line_row, name=line_name, wl=line_row[wl_col],
                    continuum=cont, center=center, flux=flux,
                    eq_width=eq_width, fwhm=fwhm,
                    chi2=result.chisqr, chi2_reduced=result.redchi,
                    detection_flag=detection_flag
                )
            )

    return SpectrumFit(lines=fitted_lines, model_fit=result) if fitted_lines else None


def fit_lines(
    selected_lines,
    full_line_list,
    spectrum,
    target,
    center_constraint,
    verbose,
    inspect,
    cont_width,
    rest_spectral_resolution,
    sky,
    mask_width=5 * u.AA,
    w=5 * u.AA,
    SN_limit=3,
    cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
):
    if not isinstance(cont_width, u.Quantity):
        cont_width = cont_width * u.AA

    wl_col = 'wavelength' if 'wavelength' in selected_lines.colnames else 'wl'
    selected_wavelengths = selected_lines[wl_col]

    all_wl_col = 'wavelength' if 'wavelength' in full_line_list.colnames else 'wl'
    all_wavelengths = full_line_list[all_wl_col]

    group_center = np.mean(selected_wavelengths)
    search_radius = cont_width * 1.5 

    close_lines_mask = np.abs(all_wavelengths - group_center) < search_radius
    not_in_current_group_mask = ~np.in1d(all_wavelengths.value, selected_wavelengths.value)

    other_wavelengths_to_mask = all_wavelengths[close_lines_mask & not_in_current_group_mask]

    mask_line = so.select_lines(
        selected_wavelengths,
        other_wavelengths_to_mask,
        spectrum, target, sky, cont_width, mask_width,
    )

    spectrum_line = spectrum[mask_line]

    if len(spectrum_line) < 3:
        name_col = 'name' if 'name' in selected_lines.colnames else 'line'
        if name_col in selected_lines.colnames:
            line_names = ", ".join(selected_lines[name_col])
            print(Fore.CYAN + f"DEBUG: Fit failed for '{line_names}'. Reason: Insufficient data points ({len(spectrum_line)}).")
        else:
            print(Fore.CYAN + f"DEBUG: Fit failed. Reason: Insufficient data points ({len(spectrum_line)}).")
        return None, None

    spectrum_fit = model_selection(
        target, selected_lines, spectrum_line,
        center_constraint, verbose, inspect,
        rest_spectral_resolution, SN_limit,
        cont_width,
    )
    return spectrum_fit, spectrum_line