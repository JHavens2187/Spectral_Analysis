__author__ = "Andra Stroe"
__version__ = "0.1"


import os, sys
from dataclasses import dataclass
from typing import List
import functools
import operator


import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.table import QTable, Column
from colorama import Fore
from colorama import init

init(autoreset=True)


def average_(x, n):
    return np.average(x.reshape((-1, n)), axis=1)


def average_err(x, n):
    return np.sqrt(np.average((x ** 2).reshape((-1, n)), axis=1) / n)


def sigma_to_fwhm(sigma):
    if sigma is None or np.isnan(sigma):
        return np.nan
    return sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))


def fwhm_to_sigma(fwhm):
    if fwhm is None or np.isnan(fwhm):
        return np.nan
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def dispersion(wl):
    return np.median(wl[1:] - wl[:-1])

def bin_spectrum(spectrum, n):
    max_len = len(spectrum) // n * n
    spectrum_trimmed = spectrum[:max_len]

    binned_wl = average_(spectrum_trimmed["wl_obs"].value, n) * u.angstrom
    binned_flux = average_(spectrum_trimmed["flux_obs"].value, n) * (
        u.erg / u.s / u.cm ** 2 / u.AA
    )
    binned_stdev = average_err(spectrum_trimmed["stdev_obs"].value, n) * (
        u.erg / u.s / u.cm ** 2 / u.AA
    )

    binned_spec = QTable(
        [binned_wl, binned_flux, binned_stdev],
        names=("wl_obs", "flux_obs", "stdev_obs"),
        masked=True,
    )
    return binned_spec


def add_restframe(spectrum, target):
    z_ref = target["Redshift"][0]
    spectrum.add_column(
        Column(spectrum["wl_obs"] / (1 + z_ref), name="wl_rest", unit=u.angstrom)
    )
    spectrum.add_column(
        Column(
            spectrum["flux_obs"] * (1 + z_ref),
            name="flux_rest",
            unit=u.erg / u.s / u.cm ** 2 / u.AA,
        )
    )
    spectrum.add_column(
        Column(
            spectrum["stdev_obs"] * (1 + z_ref),
            name="stdev_rest",
            unit=u.erg / u.s / u.cm ** 2 / u.AA,
        )
    )
    return spectrum


def group_lines(line_list, tolerance):
    """
    Group lines that are close together to be fit at the same time.
    This version is now unit-aware and correctly handles all elements.
    """
    if len(line_list) == 0:
        return line_list.group_by("group")

    line_list["group"] = 0
    group = 0
    wl_col = 'wavelength' if 'wavelength' in line_list.colnames else 'wl'
    
    if not isinstance(tolerance, u.Quantity):
        tolerance = tolerance * u.AA
        
    line_list.sort(wl_col)
    
    for i in range(len(line_list) - 1):
        line_list["group"][i] = group
        if (line_list[wl_col][i + 1] - line_list[wl_col][i]) > tolerance:
            group += 1
            
    line_list["group"][-1] = group
    
    return line_list.group_by("group")


def select_lines_for_fitting(line_list, spectrum, target, cont_width, sky_lines):
    z_ref = target["Redshift"][0]
    wl_min = np.min(spectrum["wl_rest"])
    wl_max = np.max(spectrum["wl_rest"])
    
    wl_col = 'wavelength' if 'wavelength' in line_list.colnames else 'wl'

    lines_in_range = line_list[
        (line_list[wl_col] > wl_min) & (line_list[wl_col] < wl_max)
    ]

    if sky_lines is None:
        return lines_in_range

    try:
        sky_mask_func = functools.partial(
            mask_atmosphere,
            z_ref=z_ref,
            sky=sky_lines,
            cont_width=cont_width,
            invert=True,
        )
        wl_values = lines_in_range[wl_col].value
        sky_mask = sky_mask_func(wl_values)
        return lines_in_range[sky_mask]

    except Exception:
        print(Fore.YELLOW + "Warning: Sky line masking failed. Proceeding without it.")
        return lines_in_range


def mask_line(wl, center, width):
    return (wl < center - width) | (wl > center + width)


def select_singleline(wl, center, width):
    return (wl > center - width) & (wl < center + width)


def mask_atmosphere(wl, z_ref, sky, cont_width, invert=False):
    if sky is None:
        return np.full(np.shape(wl), True)

    z_dep_sky = sky["wl"] * (1 + z_ref)

    mask = np.full(np.shape(wl), True)
    for line in z_dep_sky:
        mask = mask & mask_line(wl, line, cont_width / 2.0)

    if invert:
        return ~mask
    else:
        return mask


def select_lines(selected_lines, other_lines, spectrum, target, sky, cont_width, mask_width):
    z_ref = target["Redshift"][0]
    wl_rest = spectrum["wl_rest"]

    if not isinstance(cont_width, u.Quantity):
        cont_width = cont_width * u.AA
    if not isinstance(mask_width, u.Quantity):
        mask_width = mask_width * u.AA

    masked_otherlines = np.full(np.shape(wl_rest), True)
    for line_wl in other_lines:
        mask = mask_line(wl_rest, line_wl, mask_width)
        masked_otherlines = masked_otherlines & mask

    select_region = np.full(np.shape(wl_rest), False)
    for line_wl in selected_lines:
        mask = select_singleline(wl_rest, line_wl, cont_width)
        select_region = select_region | mask

    masked_atm = mask_atmosphere(wl_rest, z_ref, sky, cont_width)

    return masked_atm & masked_otherlines & select_region


@dataclass
class Component:
    segments: List
    beginning: float
    ending: float


def connected_components(segments, left, right):
    segments = sorted(segments, key=left)
    try:
        head, *segments = segments
    except:
        return []

    ref_component = Component(segments=[head], beginning=left(head), ending=right(head))
    components = [ref_component]

    for segment in segments:
        opening = left(segment)
        closing = right(segment)
        if ref_component.ending > opening:
            ref_component.segments.append(segment)
            if closing > ref_component.ending:
                ref_component.ending = closing
        else:
            ref_component = Component(
                segments=[segment], beginning=opening, ending=closing
            )
            components.append(ref_component)
    return components