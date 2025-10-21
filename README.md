# Spectral Analysis of JADES and SMACS Data

This repository contains Python code, primarily in Jupyter Notebook format, for downloading, processing, and analyzing astronomical spectra obtained from the JWST Advanced Deep Extragalactic Survey (JADES) and SMACS surveys using the NIRSpec instrument.

## Overview

The notebooks provide workflows for:

* Downloading publicly available JADES and SMACS spectral data (`.fits` files).
* Configuring analysis parameters using `.yaml` files.
* Cross-matching spectral data with target catalogs (`.csv` files).
* Combining multi-grating spectra into single `Spectrum1D` objects.
* Performing continuum subtraction and redshift correction.
* Fitting emission lines using tools like `specutils` and **GLEAM** (`lmfit`).
* Generating diagnostic plots and final data catalogs.
* Interacting with Google Sheets for data management.

## Data Sources

* **JADES:** JWST Advanced Deep Extragalactic Survey
* **SMACS:** SMACS 0723 Galaxy Cluster Field

## Key Libraries & Tools

This analysis relies heavily on the Python scientific ecosystem, including:

* **Core:** `numpy`, `pandas`, `matplotlib`, `GLEAM`
* **Astronomy:** `astropy`, `specutils`, `astro-gleam`, `lmfit`
* **Performance/Imaging:** `numba`, `opencv-python` (`cv2`)
* **Data Handling:** `requests`, `json`

## Usage

The notebooks are designed to be run sequentially or independently, depending on the analysis step. Configuration details (file paths, analysis parameters) are typically set at the beginning of each notebook or in separate configuration files. Ensure all required Python packages are installed.

## Acknowledgments

This work is conducted as part of undergraduate research at the University of Kansas under the supervision of Dr. Allison Kirkpatrick and Dr. Bren Backhaus.

---

*Contact: Joseph Havens (joe.havens79@ku.edu)*
