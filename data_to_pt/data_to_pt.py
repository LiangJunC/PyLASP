# -*- coding: utf-8 -*-
# @Time    : 2025/11/23 10:18
# @Author  : ljc
# @FileName: data_to_pt.py
# @Software: Pycharm


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Converting FITS files to .pt format for LASP-Adam-GPU.

1.1 Purpose
-----------
Convert observed spectral FITS files to PyTorch tensor format (.pt) for
efficient batch processing in LASP-Adam-GPU. This module handles
spectral reading, initial parameter, and parallel data storage.

1.2 Functions
-------------
1) data_to_pt: Main function for batch conversion of FITS spectra to
   PyTorch tensor format with parallel processing support.

1.3 Explanation
---------------
1.3.1 Data Processing Workflow
The data_to_pt function processes observed spectra through the
following steps:
    1) Read spectral FITS files and construct spectrum dictionary
       structure
    2) Generate Legendre polynomial array for correcting shape
       differences
    3) Assign initial stellar parameters (Teff, logg, [Fe/H]) and mu,
       sigma to each spectrum
    4) Call ulyss function to obtain ELODIE_wave, borders, NewBorders,
       flat, lamrange, flux_obs, goodpix_, spec_coef
    5) Convert all data to PyTorch tensors
    6) Group spectra into batches and save as .pt files

1.3.2 Parallel Processing
The function uses joblib's Parallel and delayed to process multiple
spectrum groups simultaneously, significantly reducing total processing
time for large datasets.

"""

# ======================================================================
# 2. Import libraries
# ======================================================================
import numpy as np
import time
from tqdm import *
from config.config import default_set, set_all_seeds
import torch
from joblib import Parallel, delayed
from uly_read_lms.uly_spect_read_lms import uly_spect_read_lms
from uly_fit.ulyss_pytorch import ulyss
from scipy.special import eval_legendre
from file_paths import TEST_DATA_DIR

# 2.1 Set random seed
set_all_seeds()
# 2.2 Call GPU and specify data type
dtype, device = default_set()
# 2.3 Set default data type
torch.set_default_dtype(dtype)


# ======================================================================
# 3. Convert FITS spectra to PyTorch tensor format
# ======================================================================
def data_to_pt(
    all_spec_fits_names: list,
    base_fits_dir: str,
    model_file: str,
    save_pt_file: str,
    each_pt_number: int = 20000,
    npix: int = 1327,
    lmin: list = [4200],
    lmax: list = [5700],
    n_jobs: int = 9,
    ini_Teff: float = 7500.0,
    ini_logg: float = 3.5,
    ini_FeH: float = -0.5,
    ini_mu: float = 0.0,
    ini_sigma: float = 1.0,
) -> None:
    r"""Convert observed spectral FITS files to PyTorch tensor format.

    Parameters
    ----------
    all_spec_fits_names : list
       List of FITS file names (without directory path) to be
       processed. Example: ["spec001.fits", "spec002.fits", ...].
    base_fits_dir : str
        Directory path containing all FITS files. Example:
        "/test_data/all_spec_fits/".
    model_file : str
        Path to the model file.
    save_pt_file : str
        Base path for saving output .pt files. Files will be saved as
        save_pt_file + "0.pt", save_pt_file + "1.pt", etc.
    each_pt_number : int, default=20000
        Number of spectra to include in each .pt file.
    npix : int, default=1327
        Number of pixels in the resampled spectrum.
    lmin : list, default=[4200]
        Minimum wavelength(s) in Angstroms for spectral range.
    lmax : list, default=[5700]
        Maximum wavelength(s) in Angstroms for spectral range.
    n_jobs : int, default=9
        Number of parallel jobs for processing. Adjust based on
        available CPU cores.
    ini_Teff : float, default=7500.0
        Initial effective temperature in Kelvin for all spectra.
    ini_logg : float, default=3.5
        Initial surface gravity (log g) for all spectra.
    ini_FeH : float, default=-0.5
        Initial metallicity [Fe/H] for all spectra.
    ini_mu : float, default=0.0
        Initial mean of the Gaussian convolution kernel.
    ini_sigma : float, default=1.0
         Initial standard deviation of the Gaussian convolution kernel.

    Returns
    -------
    None
        Saves processed data as .pt files to disk.

    Notes
    -----
    - Each .pt file contains a dictionary with keys: spectra_data,
      wavelength_data, borders, NewBorders, flat, spec_coef, leg_array,
      ini_params, ID.
    - The function uses parallel processing to handle large datasets
      efficiently.
    - Total processing time is printed at the end in hours.

    """

    # ------------------------------------------------------------------
    # 3.1 Calculate number of .pt files needed
    # ------------------------------------------------------------------
    number_pt = int(np.ceil(len(all_spec_fits_names) / each_pt_number))

    # ------------------------------------------------------------------
    # 3.2 Define processing function for each group
    # ------------------------------------------------------------------
    def f(ii):
        r"""Process and save one group of spectra.

        Parameters
        ----------
        ii : int
            Group index number.

        """
        # --------------------------------------------------------------
        # 3.2.1 Determine spectrum addresses and group size for current
        #       group
        # --------------------------------------------------------------
        if ii < (number_pt - 1):
            ii_group_spec_fits_address = all_spec_fits_names[
                ii * each_pt_number : (ii + 1) * each_pt_number
            ]
            group_size = each_pt_number
        if ii == (number_pt - 1):
            ii_group_spec_fits_address = all_spec_fits_names[
                (number_pt - 1) * each_pt_number :
            ]
            group_size = len(all_spec_fits_names) - (number_pt - 1) * each_pt_number

        # --------------------------------------------------------------
        # 3.2.2 Initialize data arrays for current group
        # --------------------------------------------------------------
        wavelength_data, spectra_data, goodpix, ini_params, IDS, x = (
            np.zeros(shape=(group_size, npix)),
            np.zeros(shape=(group_size, npix)),
            np.zeros(shape=(group_size, npix - 1)),
            np.zeros(shape=(group_size, 5)),
            np.zeros(shape=(group_size, 1), dtype=object),
            2.0 * np.arange(npix) / npix - 1.0,
        )
        leg_array = eval_legendre(np.arange(51), x[:, np.newaxis])

        # --------------------------------------------------------------
        # 3.2.3 Process each spectrum in current group
        # --------------------------------------------------------------
        for ff in range(group_size):
            # 3.2.3.1 Store spectrum file name as ID
            IDS[ff, 0] = ii_group_spec_fits_address[ff]
            # 3.2.3.2 Assign initial stellar parameters (Teff, logg,
            #         [Fe/H], mu, sigma)
            ini_params[ff, :] = np.array(
                [ini_Teff, ini_logg, ini_FeH, ini_mu, ini_sigma]
            ).reshape(1, -1)

            # 3.2.3.3 Construct spectrum dictionary structure
            inspectr = uly_spect_read_lms(
                lmin=lmin,
                lmax=lmax,
                file_in=base_fits_dir + ii_group_spec_fits_address[ff],
                public=True,
                flux_median=False,
            )

            # 3.2.3.4 Process spectrum with ulyss function
            (
                ELODIE_wave,
                borders,
                NewBorders,
                flat,
                lamrange,
                flux_obs,
                goodpix_,
                spec_coef,
            ) = ulyss(inspectr=inspectr, model_file=model_file)

            # 3.2.3.5 Store processed data for current spectrum
            spectra_data[ff, :], goodpix[ff, :], wavelength_data[ff, :] = (
                flux_obs,
                goodpix_,
                lamrange,
            )

        # --------------------------------------------------------------
        # 3.2.4 Correct resampled wavelength range boundaries
        # --------------------------------------------------------------
        if NewBorders[0] < borders[0]:
            NewBorders[0] = borders[0]
        if NewBorders[-1] > borders[-1]:
            NewBorders[-1] = borders[-1]

        # --------------------------------------------------------------
        # 3.2.5 Convert numpy arrays to PyTorch tensors
        # --------------------------------------------------------------
        (
            spectra_data,
            wavelength_data,
            borders,
            NewBorders,
            flat,
            spec_coef,
            ini_params,
            leg_array,
        ) = (
            torch.tensor(spectra_data).to(dtype=dtype),
            torch.tensor(wavelength_data[0, :]).to(dtype=dtype),
            torch.tensor(np.array(borders).reshape(1, -1)).to(dtype=dtype),
            torch.tensor(np.array(NewBorders).reshape(1, -1)).to(dtype=dtype),
            torch.tensor(np.array(flat).reshape(1, -1)).to(dtype=dtype),
            torch.tensor(spec_coef).to(dtype=dtype),
            torch.tensor(ini_params).to(dtype=dtype),
            torch.tensor(leg_array).to(dtype=dtype),
        )

        # --------------------------------------------------------------
        # 3.2.6 Create data dictionary for current group
        # --------------------------------------------------------------
        data_dict = {
            "spectra_data": spectra_data,
            "wavelength_data": wavelength_data,
            "borders": borders,
            "NewBorders": NewBorders,
            "flat": flat,
            "spec_coef": spec_coef,
            "leg_array": leg_array,
            "ini_params": ini_params,
            "ID": IDS,
        }

        # --------------------------------------------------------------
        # 3.2.7 Save data dictionary as .pt file
        # --------------------------------------------------------------
        torch.save(data_dict, save_pt_file + str(ii) + ".pt")

    # ------------------------------------------------------------------
    # 3.3 Execute parallel processing for all groups
    # ------------------------------------------------------------------
    time1 = time.time()
    Parallel(n_jobs=n_jobs)(delayed(f)(i) for i in tqdm(range(0, number_pt)))
    time2 = time.time()
    print("Elapsed time: ", str(round((time2 - time1) / 3600, 5)) + " h!")
