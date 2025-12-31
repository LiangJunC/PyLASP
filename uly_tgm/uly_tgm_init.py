# -*- coding: utf-8 -*-
# @Time    : 04/12/2024 17.16
# @Author  : ljc
# @FileName: uly_tgm_init.py
# @Software: PyCharm
# Update:  2025/11/26 22:02:31


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of model component initial for LASP-CurveFit.

1.1 Purpose
-----------
Initialize a model component (cmp) structure, converted from IDL
uly_tgm_init.pro implementation.

1.2 Functions
-------------
1) uly_tgm_coef_load: Load model polynomial coefficients from FITS file
   and return model information dictionary s.
2) uly_tgm_init: Initialize model component structure and return configured
   component dictionary. Used in 'uly_fit/uly_fit_init.py'.

1.3 Explanation
---------------
This module provides model component (cmp) initialization for PyLASP.
Steps:
    1) Load model polynomial coefficients from a FITS file (warm, hot, cold).
    2) Extract wavelength information from the FITS header.
    3) Configure pixel mask for NaD line region.
    4) Resample model spectrum to match observed wavelength grid.
    5) Update and return component structure (cmp).

1.4 Notes
---------
- This is a Python-specific rewrite and optimization, not a complete
  port of all IDL features, applicable to LASP-CurveFit.
- This module uses 'warnings.filterwarnings("ignore")' to globally
  suppress Python warnings for cleaner large-scale runs. If you
  rely on warnings from NumPy/SciPy or other libraries, please
  comment out or adjust this line in your environment.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
import numpy as np
from astropy.io import fits
from file_paths import TGM_MODEL_FILE
from uly_read_lms.uly_spect_alloc import uly_spect_alloc
from uly_read_lms.uly_spect_extract import uly_spect_extract
from uly_read_lms.uly_spect_get import uly_spect_get
from WRS.uly_spect_logrebin import uly_spect_logrebin
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray
C_light = 299792.458


# ======================================================================
# 4. Model coefficient loading function
# ======================================================================
def uly_tgm_coef_load(filename: str | None = None) -> dict:
    r"""Load model polynomial coefficients and wavelength information.

    Parameters
    ----------
    filename : str, optional
        Path to model FITS file (ELODIE version 3.2).
        If None, use a default path from file_paths.TGM_MODEL_FILE().

    Returns
    -------
    s : dict
        Model information dictionary.

    Raises
    ------
    ValueError
        - If the wavelength sampling type read from the FITS header
          (CTYPE1) is neither 'AWAV' nor 'AWAV-LOG'.
        - If no good pixels remain after applying the NaD region mask.

    Notes
    -----
    - Model contains three coefficient sets for different stellar
      temperature regimes: warm, hot, and cold stars.
    - Coefficient matrix shape is (n_coeffs, n_wavelengths, 3) where
      3 represents warm, hot, and cold regimes.
    - Default pixel mask excludes NaD line region (5876.87-5909.35 $\AA$).

    Examples
    --------
    >>> from file_paths import TGM_MODEL_FILE
    >>> s = uly_tgm_coef_load(TGM_MODEL_FILE())
    >>> print(s['data'].shape)
    (26, 14501, 3)

    """

    # ------------------------------------------------------------------
    # 4.1 Set default model file path if not provided
    # ------------------------------------------------------------------
    if filename is None:
        filename = TGM_MODEL_FILE()

    # ------------------------------------------------------------------
    # 4.2 Load model polynomial coefficients from FITS file
    # ------------------------------------------------------------------
    with fits.open(filename) as hdul:
        # 4.2.1 Load hdr, warm/hot/cold stars coefficient matrix
        hdr, warm, hot, cold = hdul[0].header, hdul[0].data, hdul[1].data, hdul[2].data
        # 4.2.2 Stack coefficient matrices along third axis
        spec_coef = np.nan_to_num(np.stack([warm, hot, cold], axis=-1))
        # 4.2.3 Extract wavelength parameters from header
        start, step, npix, sampling_type = (
            hdr.get("CRVAL1", 0.0),
            hdr.get("CDELT1", 1.0),
            hdr.get("NAXIS1", 0),
            hdr.get("CTYPE1", "AWAV"),
        )

    # ------------------------------------------------------------------
    # 4.3 Determine wavelength sampling mode
    # ------------------------------------------------------------------
    if sampling_type == "AWAV":
        sampling = 0
    elif sampling_type == "AWAV-LOG":
        sampling = 1
    else:
        raise ValueError(
            f"Invalid CTYPE1 value {sampling_type}. "
            "Expected 'AWAV' or 'AWAV-LOG'. Please check the wavelength sampling."
        )

    # ------------------------------------------------------------------
    # 4.4 Create default pixel mask excluding NaD line region
    # ------------------------------------------------------------------
    # 4.4.1 Compute wavelength array and initialize all pixels as good
    wl, mask = (
        start + np.linspace(start=0, stop=npix - 1, num=npix) * step,
        np.ones(npix, dtype=np.uint8),
    )
    # 4.4.2 Set bad pixels according to wavelength sampling format
    if sampling == 0:
        bad = np.where((wl >= 5876.87) & (wl <= 5909.35))[0]
    if sampling == 1:
        # log10 scale (20000 $\AA$, log10(20000) = 4.3 < 5)
        if wl[-1] < 5:
            bad = np.where((wl >= np.log10(5876.87)) & (wl <= np.log10(5909.35)))[0]
        # natural log scale
        else:
            bad = np.where((wl >= np.log(5876.87)) & (wl <= np.log(5909.35)))[0]
    # 4.4.3 Generate indices of good pixels
    mask[bad] = 0
    goodpix = np.where(mask == 1)[0]
    if len(goodpix) == 0:
        raise ValueError("Error, no good pixels found.")

    # ------------------------------------------------------------------
    # 4.5 Construct model information dictionary
    # ------------------------------------------------------------------
    s = uly_spect_alloc(
        data=spec_coef,
        start=start,
        step=step,
        sampling=sampling,
        goodpix=goodpix,
        header=hdr,
    )

    # ------------------------------------------------------------------
    # 4.6 Return model information dictionary
    # ------------------------------------------------------------------
    return s


# ======================================================================
# 5. Model component initialization function
# ======================================================================
def uly_tgm_init(
    cmp: dict,
    lamrange: ArrayLike,
    velscale: float = None,
    sampling: int = None,
    step: float = None,
    sampling_function: str = None,
) -> dict:
    r"""Initialize model component structure (cmp) for spectral fitting.

    Parameters
    ----------
    cmp : dict
        Component dictionary defined by uly_tgm function. Contains model
        information, initial parameter values, and configuration.
        Single component mode: cmp is a dictionary, not a list.
    lamrange : np.ndarray
        shape (2,)
        Wavelength range in Angstroms [lambda_min, lambda_max].
    velscale : float, optional
        Velocity scale in km/s corresponding to observed wavelength step.
        Formula: velscale = ln_step * 299792.458 (C_light)
                        = log10_step * ln(10) * 299792.458
        Default is 69.029764 km/s.
    sampling : int, optional
        Wavelength sampling mode:
        - 0: Linear wavelength sampling
        - 1: Logarithmic wavelength sampling (ln scale)
        - 2: Non-uniform wavelength sampling
    step : float, optional
        Wavelength step size.
    sampling_function : str, optional
        Interpolation method for spectral resampling.
        Options: "splinf", "cubic", "slinear", "quadratic", "linear".
        Default is "linear" interpolation.

    Returns
    -------
    cmp : dict

    Raises
    ------
    ValueError
        - If cmp is None.
        - If the model file does not exist or cannot be loaded correctly
          (for example, when init_data or the model structure s is None).
        - If lamrange is None or its length is not equal to 2.
        - If the FITS header keyword ULY_TYPE exists and is not equal
          to 'TGM'.
        - If the interpolator version (INTRP_V) and NAXIS2 are
          inconsistent with the expected format for the given version.
        - If velscale is provided while the requested sampling is not
          logarithmic (sampling != 1), causing an inconsistency between
          velocity scale and wavelength sampling.

    Notes
    -----
    - LASP uses CFI-based initialization with single component.
    - Speed of light is taken as c = 299792.458 km/s.

    Examples
    --------
    >>> from file_paths import TGM_MODEL_FILE
    >>> from uly_tgm import uly_tgm
    >>> cmp = uly_tgm(model_file=TGM_MODEL_FILE())
    >>> cmp = uly_tgm_init(
    ...     cmp,
    ...     lamrange=np.array([4000.0, 6000.0]),
    ...     velscale=69.029764
    ... )
    >>> print(cmp['npix'])
    1760

    """

    # ------------------------------------------------------------------
    # 5.1 Validate component structure
    # ------------------------------------------------------------------
    if cmp is None:
        raise ValueError("Error, cmp is None.")
    else:
        cmp["eval_fun"], init_data, s = (
            "uly_tgm_eval",
            cmp.get("init_data", None),
            uly_tgm_coef_load(),
        )
        if (init_data is None) or (s is None):
            raise ValueError("Error, model file does not exist.")
    if (lamrange is None) or (len(lamrange) != 2):
        raise ValueError("Error, lamrange is invalid.")

    # ------------------------------------------------------------------
    # 5.2 Extract header information
    # ------------------------------------------------------------------
    uly_type, NAXIS1, naxis2, ctype1, CRVAL1, CDELT1, version, calibration = (
        s["hdr"].get("ULY_TYPE", None),
        s["hdr"].get("NAXIS1", None),
        s["hdr"].get("NAXIS2", None),
        s["hdr"].get("CTYPE1", None),
        s["hdr"].get("CRVAL1", None),
        s["hdr"].get("CDELT1", None),
        s["hdr"].get("INTRP_V", None),
        s["hdr"].get("INTRP_C", None),
    )

    # ------------------------------------------------------------------
    # 5.3 Validate model type and version
    # ------------------------------------------------------------------
    if uly_type is not None:
        if uly_type.strip() != "TGM":
            raise ValueError(
                "Invalid model file, expect ULY_TYPE=TGM, get " + uly_type + "."
            )
    if version is None:
        version = 1
    else:
        if int(version) <= 2:
            if naxis2 < 23:
                raise ValueError(
                    "Invalid interpolator format for version="
                    + str(version)
                    + "naxis2="
                    + str(naxis2)
                    + " ("
                    + init_data["model"]
                    + ")."
                )
        if int(version) == 3:
            if naxis2 < 26:
                raise ValueError(
                    "Invalid interpolator format for version=" + str(version) + "."
                )

    # ------------------------------------------------------------------
    # 5.4 Extract wavelength range
    # ------------------------------------------------------------------
    wr = np.array([CRVAL1, CRVAL1 + (NAXIS1 - 1) * CDELT1])
    if ctype1 == "AWAV-LOG":
        wr = np.exp(wr)
    wr[0], wr[-1] = max([lamrange[0], wr[0]]), min([lamrange[-1], wr[-1]])
    s = uly_spect_extract(SignalIn=s, waverange=wr)

    # ------------------------------------------------------------------
    # 5.5 Update cmp
    # ------------------------------------------------------------------
    mod_samp, mod_start, mod_step, spec_coef = (
        s.get("sampling", None),
        s.get("start", None),
        s.get("step", None),
        s.get("data", None),
    )
    if sampling is not None:
        cmp["sampling"] = sampling
    else:
        cmp["sampling"] = 1
    if cmp["sampling"] == mod_samp:
        cmp["step"] = mod_step
    if velscale is not None:
        if cmp["sampling"] != 1:
            raise ValueError("Inconsistency in the arguments.")
        cmp["step"] = velscale / C_light
    if step is not None:
        cmp["step"] = step

    # ------------------------------------------------------------------
    # 5.6 Update s
    # ------------------------------------------------------------------
    npix, cmp_sampling, cmp_step = (
        s["data"].shape[1],
        cmp.get("sampling", None),
        cmp.get("step", None),
    )
    if (
        (mod_samp != cmp_sampling)
        | (mod_step != cmp_step)
        | (lamrange.all() != wr.all())
    ):
        # 5.6.1 Initial model spectrum shape; s["data"] is only used for
        # initialization
        s["data"] = np.ones(shape=(npix,))
        msk = uly_spect_get(SignalIn=s, mask_bool=True)[3]
        s["goodpix"] = np.where(msk == 1)[0]
        # 5.6.2 Linear wavelength sampling (not used here)
        # if cmp_sampling == 0:
        #     del step
        #     if(cmp_step is not None) and (cmp_step != 0):
        #         step = cmp_step
        #     s = uly_spect_linrebin(
        #         s, step, waverange=lamrange, sampling_function=sampling_function
        #     )
        # 5.6.3 Natural-logarithmic wavelength sampling
        if cmp_sampling == 1:
            del velscale
            if (cmp_step is not None) and (cmp_step != 0):
                s = uly_spect_logrebin(
                    SignalIn=s,
                    vsc=cmp_step * C_light,
                    waverange=lamrange,
                    sampling_function=sampling_function,
                )

    # ------------------------------------------------------------------
    # 5.7 Update cmp
    # ------------------------------------------------------------------
    # 5.7.1 Update wavelength information and MASK array of cmp
    if s.get("goodpix", None) is not None:
        cmp_mask = np.zeros(s["data"].shape[0], dtype=np.uint8)
        cmp_mask[s["goodpix"]] = 1
    cmp["start"], cmp["step"], cmp["npix"], cmp["sampling"], cmp["mask"] = (
        s.get("start", None),
        s.get("step", None),
        s["data"].shape[0],
        s.get("sampling", None),
        cmp_mask,
    )
    # 5.7.2 Update uly_tgm_eval-related data stored in cmp
    eval_data = {
        "spec_coef": spec_coef,
        "start": cmp["start"],
        "step": cmp["step"],
        "npix": cmp["npix"],
        "sampling": cmp["sampling"],
        "mod_samp": mod_samp,
        "mod_start": mod_start,
        "mod_step": mod_step,
        "lsf": "no_lsf",
        "version": version,
        "calibration": calibration,
    }
    cmp["eval_data"] = eval_data

    # ------------------------------------------------------------------
    # 5.8 Return initialized component structure
    # ------------------------------------------------------------------
    return cmp
