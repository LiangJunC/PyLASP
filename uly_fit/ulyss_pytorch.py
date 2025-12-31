# -*- coding: utf-8 -*-
# @Time    : 2025/1/13 21:22
# @Author  : ljc
# @FileName: ulyss_pytorch.py
# @Software: PyCharm
# Update:  2025/11/26 21:33:14


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Prepare PyTorch-ready spectra for LASP-Adam-GPU.

1.1 Purpose
-----------
Provide a lightweight wrapper to extract the information needed by
LASP-Adam-GPU and store it in a format suitable for saving as
.pt files. The implementation is adapted from the ULySS (IDL/CPU)
version but tailored for batch processing with LASP-Adam-GPU.

1.2 Functions
-------------
1) uly_cmp_read:
       Read the model FITS file and return the initial component
       structure (cmp).
2) ulyss:
       Prepare the observed spectrum and model information, and
       return the arrays needed to build LASP-Adam-GPU training /
       inference datasets.

1.3 Explanation
---------------
The ulyss function:
    1) Reads the model and builds the cmp structure (if not provided).
    2) Extracts flux and (optionally) log-rebins the observed spectrum.
    3) Validates or constructs flux-error estimates if needed.
    4) Aligns model and observed wavelength ranges so that the
       model slightly covers the observed spectrum.
    5) Computes linear wavelength grids, integration borders, and
       scaling factors.
    6) Extracts good-pixel indices and polynomial spectral-emulator
       coefficients.

The returned arrays can then be converted to torch.Tensor and saved as
.pt files (for example via the helper in data_to_pt/data_to_pt.py),
enabling grouped optimization with the Adam optimizer.

1.4 Notes
---------
- This is a PyTorch-oriented preparation module; it does not perform
  parameter inference itself but only gathers and returns the required
  inputs.
- The design follows the same cmp structure and wavelength handling
  as the CPU-based ULySS/PyLASP implementation to keep the GPU and
  CPU pipelines consistent.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
from astropy.io import fits
from uly_tgm.uly_tgm import uly_tgm
from uly_read_lms.uly_spect_extract import uly_spect_extract
import numpy as np
from WRS.uly_spect_logrebin import uly_spect_logrebin
from uly_read_lms.uly_spect_get import uly_spect_get
from uly_fit.uly_fit_init import uly_fit_init
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions and constants for better code readability
# ======================================================================
ArrayLike = np.ndarray
C_light = 299792.458  # Speed of light


# ======================================================================
# 4. Read model file and return cmp initial structure information
# ======================================================================
def uly_cmp_read(
    model_file: str,
    t_guess: float | None = None,
    l_guess: float | None = None,
    z_guess: float | None = None,
) -> dict:
    """Read model file and return cmp initial structure.

    Parameters
    ----------
    model_file : str
        model file path.
    t_guess : float or None, optional
        Initial value for Teff.
    l_guess : float or None, optional
        Initial value for log g.
    z_guess : float or None, optional
        Initial value for [Fe/H].

    Returns
    -------
    cmp : dict
        Cmp initial dictionary structure information.

    Raises
    ------
    TypeError
        - If model_file is not specified.
    IOError
        - If error occurs reading the file.
    ValueError
        - If ULY_TYPE is not TGM (only TGM is currently supported).

    Notes
    -----
    - LASP uses ULY_TYPE="TGM".
    - Other model types such as SSP and STAR are reserved for future
      versions if needed.
    - For non-TGM models, please refer to the IDL code.

    Examples
    --------
    >>> from file_paths import TGM_MODEL_FILE
    >>> cmp = uly_cmp_read(
    ...     model_file=TGM_MODEL_FILE(),
    ...     t_guess=7000,
    ...     l_guess=3,
    ...     z_guess=-0.5
    ... )
    >>> print(cmp["init_fun"])
    uly_tgm_init

    """

    # ------------------------------------------------------------------
    # 4.1 Validate that model_file
    # ------------------------------------------------------------------
    if model_file is None:
        raise TypeError("model_file must be specified.")

    # ------------------------------------------------------------------
    # 4.2 Read model file
    # ------------------------------------------------------------------
    try:
        with fits.open(model_file) as hdul:
            header = hdul[0].header
    except Exception as e:
        raise IOError(f"Error reading file: {str(e)}.")

    # ------------------------------------------------------------------
    # 4.3 Check ULY_TYPE key in header and branch accordingly
    # ------------------------------------------------------------------
    # 4.3.1 LASP uses ULY_TYPE == "TGM"
    uly_type = header.get("ULY_TYPE", None)
    if uly_type == "TGM":
        return uly_tgm(
            model_file=model_file,
            t_guess=t_guess,
            l_guess=l_guess,
            z_guess=z_guess,
        )
    else:
        # 4.3.2 Other model types are not supported
        raise ValueError(
            f"Current ulyss_pytorch implementation only supports TGM models. "
            f"Invalid ULY_TYPE: {uly_type}."
        )


# ======================================================================
# 5. Prepare spectra for LASP-Adam-GPU and return .pt-ready arrays
# ======================================================================
def ulyss(
    inspectr: dict,
    model_file: str,
    sampling_function: str | None = None,
    t_guess: float | None = None,
    l_guess: float | None = None,
    z_guess: float | None = None,
    snr: float | None = None,
    velscale: float | None = None,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    r"""Prepare spectral information for saving to .pt files.

    Parameters
    ----------
    inspectr : dict
        Structure containing observed spectrum.
    model_file : str
        model file path.
    sampling_function : str or None, optional
        Interpolation method. Options: "splinf", "cubic", "slinear",
        "quadratic", "linear". Default "linear".
    t_guess : float or None, optional
        Initial value for Teff. Currently only a single scalar
        initial value is supported.
    l_guess : float or None, optional
        Initial value for log g. Currently only a single scalar
        initial value is supported.
    z_guess : float or None, optional
        Initial value for [Fe/H]. Currently only a single scalar
        initial value is supported.
    snr : float or None, optional
        Signal-to-noise ratio of observed spectrum.
    velscale : float or None, optional
        Velocity per step in ln wavelength space (km/s).
        Note: velscale = ln_step * c = log10_step * c * ln(10).

    Returns
    -------
    ELODIE_wave : np.ndarray
       Wavelength of the ELODIE model spectra.
    borders : np.ndarray
       Integration borders of the model spectra.
    NewBorders : np.ndarray
       Integration borders of the observed spectrum in the model
       wavelength frame.
    flat : np.ndarray
       Scaling factor proportional to
       Δλ₁ / (λ₂(i+1) - λ₂(i)), as used in the LASP-Adam-GPU
       method (see Step 3 in the paper).
    lamrange : np.ndarray
       Wavelength of the trimmed observed spectrum used for fitting.
    flux_obs : np.ndarray
       Flux values of the observed spectrum within the fitting range.
    goodpix : np.ndarray
       Indices of good pixels to be used during fitting. This can be
       combined with official MASK fields to construct a binary mask.
    spec_coef : np.ndarray
       Polynomial spectral-emulator coefficient array from the model
       (e.g. ELODIE-based polynomial coefficients).

    """

    # ==================================================================
    # 5.1. Initialization
    # ==================================================================
    # 5.1.1 Initialize model component cmp
    cmp = uly_cmp_read(
        model_file,
        t_guess=t_guess,
        l_guess=l_guess,
        z_guess=z_guess,
    )
    # 5.1.2 Initialize observed spectrum structure: check if resampling
    # to ln log-equal spacing is needed Spectral fitting uses ln
    # log-equal spacing, so resampling is required when
    # inspectr["sampling"] is 0 or 2
    SignalLog = uly_spect_logrebin(
        SignalIn=inspectr,
        vsc=velscale,
        sampling_function=sampling_function,
        overwrite=True,
    )

    # 5.1.3 Extract observed spectrum flux and related information
    flux, goodpix, flux_err, obs_start, obs_step = (
        SignalLog.get("data"),
        SignalLog.get("goodpix"),
        SignalLog.get("err"),
        SignalLog.get("start"),
        SignalLog.get("step"),
    )

    # ==================================================================
    # 5.2 Ensure positive errors if flux error exists or SNR is provided
    # ==================================================================
    if (flux_err is None) & (snr is not None):
        mean_error = np.mean(flux[goodpix]) / snr
        if not np.isfinite(mean_error).all():
            raise ValueError("Cannot compute the mean of the signal.")
        flux_err = mean_error * np.ones_like(flux)
    if flux_err is not None:
        negerr, poserr = (
            np.where(flux_err[goodpix] <= 0)[0],
            np.where(flux_err[goodpix] > 0)[0],
        )
        if len(poserr) == 0:
            raise ValueError("The flux noise is invalid.")
        if (len(negerr) > 0) and (len(poserr) > 0):
            SignalLog["err"][goodpix][negerr] = np.min(flux_err[goodpix][poserr])
        weight = 1 / (SignalLog["err"][goodpix]) ** 2
        large_weight = np.where(weight > 100 * np.mean(weight))[0]
        if len(large_weight) > 0:
            raise ValueError(
                "Some pixels have more than 100 times the average weight, so the flux noise may be invalid."
            )
    if flux_err is None:
        flux_err = np.ones_like(flux)
    SignalLog["err"] = flux_err

    # ==================================================================
    # 5.3 Update cmp based on observed spectrum information
    # ==================================================================
    # 5.3.1 Get observed spectrum wavelength range
    lamrange = uly_spect_get(SignalIn=SignalLog, waverange_bool=True)[0]
    velscale = obs_step * C_light
    # 5.3.2 Update model spectrum wavelength based on observed spectrum
    cmp = uly_fit_init(
        cmp,
        lamrange=lamrange,
        velscale=velscale,
    )

    # ==================================================================
    # 5.4 Trim observed spectrum to narrower wavelength range
    # ==================================================================
    # Calculate wavelength range with half-step offset to ensure safe
    # grid overlap when observed spectrum exceeds model coverage
    (
        mod_npix,
        mod_start,
        mod_step,
        final_mod_npix,
        final_mod_start,
        final_mod_step,
        spec_coef,
    ) = (
        cmp["eval_data"]["spec_coef"].shape[1],
        cmp["eval_data"]["mod_start"],
        cmp["eval_data"]["mod_step"],
        cmp["npix"],
        cmp["start"],
        cmp["step"],
        cmp["eval_data"]["spec_coef"][:23, :, :],
    )
    final_wave_range = np.exp(
        [
            final_mod_start + 0.5 * final_mod_step,
            final_mod_start + (final_mod_npix - 1.5) * final_mod_step,
        ]
    )
    SignalLog = uly_spect_extract(
        SignalIn=SignalLog,
        waverange=final_wave_range,
        overwrite=True,
    )

    # ==================================================================
    # 5.5 Compute wavelength grids, borders, scaling, and coefficients
    # ==================================================================
    (
        flux_obs,
        obs_start,
        obs_step,
        obs_npix,
        goodpix,
    ) = (
        SignalLog["data"],
        SignalLog["start"],
        SignalLog["step"],
        SignalLog["data"].size,
        SignalLog["goodpix"],
    )
    lamrange, ELODIE_wave, borders, NewBorders, flat = (
        np.exp([obs_start + np.arange(obs_npix) * obs_step])[0],
        mod_start + np.arange(mod_npix) * mod_step,
        mod_start + (np.arange(mod_npix + 1) - 0.5) * mod_step,
        np.exp(
            [final_mod_start + (np.arange(final_mod_npix + 1) - 0.5) * final_mod_step]
        )[0],
        np.exp(final_mod_start + np.arange(final_mod_npix) * obs_step)
        * obs_step
        / mod_step,
    )

    # ==================================================================
    # 5.6 Return prepared arrays (NumPy, PyTorch-ready)
    # ==================================================================
    return (
        ELODIE_wave,
        borders,
        NewBorders,
        flat,
        lamrange,
        flux_obs,
        goodpix,
        spec_coef,
    )
