# -*- coding: utf-8 -*-
# @Time    : 10/12/2024 14.31
# @Author  : ljc
# @FileName: ulyss.py
# @Software: PyCharm
# Update:  2025/11/26 21:27:11


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python wrapper configuring inputs and calling uly_fit_a_cmp for
    stellar parameter inference.

1.1 Purpose
-----------
Configure input variables for uly_fit_a_cmp and call uly_fit_a_cmp to
obtain stellar parameters for the observed spectrum, converted from
IDL ulyss.pro.

1.2 Functions
-------------
1) uly_cmp_read: Returns cmp initial structure information.
2) ulyss: Calls uly_fit_a_cmp function to obtain stellar parameters
   and error inference values for the observed spectrum.

1.3 Explanation
---------------
The ulyss function:
    1) Reads the model and builds the cmp structure.
    2) Log-rebins the observed spectrum to ln(lambda) space.
    3) Aligns the model and observed wavelength ranges.
    4) Constructs or validates the flux-error spectrum so that
       $\chi^{2}$ values and weights can be computed reliably.
    5) Constructs LOSVD and stellar-parameter information.
    6) Calls uly_fit_a_cmp and returns radial velocity, Teff,
       log g, [Fe/H] and their errors, together with timing and
       residual statistics.

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
from uly_tgm.uly_tgm import uly_tgm
from uly_read_lms.uly_spect_extract import uly_spect_extract
from WRS.uly_spect_logrebin import uly_spect_logrebin
from uly_read_lms.uly_spect_get import uly_spect_get
from uly_fit.uly_fit_init import uly_fit_init
from uly_fit.uly_makeparinfo import uly_get_infer_params_set
from uly_fit.uly_fit_a_cmp import uly_fit_a_cmp
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
    t_limits: list | None = None,
    l_limits: list | None = None,
    z_limits: list | None = None,
) -> dict:
    """Read model file and return cmp initial structure.

    Parameters
    ----------
    model_file : str
        model file path.
    t_guess : float or list, optional
        Initial guess for Teff in Kelvin. If None, defaults to 7500.0 K.
    l_guess : float or list, optional
        Initial guess for log g. If None, defaults to 3.0.
    z_guess : float or list, optional
        Initial guess for [Fe/H] in dex. If None, defaults to -0.5.
    t_limits : list, optional
        Effective temperature inference range in Kelvin.
        If None, defaults to [-np.inf, np.inf].
    l_limits : list, optional
        Surface gravity (log g) inference range.
        If None, defaults to [-np.inf, np.inf].
    z_limits : list, optional
        Metallicity ([Fe/H]) inference range in dex.
        If None, defaults to [-np.inf, np.inf].

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
    # LASP uses ULY_TYPE == "TGM"
    uly_type = header.get("ULY_TYPE", None)
    if uly_type is None:
        raise ValueError("uly_type not found in header. Please check the model file.")
    if uly_type == "TGM":
        return uly_tgm(
            model_file=model_file,
            t_guess=t_guess,
            l_guess=l_guess,
            z_guess=z_guess,
            t_limits=t_limits,
            l_limits=l_limits,
            z_limits=z_limits,
        )
    elif uly_type == "SSP":
        return uly_ssp(model_file=model_file)
    else:
        return uly_star(model_file=model_file)


# ======================================================================
# 5. Call uly_fit_a_cmp function to obtain stellar parameters and
#    error inference values for the observed spectrum
# ======================================================================
def ulyss(
    inspectr: dict,
    model_file: str,
    clean: bool = False,
    allow_polynomial_reduction: bool = False,
    sampling_function: str | None = None,
    deep_copy: bool = False,
    t_guess: float | dict | None = None,
    l_guess: float | dict | None = None,
    z_guess: float | dict | None = None,
    t_limits: list | None = None,
    l_limits: list | None = None,
    z_limits: list | None = None,
    snr: float | None = None,
    adegree: int = -1,
    mdegree: int = 50,
    velscale: float | None = None,
    kmoment: int = 2,
    cz_guess: float | None = None,
    sigma_guess: float | None = None,
    polpen: float = 0.0,
    klimits: ArrayLike | None = None,
    kfix: ArrayLike | None = None,
    kpen: float = 0.0,
    quiet: bool = True,
    full_output: bool = False,
    plot_fitting: bool = False,
) -> tuple[float, float, float, float, float, float, float, float, float, float, int]:
    """Infer stellar parameters for observed spectrum.

    Parameters
    ----------
    inspectr : dict
        Structure containing observed spectrum.
    model_file : str
        model file path.
    clean : bool, optional
        Whether to iteratively detect and clip outliers in model vs
        observed spectrum flux residuals. Default False.
    allow_polynomial_reduction : bool, optional
        Allow polynomial degree reduction if correction factor < 0.
        Default False (treat as poor quality spectrum).
    sampling_function : str or None, optional
        Interpolation method. Options: "splinf", "cubic", "slinear",
        "quadratic", "linear". Default "linear".
    deep_copy : bool, optional
        Controls whether cmp and poly are deep-copied during the
        iterative optimization. If False (default), uly_fitfunc
        operates in-place on the external poly["poly"] and
        cmp["weight"], reproducing the behavior of the original IDL
        implementation. If True, uly_fitfunc works on internal deep
        copies created at each call, so that the external poly["poly"]
        and cmp["weight"] remain unchanged throughout the fit.
    t_guess : float or list, optional
        Initial guess for Teff in Kelvin. If None, defaults to 7500.0 K.
    l_guess : float or list, optional
        Initial guess for log g. If None, defaults to 3.0.
    z_guess : float or list, optional
        Initial guess for [Fe/H] in dex. If None, defaults to -0.5.
    t_limits : list, optional
        Effective temperature inference range in Kelvin.
        If None, defaults to [-np.inf, np.inf].
    l_limits : list, optional
        Surface gravity (log g) inference range.
        If None, defaults to [-np.inf, np.inf].
    z_limits : list, optional
        Metallicity ([Fe/H]) inference range in dex.
        If None, defaults to [-np.inf, np.inf].
    snr : float or None, optional
        Signal-to-noise ratio of observed spectrum.
    adegree : int, optional
        Degree of additive Legendre polynomial. Default no additive
        polynomial. LASP sets to -1.
    mdegree : int, optional
        Legendre polynomial degree. LASP default is 50.
    velscale : float or None, optional
        Velocity per step in ln wavelength space (km/s).
        Note: velscale = ln_step * c = log10_step * c * ln(10).
    kmoment : int, optional
        Order of Gauss-Hermite moments. Set to 2 for [cz, sigma],
        4 for [cz, sigma, h3, h4], 6 for [cz, sigma, h3, h4, h5, h6].
        LASP sets to 2.
    cz_guess : float or None, optional
        Initial guess for LOSVD first parameter.
    sigma_guess : float or None, optional
        Initial guess for LOSVD second parameter.
    polpen : float, optional
        Bias level for multiplicative polynomial. Can reduce impact
        of unimportant terms in multiplicative polynomial.
        Default 0. (no penalty).
    klimits : np.ndarray or None, optional
        Upper and lower bounds for parameters to test.
    kfix : np.ndarray or None, optional
        Whether to fix LOSVD parameters. 1 fixes corresponding LOSVD
        parameter during minimization.
    kpen : float, optional
        This parameter biases the (h3,h4,...) measurements towards zero
        (Gaussian LOSVD) unless their inclusion significantly decreases
        the error in the fit. LASP default is 0 (no penalty).
    quiet : bool, optional
        Whether to suppress screen messages. False prints inference
        process, True doesn't.
    full_output : bool, optional
        Whether to return all fitting results and information.
        True returns all, False returns only Rv, Teff, log g, [Fe/H]
        and errors. full_output = True only takes effect when
        quiet = False. It is recommended to set quiet = False and
        full_output = True for single sample detection.
    plot_fitting : bool, optional
        Whether to plot spectral fitting residual. True plots,
        False doesn't. plot_fitting = True only takes effect when
        quiet = False and full_output = True. It is recommended to
        set quiet = False, full_output = True, and plot_fitting = True
        for single sample detection.

    Returns
    -------
    Rv : float
        Radial velocity (km/s).
    Teff : float
        Effective temperature (K).
    logg : float
        Surface gravity (dex).
    FeH : float
        Metallicity [Fe/H] (dex).
    Rv_err : float
        Radial velocity error (km/s).
    Teff_err : float
        Effective temperature error (K).
    logg_err : float
        Surface gravity error (dex).
    FeH_err : float
        Metallicity error (dex).
    used_time : float
        Time used to infer parameters for 1 spectrum (seconds).
    loss : float
        Root mean square error of fitting flux residuals.
    clean_number : int
        Number of cleaned pixels.

    Raises
    ------
    ValueError
        - If inspectr is not specified.
        - If both cmp and model_file are specified.
        - If shift_guess, lmin, or err_sp specified when inspectr
          is a structure.
        - If kfix length exceeds kmoment.

    Notes
    -----
    - used_time indicates time used to infer parameters for 1 spectrum
      (seconds).
    - loss indicates root mean square error of fitting flux residuals.
    - Can add or reduce output quantities based on task requirements.

    Examples
    --------
    No minimal runnable example is provided here, because this function
    requires a fully configured spectral model (model coefficients, cmp
    structure, etc.) and observed spectrum dictionary.

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
        t_limits=t_limits,
        l_limits=l_limits,
        z_limits=z_limits,
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
    mod_mask, mod_start, model_step, npix = (
        cmp.get("mask"),
        cmp["start"],
        cmp["step"],
        cmp.get("npix"),
    )
    final_wave_range = np.exp(
        [
            mod_start + 0.5 * model_step,
            mod_start + (npix - 1.5) * model_step,
        ]
    )
    SignalLog = uly_spect_extract(
        SignalIn=SignalLog,
        waverange=final_wave_range,
        overwrite=True,
    )

    # ==================================================================
    # 5.5 Settings related to parameters to be inferred
    # ==================================================================
    (
        obs_msk,
        obs_start,
        obs_step,
    ) = (
        uly_spect_get(SignalLog, mask_bool=True)[3],
        SignalLog.get("start"),
        SignalLog.get("step"),
    )
    voff = (mod_start - obs_start) * C_light
    parinfo = uly_get_infer_params_set(
        cmp=cmp,
        cz_guess=cz_guess,
        sigma_guess=sigma_guess,
        velscale=velscale,
        kmoment=kmoment,
        kfix=kfix,
        klimits=klimits,
        obs_step=obs_step,
        npix=npix,
        voff=voff,
        deep_copy=deep_copy,
    )
    if kpen is None:
        if mod_mask is not None:
            msk = obs_msk * mod_mask
        else:
            msk = obs_msk
        goodPixels0 = np.where(msk == 1)[0]
        kpen = 0.7 * np.sqrt(500 / len(goodPixels0))

    # ==================================================================
    # 5.6 Print initialization configuration if quiet = False
    # ==================================================================
    if not quiet:
        if allow_polynomial_reduction:
            allow_polynomial_reduction_str = (
                f"(2). Polynomial degree reduction allowed; when correction factors\n"
                f"     becomes negative, the polynomial degree will be reduced\n"
            )
        else:
            allow_polynomial_reduction_str = (
                f"(2). Polynomial degree reduction not allowed; if correction factors\n"
                f"     becomes negative, the parameter inference will fail\n"
            )
        if adegree == -1:
            adegree_str = "(3). No additive polynomial is used\n"
        else:
            adegree_str = f"(3). Degree of additive polynomial: {adegree}\n"
        lamrange = np.exp(
            [
                obs_start,
                obs_start + (SignalLog["data"].size - 1) * obs_step,
            ]
        )
        print(
            "--------------------------------------------------------------------\n"
            "1. Parameter settings\n"
            "--------------------------------------------------------------------\n"
            f"(1). Degree of Legendre multiplicative polynomial: {mdegree}\n"
            + allow_polynomial_reduction_str
            + adegree_str
            + f"(4). Initial guesses (Teff, log g, [Fe/H]): "
            + str(np.round(np.exp(cmp["para"][0]["guess"]), 2))
            + " K, "
            + str(np.round(cmp["para"][1]["guess"], 2))
            + " dex, "
            + str(np.round(cmp["para"][2]["guess"], 2))
            + " dex"
            + "\n"
            "--------------------------------------------------------------------\n"
            "--------------------------------------------------------------------\n"
            "2. Arguments passed to uly_fit_a_cmp\n"
            "--------------------------------------------------------------------"
        )
        items = [
            (
                "(1). Wavelength range",
                f"{np.round(lamrange[0], 5)}, {np.round(lamrange[1], 5)} [Å]",
            ),
            ("(2). Velscale", f"{np.round(velscale, 5)} [km/s]"),
            (
                "(3). Number of independent pixels",
                int(SignalLog["data"].size / SignalLog["dof_factor"]),
            ),
            ("(4). Number of fitted pixels", SignalLog["data"].size),
            ("(5). DOF factor", SignalLog["dof_factor"]),
        ]
        max_key_len = max(len(k) for k, _ in items)
        for key, val in items:
            print(f"{key.ljust(max_key_len)} : {val}")
        print("--------------------------------------------------------------------")

    # ==================================================================
    # 5.7 Call uly_fit_a_cmp to obtain optimal parameters for
    #     observed spectrum
    # ==================================================================
    (
        Rv,
        Teff,
        logg,
        FeH,
        Rv_err,
        Teff_err,
        logg_err,
        FeH_err,
        used_time,
        loss,
        clean_number,
    ) = uly_fit_a_cmp(
        signalLog=SignalLog,
        cmp=cmp,
        kmoment=kmoment,
        parinfo=parinfo,
        kpen=kpen,
        adegree=adegree,
        mdegree=mdegree,
        polpen=polpen,
        clean=clean,
        allow_polynomial_reduction=allow_polynomial_reduction,
        deep_copy=deep_copy,
        sampling_function=sampling_function,
        quiet=quiet,
        full_output=full_output,
        plot_fitting=plot_fitting,
    )

    # ==================================================================
    # 5.8 Return Rv, Teff, log g, [Fe/H] with errors and so on
    # ==================================================================
    return (
        Rv,
        Teff,
        logg,
        FeH,
        Rv_err,
        Teff_err,
        logg_err,
        FeH_err,
        used_time,
        loss,
        clean_number,
    )
