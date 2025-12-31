# -*- coding: utf-8 -*-
# @Time    : 2025/1/4 21:51
# @Author  : ljc
# @FileName: ulyss_fit_a_cmp.py
# @Software: PyCharm
# Update:  2025/11/26 20:50:48


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of stellar parameter inference for
    LASP-CurveFit.

1.1 Purpose
-----------
Implement numerical optimization methods from scipy library to infer
stellar parameters and parameter errors, converted from IDL uly_fit.pro.

1.2 Functions
-------------
1) uly_fit_pparse: Parse parameters into par_losvd and cmp, update
   LOSVD parameters and stellar parameters.
2) uly_fitfunc_init: Initialize Legendre polynomial coefficients
   for correction factors.
2) uly_fitfunc: This function is iteratively optimized by scipy
   optimizer to minimize flux differences.
3) uly_fit_a_cmp: Main fitting function using scipy numerical
   optimization, including Clean and NoClean strategies.

1.3 Explanation
---------------
It is used for inferring the stellar parameters of one spectrum,
including both the Clean and No Clean strategies, while also allowing
visualization of the flux residuals.

1.4 Notes
---------
- Currently, only a single component (cmp, number_cmp = 1) is supported.
  In future versions, linear combinations of multiple components (n_cmp,
  number_cmp != 1) will be added to decompose the observed spectrum as:
     f = w_{1} * f_{1} + ... + w_{number_cmp} * f_{number_cmp},
  where f_{i} belongs to the i-th component (cmp_{i}), and each
  component corresponds to a dictionary structure containing wavelength
  information, initial parameter guesses, and other metadata.
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
from scipy.special import eval_legendre
from uly_fit.uly_fit_conv_weight_poly import uly_fit_conv_weight_poly
from uly_read_lms.uly_spect_get import uly_spect_get
import time
from scipy.optimize import curve_fit
from clean_outliers.clean import clean_outliers
from uly_fit.robust_sigma import robust_sigma
import copy
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Parse parameters into par_losvd and cmp
# ======================================================================
def uly_fit_pparse(
    pars: ArrayLike,
    cmp: dict | list,
    deep_copy: bool = False,
    kmoment: int = 2,
    error: ArrayLike | None = None,
) -> tuple[list, dict]:
    r"""Parse pars content into par_losvd and cmp structures.

    Parameters
    ----------
    pars : np.ndarray
        Parameter array to be inferred. The first kmoment elements are
        LOSVD parameters, and others are stellar parameters. In LASP,
        the LOSVD parameters are cz and sigma, used to broaden the model
        spectrum and calculate the radial velocity (Rv).
    cmp : dict or list of dict
        Component structure(s). For single-component fits, this is a
        single component dictionary. For multi-component fits, this
        should be a list of component dictionaries.
    deep_copy : bool, optional
        Controls whether cmp and poly are deep-copied during the
        iterative optimization. If False (default), uly_fitfunc
        operates in-place on the external poly["poly"] and
        cmp["weight"], reproducing the behavior of the original IDL
        implementation. If True, uly_fitfunc works on internal deep
        copies created at each call, so that the external poly["poly"]
        and cmp["weight"] remain unchanged throughout the fit.
    kmoment : int, optional
        Order of Gauss-Hermite moments. Set to 2 for [cz, sigma],
        4 for [cz, sigma, h3, h4], 6 for [cz, sigma, h3, h4, h5, h6].
        LASP sets to 2.
    error : np.ndarray, optional
        Error array for parameters to be tested.

    Returns
    -------
    par_losvd : list
        Updated LOSVD parameters for broadening model spectrum.
    cmp : dict
        Cmp dictionary with updated stellar parameters and errors.

    Raises
    ------
    ValueError
        - If pars is None.
        - kmoment <= 1.

    Notes
    -----
    - For PyLASP, cmp is a single component for now (n_cmp,
      number_cmp = 1); support for multiple components (n_cmp,
      number_cmp != 1) will be added gradually.

    Examples
    --------
    Single-component example
    ~~~~~~~~~~~~~~~~~~~~~~~~
    >>> import numpy as np
    >>> pars = np.array([30.0, 150.0, 5800.0, 4.3, -0.1])
    >>> error = np.array([0.0, 0.0, 50.0, 0.1, 0.05])
    >>> cmp = {"para": [{"value": 0.0}, {"value": 0.0}, {"value": 0.0}]}
    >>> par_losvd, cmp_out = uly_fit_pparse(pars, cmp, kmoment=2, error=error)
    >>> print(par_losvd.tolist())
    [30.0, 150.0]
    >>> values = [p["value"] for p in cmp_out[0]["para"]]
    >>> print(values)
    [5800.0, 4.3, -0.1]

    """

    # ------------------------------------------------------------------
    # 4.1 Deep copy to prevent external cmp dictionary from being
    #     overwritten
    # ------------------------------------------------------------------
    if deep_copy:
        cmp = copy.deepcopy(cmp)

    # ------------------------------------------------------------------
    # 4.2 Normalize to a component list: support possible
    #     multi-component cases
    # ------------------------------------------------------------------
    cmp_list = [cmp] if isinstance(cmp, dict) else list(cmp)

    # ------------------------------------------------------------------
    # 4.3 Extract LOSVD parameters used for spectral broadening
    # ------------------------------------------------------------------
    if pars is None:
        raise ValueError("Parameter array is empty.")
    if (kmoment is None) or (kmoment <= 1):
        raise ValueError("The least kmoment must be >= 2.")
    par_losvd = pars[0:kmoment]

    # ------------------------------------------------------------------
    # 4.4 Store stellar parameters and errors into cmp structure:
    #     compatible with future multi-component support
    # ------------------------------------------------------------------
    indx = 0
    for cmp_number in range(len(cmp_list)):
        il = len(cmp_list[cmp_number]["para"])
        for i in range(il):
            cmp_list[cmp_number]["para"][i]["value"] = pars[
                kmoment + indx : kmoment + il + indx
            ][i]
            if error is not None:
                cmp_list[cmp_number]["para"][i]["error"] = error[
                    kmoment + indx : kmoment + il + indx
                ][i]
        indx = indx + il

    # ------------------------------------------------------------------
    # 4.5 Return LOSVD parameters and cmp with updated stellar
    #     parameters and their errors
    # ------------------------------------------------------------------
    return par_losvd, cmp_list


# ======================================================================
# 5. Initialize Legendre polynomial structures for
#    uly_fit_conv_weight_poly
# ======================================================================
def uly_fitfunc_init(
    spec: dict,
    mpoly: dict | None = None,
    mdegree: int = 50,
    deep_copy: bool = False,
) -> dict:
    r"""Initialize Legendre polynomial structures for parameter inference.

    Parameters
    ----------
    spec : dict
        Observed spectrum dictionary structure.
    mpoly : dict or None, optional
        Dictionary containing: {lmdegree, mpolcoefs, poly, leg_array}.
        - lmdegree: Maximum Legendre degree (n=50 for LASP).
        - mpolcoefs: Coefficients (c0, c1, ..., cn),
          shape (lmdegree+1,).
        - leg_array: Legendre values matrix, (1, P1(lambda_i),
          P2(lambda_i), ..., Pn(lambda_i)),
          shape (flux_dim, lmdegree+1).
        - poly: correction factor = leg_array * mpolcoefs,
          shape (flux_dim, 1).
        - Obs flux ≈ correction factor * model flux ≈ (1, c1, c2, ...,
          cn) * (1, P1(lambda_i), P2(lambda_i), ..., Pn(lambda_i))
          * model flux
    mdegree : int, optional
        Legendre polynomial degree (mpoly["lmdegree"]). LASP uses 50.
    deep_copy : bool, optional
        Controls whether cmp and poly are deep-copied during the
        iterative optimization. If False (default), uly_fitfunc
        operates in-place on the external poly["poly"] and
        cmp["weight"], reproducing the behavior of the original IDL
        implementation. If True, uly_fitfunc works on internal deep
        copies created at each call, so that the external poly["poly"]
        and cmp["weight"] remain unchanged throughout the fit.

    Returns
    -------
    mpoly : dict
        Initialized Legendre polynomial related coefficients.

    Raises
    ------
    ValueError
        - If spec, or mdegree is not specified.

    Notes
    -----
    - LASP uses 50-degree Legendre polynomial for continuum correction.
    - The polynomial is evaluated on normalized x ∈ [-1, 1] for
      numerical stability.

    Examples
    --------
    >>> import numpy as np
    >>> spec = {"data": np.ones(5)}
    >>> mpoly = uly_fitfunc_init(spec=spec, mpoly=None, mdegree=3)
    >>> print(mpoly["lmdegree"])
    3
    >>> print(mpoly["leg_array"].shape)
    (5, 4)

    """

    # ------------------------------------------------------------------
    # 5.1 Deep copy to prevent external mpoly dictionary from being
    #     overwritten
    # ------------------------------------------------------------------
    if deep_copy:
        mpoly = copy.deepcopy(mpoly)

    # ------------------------------------------------------------------
    # 5.2 Validate that mandatory inputs are provided
    # ------------------------------------------------------------------
    if spec is None:
        raise ValueError("Spectrum dictionary is not specified.")

    # ------------------------------------------------------------------
    # 5.3 Get the number of pixels in the target spectrum
    # ------------------------------------------------------------------
    npix = spec.get("data").shape[0]

    # ------------------------------------------------------------------
    # 5.4 Initialize mpoly if needed
    # ------------------------------------------------------------------
    if mpoly is not None:
        if (mpoly["lmdegree"] != mdegree) | (mpoly["poly"].shape[0] != npix):
            mpoly = None
    if mpoly is None:
        mpoly = {
            "lmdegree": mdegree,
            "mpolcoefs": np.ones(shape=(mdegree + 1)),
            "poly": np.ones(shape=(npix, 1)),
            "leg_array": eval_legendre(
                np.arange(mdegree + 1),
                (2.0 * np.arange(npix) / npix - 1.0).reshape(-1, 1),
            ),
        }

    # ------------------------------------------------------------------
    # 5.5 Return Legendre polynomial structures and cmp
    # ------------------------------------------------------------------
    return mpoly


# ======================================================================
# 6. Loss function iteratively optimized by scipy
# ======================================================================
def uly_fitfunc(
    pars: ArrayLike,
    signalLog: dict,
    cmp: dict,
    goodpixels: ArrayLike,
    voff: float,
    kpen: float = 0.0,
    polpen: float = 0.0,
    adegree: int = -1,
    kmoment: int = 2,
    outpixels: ArrayLike | None = None,
    sampling_function: str | None = None,
    mpoly: dict | None = None,
    allow_polynomial_reduction: bool = False,
    deep_copy: bool = False,
) -> tuple[ArrayLike, ArrayLike, dict, dict]:
    r"""Calculate flux residuals between model and observed spectrum.

    Parameters
    ----------
    pars : np.ndarray
        Parameter array to be inferred. The first kmoment elements are
        LOSVD parameters, and others are stellar parameters. In LASP,
        the LOSVD parameters are cz and sigma, used to broaden the model
        spectrum and calculate the radial velocity (Rv).
    signalLog : dict
        Structure containing observed spectrum.
    cmp : dict or list of dict
        Component structure(s). For single-component fits, this is a
        single component dictionary. For multi-component fits, this
        should be a list of component dictionaries.
    goodpixels : np.ndarray
        Indices of good pixels.
    voff : float, optional
        Velocity offset between the spectrum to analyse and the
        model (km/s).
    kpen : float, optional
        Penalty parameter biasing (h3, h4, ...) measurements toward
        zero unless they significantly reduce flux fitting error.
        Default 0 (no penalty). LASP default is 0.
    polpen : float, optional
        Bias level for multiplicative polynomial. Can reduce impact
        of unimportant terms in multiplicative polynomial.
        Default 0. (no penalty).
    adegree : int, optional
        Degree of additive Legendre polynomial for correcting model
        spectrum shape. Default no additive polynomial.
        LASP sets to -1 (disabled).
    kmoment : int, optional
        Order of Gauss-Hermite moments. Set to 2 for [cz, sigma],
        4 for [cz, sigma, h3, h4], 6 for [cz, sigma, h3, h4, h5, h6].
        LASP sets to 2.
    outpixels : np.ndarray or None, optional
        Output pixels (same as goodpixels if not specified).
    sampling_function : str or None, optional
        Interpolation method. Options: "splinf", "cubic", "slinear",
        "quadratic", "linear". Default "linear".
    mpoly : dict or None, optional
        Legendre polynomial dictionary structure.
    allow_polynomial_reduction : bool, optional
        Allow polynomial degree reduction if correction factor < 0.
        Default False (treat as poor quality spectrum).
    deep_copy : bool, optional
        Controls whether cmp and poly are deep-copied during the
        iterative optimization. If False (default), uly_fitfunc
        operates in-place on the external poly["poly"] and
        cmp["weight"], reproducing the behavior of the original IDL
        implementation. If True, uly_fitfunc works on internal deep
        copies created at each call, so that the external poly["poly"]
        and cmp["weight"] remain unchanged throughout the fit.

    Returns
    -------
    err : np.ndarray
        shape (flux_dim, 1)
        Flux difference between model and observed spectrum.
    bestfit : np.ndarray
        shape (flux_dim, 1)
        model spectrum.
    mpoly : dict
        Multiplicative Legendre polynomial structure.
    cmp : dict
        Updated cmp structure from uly_fit_pparse.

    Raises
    ------
    ValueError
        If pars or signalLog is None.

    Notes
    -----
    - LASP sets flux error to 1, so it's not strictly chi-square.

    Examples
    --------
    No minimal runnable example is provided here, because this function
    requires a fully configured spectral model (model coefficients, cmp
    structure, etc.) to generate the model spectrum.

    """

    # ------------------------------------------------------------------
    # 6.1 Validate that the pars and signalLog is provided
    # ------------------------------------------------------------------
    if pars is None:
        raise ValueError("Pars is not specified.")
    if signalLog is None:
        raise ValueError("SignalLog is not specified.")

    # ------------------------------------------------------------------
    # 6.2 If outpixels is not provided, use goodpixels as output mask
    # ------------------------------------------------------------------
    if outpixels is None:
        outpixels = goodpixels

    # ------------------------------------------------------------------
    # 6.3 Parse LOSVD and stellar parameters from the global pars array
    # ------------------------------------------------------------------
    par_losvd, cmp_list = uly_fit_pparse(
        pars=pars, cmp=cmp, kmoment=kmoment, deep_copy=deep_copy
    )

    # ------------------------------------------------------------------
    # 6.4 Evaluate model spectrum at the given parameter values
    #     (note: uly_fit_a_cmp is for a single component and cmp is a
    #     single-component list, so pass cmp[0])
    # ------------------------------------------------------------------
    bestfit, mpoly = uly_fit_conv_weight_poly(
        adegree=adegree,
        voff=voff,
        par_losvd=par_losvd,
        cmp=cmp_list[0],
        goodPixels=goodpixels,
        sampling_function=sampling_function,
        SignalLog=signalLog,
        mpoly=mpoly,
        polpen=polpen,
        allow_polynomial_reduction=allow_polynomial_reduction,
        deep_copy=deep_copy,
    )

    # ------------------------------------------------------------------
    # 6.5 Extract model flux and observed flux/error on output pixels
    # ------------------------------------------------------------------
    bestfit, flux, flux_err = (
        bestfit.reshape(-1),
        signalLog.get("data").reshape(-1),
        signalLog.get("err").reshape(-1),
    )

    # ------------------------------------------------------------------
    # 6.6 Compute weighted residuals between observation and model
    # ------------------------------------------------------------------
    fit_err = (flux[outpixels] - bestfit[outpixels]) / flux_err[outpixels]

    # ------------------------------------------------------------------
    # 6.7 Apply penalty on Gauss-Hermite moments when kmoment > 2
    # ------------------------------------------------------------------
    if (kmoment > 2) and (kpen != 0):
        sigma = robust_sigma(fit_err, zero=True)
        fit_err = fit_err + kpen * sigma * np.sqrt(np.sum(pars[2:kmoment] ** 2))

    # ------------------------------------------------------------------
    # 6.8 Return residuals, model spectrum, and updated mpoly/cmp
    # ------------------------------------------------------------------
    return fit_err, bestfit, mpoly, cmp_list


# ======================================================================
# 7. Infer stellar parameters using scipy numerical optimization
# ======================================================================
def uly_fit_a_cmp(
    signalLog: dict,
    cmp: dict | list[dict],
    parinfo: dict,
    clean: bool = False,
    allow_polynomial_reduction: bool = False,
    sampling_function: str | None = None,
    deep_copy: bool = False,
    adegree: int = -1,
    mdegree: int = 50,
    kmoment: int = 2,
    kpen: float = 0.0,
    polpen: float = 0.0,
    quiet: bool = True,
    full_output: bool = False,
    plot_fitting: bool = False,
) -> tuple[float, float, float, float, float, float, float, float, float, float, int]:
    r"""Infer stellar parameters using scipy numerical optimization.

    Parameters
    ----------
    signalLog : dict
        Structure containing observed spectrum.
    cmp : dict or list of dict
        Component structure(s). For single-component fits, this is a
        single component dictionary. For multi-component fits, this
        should be a list of component dictionaries.
    parinfo: dict
        Merge LOSVD and stellar parameter-information lists.
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
    adegree : int, optional
        Degree of additive Legendre polynomial. Default no additive
        polynomial. LASP sets to -1.
    mdegree : int, optional
        Legendre polynomial degree. LASP default is 50.
    kmoment : int, optional
        Order of Gauss-Hermite moments. Set to 2 for [cz, sigma],
        4 for [cz, sigma, h3, h4], 6 for [cz, sigma, h3, h4, h5, h6].
        LASP sets to 2.
    kpen : float, optional
        This parameter biases the (h3,h4,...) measurements towards zero
        (Gaussian LOSVD) unless their inclusion significantly decreases
        the error in the fit. LASP default is 0 (no penalty).
    polpen : float, optional
        Bias level for multiplicative polynomial. Can reduce impact
        of unimportant terms in multiplicative polynomial.
        Default 0. (no penalty).
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
        Time used for inference (seconds).
    loss : float
        Root mean square error of fitting flux residuals.
    clean_number : int
        Number of cleaned pixels.

    Raises
    ------
    ValueError
        - If kmoment not in [0, 2, 4, 6], if kguess missing elements,
        - if wavelength range too small, if no good pixels left, if
          goodpixels out of range, if parameter guess outside limits.

    Notes
    -----
    - Unconstrained fits use the Levenberg–Marquardt algorithm (LM) by
      default. If bounds are provided, LASP-CurveFit switches to a
      trust-region reflective method (TRF).
    - Cannot guarantee 100% identical results between Python and IDL,
      but can achieve consistent results due to: (1) computation
      precision differences, (2) numerical optimization method
      differences.
    - LASP-CurveFit may have -9999 anomalies (poor quality spectra)
      due to: (1) max iterations reached without convergence,
      (2) abnormal spectrum without Clean mode, (3) unreasonable
      initial parameter values, (4) singular matrix, (5) negative
      correction factors for low quality spectra, (6) unknown reasons.
    - The computation order of resc0 is slightly different from the IDL
      implementation. In IDL, the calculation of resc0 (see Sect.
      7.2.3.6.2) is placed before Sect. 7.2.3.2. Since in each call to
      uly_fitfunc the input mpoly["poly"] and cmp["weight"] change with
      the previous iteration, and are re-optimized and updated in
      uly_fit_conv_weight_poly to obtain the optimal mpoly["poly"]
      and cmp["weight"], the impact on the final parameter inference
      results is negligible. It is also worth noting that when
      deep_copy is set to True, this ordering has no effect at all.

    Examples
    --------
    No minimal runnable example is provided here, because this function
    requires a fully configured spectral model (model coefficients, cmp
    structure, etc.) to generate the model spectrum.

    """

    # ------------------------------------------------------------------
    # 7.1 Data preparation
    # ------------------------------------------------------------------
    # 7.1.1 Initialize flux, wavelength, mask and related information
    flux, flux_err, obs_msk, mod_mask, obs_start, obs_step, mod_start, C, npix = (
        signalLog.get("data"),
        signalLog.get("err"),
        uly_spect_get(signalLog, mask_bool=True)[3],
        cmp.get("mask"),
        signalLog.get("start"),
        signalLog.get("step"),
        cmp["start"],
        299792.458,
        cmp.get("npix"),  # flux.shape[0]
    )
    if mod_mask is not None:
        msk = obs_msk * mod_mask
    else:
        msk = obs_msk
    goodPixels0 = np.where(msk == 1)[0]
    if (len(goodPixels0) == 0) or (np.max(goodPixels0) > npix - 1):
        raise ValueError("No good pixels left or goodpixels out of range.")
    # 7.1.2 Velocity scale in km/s corresponding to wavelength step.
    voff, velScale = (mod_start - obs_start) * C, C * obs_step
    # 7.1.3 Convert parameter limits into curve_fit bounds format
    bounds = np.zeros(shape=(2, len(parinfo)))
    bounds[0, :], bounds[1, :] = (
        np.array([parinfo[i]["limits"][0] for i in range(len(parinfo))]),
        np.array([parinfo[i]["limits"][1] for i in range(len(parinfo))]),
    )
    if np.any(np.isinf(bounds)):
        bounds = (-np.inf, np.inf)

    # ------------------------------------------------------------------
    # 7.2 Parameter inference stage
    # ------------------------------------------------------------------
    # 7.2.1 Decide whether to enable Clean strategy, copy good pixels,
    #       and set optimizer tolerances
    goodPixels, nclean = copy.deepcopy(goodPixels0), 0
    if clean is True:
        nclean = 10
    ftol = 1e-5 if nclean == 0 else 1e-2
    xtol = 1e-10 if nclean == 0 else 1e-8
    (
        first_stage_outer_number,
        second_stage_outer_number,
        third_stage_outer_number,
    ) = (
        0,
        0,
        0,
    )
    # 7.2.2 Check if parameter optimization is needed
    nlin_free = 0
    for k in range(len(parinfo)):
        if parinfo[k]["fixed"] == 0:
            nlin_free += 1

    # ------------------------------------------------------------------
    # 7.2.3 Iterative inference (NoClean loops once, Clean loops
    #       nclean times)
    # ------------------------------------------------------------------
    time0 = time.time()
    for j in range(nclean + 1):
        # 7.2.3.1 Initialize polynomial, cmp dictionary, and parameters
        #         at each Clean iteration
        mpoly = uly_fitfunc_init(spec=signalLog, mdegree=mdegree, deep_copy=deep_copy)
        value = [parinfo[i]["value"] for i in range(len(parinfo))]

        # 7.2.3.2 Build the objective function for scipy optimizer
        def min_fun(temp, *pars) -> np.ndarray:
            fit_flux_err = uly_fitfunc(
                pars=pars,
                kpen=kpen,
                polpen=polpen,
                cmp=cmp,
                adegree=adegree,
                signalLog=signalLog,
                mpoly=mpoly,
                goodpixels=goodPixels,
                sampling_function=sampling_function,
                voff=voff / velScale,
                kmoment=kmoment,
                allow_polynomial_reduction=allow_polynomial_reduction,
                deep_copy=deep_copy,
            )[0]

            return fit_flux_err

        # 7.2.3.3 If there are free parameters, start the optimization
        if nlin_free > 0:
            # 7.2.3.3.1 If full_output is True, return full diagnostics
            # Note on optimization method:
            #   - Unconstrained fits use the Levenberg–Marquardt
            #     algorithm (LM) by default.
            #   - If bounds are provided, LASP-CurveFit switches to a
            #     trust-region reflective method (TRF).
            if full_output is True:
                res, res_cov, infodict, errmsg, ier = curve_fit(
                    min_fun,
                    xdata=[],
                    ydata=0.0,
                    p0=value,
                    bounds=bounds,
                    ftol=ftol,
                    xtol=xtol,
                    full_output=True,
                )
                loss = np.sqrt(np.mean(infodict["fvec"] ** 2))
            # 7.2.3.3.2 If full_output is False, return parameter
            if full_output is False:
                res, res_cov = curve_fit(
                    min_fun,
                    xdata=[],
                    ydata=0.0,
                    p0=value,
                    bounds=bounds,
                    ftol=ftol,
                    xtol=xtol,
                )
                loss = -9999
            # 7.2.3.3.3 Compute derived parameters and their errors
            result_std = np.diag(res_cov) ** 0.5
            Rv, Rv_s, Teff, logg, FeH = (
                C * (np.exp(obs_step * res[0]) - 1),
                C * (np.exp(obs_step * res[1]) - 1),
                np.exp(res[-3]),
                res[-2],
                res[-1],
            )
            Rv_err, Rv_s_err, Teff_err, logg_err, FeH_err = (
                result_std[0] * velScale,
                result_std[1] * velScale,
                Teff * result_std[-3],
                result_std[-2],
                result_std[-1],
            )

            # 7.2.3.3.4 If quiet = False, print inferred parameters
            if not quiet:
                if clean is True:
                    strategy_name = f"==================== Clean strategy (Number {j + 1}) =====================\n"
                    title_name = f"Clean strategy (Number {j + 1})\n"
                else:
                    strategy_name = f"========================= NoClean strategy =========================\n"
                    title_name = f"NoClean strategy\n"
                print(
                    "====================================================================\n"
                    + strategy_name
                    + "====================================================================\n"
                    "--------------------------------------------------------------------\n"
                    "3. Inferred stellar atmospheric parameters\n"
                    "--------------------------------------------------------------------\n"
                    "  Parameter  =   BestFit ±   Error\n"
                    f"(1). {'RV':<6}  = {Rv:>8.2f}  ± {Rv_err:>6.2f}\n"
                    f"(2). {'RV_s':<6}  = {Rv_s:>8.2f}  ± {Rv_s_err:>6.2f}\n"
                    f"(3). {'Teff':<6}  = {Teff:>8.2f}  ± {Teff_err:>6.2f}\n"
                    f"(4). {'log g':<6}  = {logg:>8.2f}  ± {logg_err:>6.2f}\n"
                    f"(5). {'[Fe/H]':<6}  = {FeH:>8.2f}  ± {FeH_err:>6.2f}\n"
                    "--------------------------------------------------------------------"
                )
                if full_output is True:
                    if ier <= 4:
                        inference_succeeded_or_not = (
                            "(5). Parameter inference succeeded."
                        )
                    else:
                        inference_succeeded_or_not = "(5). Parameter inference failed."
                    print(
                        "--------------------------------------------------------------------\n"
                        "4. Convergence status of the fit\n"
                        "--------------------------------------------------------------------\n"
                        f"(1). Number of function evaluations: {infodict['nfev']}\n"
                        f"(2). Final RMS of flux residuals: {np.round(loss, 6)}\n"
                        # f"(2). Final RMS of flux residuals: {infodict['fvec']}\n"
                        f"(3). Fit status flag: {ier}\n"
                        f"(4). Termination message: {errmsg}\n"
                        + inference_succeeded_or_not
                    )
                    if clean is False:
                        print(
                            "--------------------------------------------------------------------"
                        )
                    # 7.2.3.3.4.1 If plot_fitting is True, visualize
                    #             flux residuals
                    if plot_fitting is True:
                        import matplotlib.pyplot as plt
                        from matplotlib.ticker import MaxNLocator

                        plt.style.use("classic")
                        fig = plt.figure(figsize=(30, 8), dpi=300)
                        fig.subplots_adjust(
                            left=0.08, right=0.98, top=0.88, bottom=0.12
                        )
                        plt.plot(
                            np.exp(obs_start + np.arange(npix) * obs_step)[goodPixels],
                            infodict["fvec"],
                            label="Residual flux",
                        )
                        if j > 0:
                            plt.scatter(
                                np.exp(
                                    obs_start
                                    + np.setdiff1d(goodPixels0, goodPixels) * obs_step
                                ),
                                ((signalLog["data"] - bestfit) / flux_err)[
                                    np.setdiff1d(goodPixels0, goodPixels)
                                ],
                                s=200,
                                c="r",
                                marker="*",
                                label="Outlier",
                            )
                        plt.xlabel("Wavelength [Å]", fontsize=30)
                        plt.ylabel("Residual flux", fontsize=30)
                        plt.title(title_name, fontsize=30)
                        plt.xlim(
                            np.array(
                                [
                                    np.exp(obs_start),
                                    np.exp(obs_start + (npix - 1) * obs_step),
                                ]
                            )
                        )
                        plt.tick_params(
                            top="on", right="on", which="both", labelsize=25
                        )
                        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
                        plt.legend(loc="best", fontsize=25)
                        plt.show()

            # 7.2.3.3.5 Update cmp using the inferred parameters
            cmp = uly_fit_pparse(
                pars=res,
                cmp=cmp,
                error=result_std[0:5],
                kmoment=kmoment,
            )[1]

        # 7.2.3.4 If all parameters are fixed, skip optimization
        else:
            resc0 = uly_fitfunc(
                pars=value,
                kpen=kpen,
                polpen=polpen,
                cmp=cmp,
                adegree=adegree,
                signalLog=signalLog,
                mpoly=mpoly,
                goodpixels=goodPixels,
                sampling_function=sampling_function,
                voff=voff / velScale,
                kmoment=kmoment,
                allow_polynomial_reduction=allow_polynomial_reduction,
                deep_copy=deep_copy,
            )[0]
            loss = np.sqrt(np.mean(resc0**2))
            Rv, Rv_s, Teff, logg, FeH, Rv_err, Rv_s_err, Teff_err, logg_err, FeH_err = (
                value[0],
                value[1],
                value[-3],
                value[-2],
                value[-1],
                0,
                0,
                0,
                0,
                0,
            )

        # 7.2.3.5 If currently at nclean-th Clean iteration, break loop
        if j == nclean:
            break

        # --------------------------------------------------------------
        # 7.2.3.6 Clean strategy to remove anomalous flux
        # --------------------------------------------------------------
        # 7.2.3.6.1 Save pixel list from previous Clean for
        #           comparison in next Clean round
        goodOld = copy.deepcopy(goodPixels)
        # 7.2.3.6.2 Calculate flux residuals (best_param = value)
        resc0 = uly_fitfunc(
            pars=value,
            kpen=kpen,
            polpen=polpen,
            cmp=cmp,
            adegree=adegree,
            signalLog=signalLog,
            mpoly=mpoly,
            goodpixels=goodPixels,
            sampling_function=sampling_function,
            voff=voff / velScale,
            kmoment=kmoment,
            allow_polynomial_reduction=allow_polynomial_reduction,
            deep_copy=deep_copy,
        )[0]
        rbst0 = np.std(resc0, ddof=1)
        # 7.2.3.6.3 Calculate flux residuals (best_param = res)
        resc = np.zeros(npix)
        (
            resc[goodPixels0],
            bestfit,
        ) = uly_fitfunc(
            pars=res,
            kpen=kpen,
            polpen=polpen,
            cmp=cmp,
            adegree=adegree,
            signalLog=signalLog,
            mpoly=mpoly,
            goodpixels=goodPixels,
            sampling_function=sampling_function,
            voff=voff / velScale,
            kmoment=kmoment,
            outpixels=goodPixels0,
            allow_polynomial_reduction=allow_polynomial_reduction,
            deep_copy=deep_copy,
        )[:2]

        # 7.2.3.6.4 Call the clean strategy
        (
            rbst_sig,
            goodPixels,
            first_stage_outer_number,
            second_stage_outer_number,
            third_stage_outer_number,
        ) = clean_outliers(
            j=j,
            npix=npix,
            bestfit=bestfit,
            flux_err=flux_err,
            resc=resc,
            goodPixels=goodPixels,
            goodPixels0=goodPixels0,
            first_stage_outer_number=first_stage_outer_number,
            second_stage_outer_number=second_stage_outer_number,
            third_stage_outer_number=third_stage_outer_number,
            quiet=quiet,
        )
        # 7.2.3.6.4.1 If pixel list from previous Clean is same as
        #             current Clean, exit loop
        if np.array_equal(goodOld, goodPixels):
            if not quiet:
                print(
                    f"(6). Final number of clipped outliers: {first_stage_outer_number}+{second_stage_outer_number}+{third_stage_outer_number}"
                    f" out of {len(goodPixels0)}\n"
                    "--------------------------------------------------------------------"
                )
            break
        # 7.2.3.6.4.2 Adjust optimizer settings
        ftol = 1e-5
        if j + 2 < nclean:
            ftol = 1e-3
        elif j + 1 < nclean:
            ftol = 1e-4

        # 7.2.3.6.4.3 Assess whether ratio between current optimization
        #           effect and Clean strategy impact is reasonable by
        #           analyzing flux residual changes
        rbst1 = np.std(resc[goodPixels], ddof=1)
        if ((rbst_sig**2) / (rbst0 * rbst1)) < 1.5:
            for i in range(len(parinfo)):
                parinfo[i]["value"] = res[i]
        elif j + 1 < nclean:
            ftol = 1e-2

    # 7.2.4 Calculate time used to infer optimal stellar parameters
    #       for 1 spectrum
    time1 = time.time()
    used_time = time1 - time0
    if not quiet:
        print(
            "####################################################################\n"
            "###################### Elapsed time: ",
            round(used_time, 2),
            " s" + " ######################\n"
            "####################################################################",
        )
        print("--------------------------------------------------------------------")

    # ------------------------------------------------------------------
    # 7.3 Output stellar parameters and errors, time used, root mean
    #       square error of spectral flux residuals
    # ------------------------------------------------------------------
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
        first_stage_outer_number + second_stage_outer_number + third_stage_outer_number,
    )
