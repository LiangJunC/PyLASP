# -*- coding: utf-8 -*-
# @Time    : 11/12/2024 15.18
# @Author  : ljc
# @FileName: uly_fit_conv_weight_poly.py
# @Software: PyCharm
# Update:  2025/11/26 21:12:35


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of fitting for model spectra in LASP-CurveFit.

1.1 Purpose
-----------
Model-spectra resolution degradation, weighting, and correction factors
in LASP-CurveFit, converted from IDL uly_fit_lin.pro.

1.2 Functions
-------------
1) uly_fit_fractsolve: Solve weights between the model and the observed
   spectrum (and additive-spectrum coefficients, if enabled);
   currently a cmp only, with planned multi-component extension
   f = w_{1} * f_{1} + ... + w_{number_cmp} * f_{number_cmp}.
2) uly_fit_mulpol: Build multiplicative Legendre polynomial and
   return corrected model spectrum with updated mpoly.
3) uly_fit_weight: Return the model spectrum (or its linear
   combination with the additive component) weighted by
   uly_fit_fractsolve; currently a cmp only, with planned
   multi-component form f = w_{1} * f_{1} + ... + w_{number_cmp}
   * f_{number_cmp}.
4) uly_fit_conv_weight_poly: Return the convolved, weighted model
   spectrum with correction factors applied.

1.3 Explanation
---------------
The module performs resolution degradation, weighting, and correction
factors.
Steps:
    1) Resolution degradation: Convolve model spectra to match the
       observed spectral resolution using resolution_reduction/convol.
    2) Weighting: Solve and apply linear weights between model and
       observed spectra (and optional additive components).
    3) Correction factors: Apply multiplicative correction factors
       to account for continuum mismatch and global flux-scaling
       differences between model and observation.

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
from legendre_polynomial.mregress import mregress
from uly_tgm_eval.uly_tgm_eval import uly_tgm_eval
from resolution_reduction.convol import convol
from scipy.special import eval_legendre
from scipy.optimize import lsq_linear
import copy
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Weight coefficient calculation function
# ======================================================================
def uly_fit_fractsolve(
    npoly: int | None = None,
    a: ArrayLike | None = None,
    b: ArrayLike | None = None,
    bounds: ArrayLike | None = None,
) -> ArrayLike:
    r"""Calculate weight coefficients for a cmp.

    Parameters
    ----------
    npoly : int
        Number of additive polynomial terms.
    a : np.ndarray, optional
        shape (flux_dim, 1) or (flux_dim, npoly + 1)
        For a single component, the first npoly columns contain the
        Legendre polynomial values, and the last column contains the
        model spectrum vector. For multiple components, the first npoly
        columns contain the Legendre polynomial values, and the
        remaining columns contain the model spectrum vectors for
        the different components.
    b : np.ndarray, optional
        shape (flux_dim,)
        Observed spectrum vector.
        w * a (or a * w) = b, where w is the weight array.
    bounds : array_like, optional
        Lower and upper bounds of weights.

    Returns
    -------
    soluz : np.ndarray
        Weight array called by uly_fit_weight.

    Notes
    -----
    - For PyLASP, cmp is a single component for now (n_cmp,
      number_cmp = 1); support for multiple components (n_cmp,
      number_cmp != 1) will be added gradually.
    - Goal: Minimize ||b - w*a||² to align model and observed spectra.
    - When the solution is not unique, the recovered weights w may
      differ between IDL and Python due to different numerical
      implementations of the least-squares solver.

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 4, 6])
    >>> w = uly_fit_fractsolve(a=a, b=b)
    >>> print(w)
    [2.]
    >>> a = np.array([[1, 0, 0], [0, -1, 0], [0, 1, 1]])
    >>> b = np.array([6, 8, 11])
    >>> w = uly_fit_fractsolve(a=a, b=b, npoly=2)
    >>> print(w)
    [ 6. -8. 19.]
    >>> a = np.array([[1, 0, 0], [0, -1, 0], [0, 1, 1]])
    >>> b = np.array([6, 8, 11])
    >>> w = uly_fit_fractsolve(a=a, b=b, npoly=2, bounds=(-100, 100))
    >>> print(w)
    [ 6. -8. 19.]

    """

    # ------------------------------------------------------------------
    # 4.1 Check for None values and return default solution
    # ------------------------------------------------------------------
    if (a is None) or (b is None):
        return 1.0

    # ------------------------------------------------------------------
    # 4.2 Single component (a cmp), without additive polynomial
    # ------------------------------------------------------------------
    if (npoly is None) or (npoly == -1):
        a, b = np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1)

        # --------------------------------------------------------------
        # 4.2.1 Calculate weight coefficient
        # --------------------------------------------------------------
        soluz = np.array([np.sum(a * b) / np.sum(a**2)])

        # --------------------------------------------------------------
        # 4.2.2 Return weight array
        # --------------------------------------------------------------
        return soluz

    # ------------------------------------------------------------------
    # 4.3 Single component (a cmp), with additive polynomial
    # ------------------------------------------------------------------
    if (npoly is not None) and (a.shape[1] == npoly + 1):

        # --------------------------------------------------------------
        # 4.3.1 Unconstrained weights
        # --------------------------------------------------------------
        if bounds is None:

            # ----------------------------------------------------------
            # 4.3.1.1 Calculate weight coefficient
            # ----------------------------------------------------------
            soluz, *_ = np.linalg.lstsq(a, b, rcond=None)

            # ----------------------------------------------------------
            # 4.3.1.2 Return weight array
            # ----------------------------------------------------------
            return soluz

        # --------------------------------------------------------------
        # 4.3.2 Constrained weights
        # --------------------------------------------------------------
        if bounds is not None:

            # ----------------------------------------------------------
            # 4.3.2.1 Calculate weight coefficient
            # ----------------------------------------------------------
            all_bounds = np.empty(shape=(2, npoly + 1))
            all_bounds[0, :npoly] = -np.inf
            all_bounds[1, :npoly] = np.inf
            all_bounds[0, npoly] = bounds[0]
            all_bounds[1, npoly] = bounds[1]

            result = lsq_linear(a, b, bounds=all_bounds)

            # ----------------------------------------------------------
            # 4.3.2.2 Return weight array
            # ----------------------------------------------------------
            return result.x


# ======================================================================
# 5. Multiplicative polynomial construction function
# ======================================================================
def uly_fit_mulpol(
    bestfit: ArrayLike,
    mpoly: dict,
    SignalLog: dict,
    goodPixels: ArrayLike,
    polpen: float = 0.0,
    allow_polynomial_reduction: bool = False,
    deep_copy: bool = False,
) -> tuple[ArrayLike, dict]:
    r"""Build flux correction factor using Legendre polynomials.

    Parameters
    ----------
    bestfit : np.ndarray
        shape (flux_dim, 1)
        model spectrum (bestfit) * correction factor ≈ observed spectrum.
    mpoly : dict
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
    SignalLog : dict
        Dict Structure containing observed spectrum.
    goodPixels : np.ndarray
        Indices of valid pixels.
    polpen : float, optional
        Bias level for multiplicative polynomial. Can reduce impact
        of unimportant terms in multiplicative polynomial.
        Default 0. (no penalty).
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
    bestfitp : np.ndarray
        shape (flux_dim, 1)
        model spectrum multiplied by correction factor.
    mpoly : dict
        Updated mpoly dictionary.

    Raises
    ------
    ValueError
        - All flux values are non-positive, or some flux errors are
          negative.
        - The maximum of the correction factor <= 0.
        - The minimum of the correction factor < 0 when reduction not
          allowed.

    Notes
    -----
    - Correction factor should not be negative. Two handling methods:
      1) LASP-MPFit: Reduce polynomial degree iteratively.
      2) LASP-CurveFit: Treat as poor quality spectrum (default).
    - LASP-Adam-GPU does not provide reduce polynomial degree
      iteratively.

    Examples
    --------
    >>> bestfit = np.array([10., 12., 11., 13., 12.]).reshape(-1, 1)
    >>> err = np.full_like(bestfit, 0.5)
    >>> x = np.linspace(-1.0, 1.0, bestfit.shape[0])
    >>> leg_array = np.vstack([np.ones_like(x), x]).T
    >>> mpoly = {
    ...     "lmdegree": 1,
    ...     "mpolcoefs": np.array([1.0, 0.0]),
    ...     "poly": np.ones_like(bestfit),
    ...     "leg_array": leg_array,
    ... }
    >>> SignalLog = {"data": bestfit.ravel(), "err": err.ravel()}
    >>> goodPixels = np.arange(bestfit.shape[0])
    >>> bestfitp, mpoly_out = uly_fit_mulpol(
    ...     bestfit=bestfit,
    ...     mpoly=mpoly,
    ...     SignalLog=SignalLog,
    ...     goodPixels=goodPixels,
    ... )
    >>> print(bestfitp.shape)
    (5, 1)

    """

    # ------------------------------------------------------------------
    # 5.1 Deep copy to prevent external mpoly dictionary from being
    #     overwritten
    # ------------------------------------------------------------------
    if deep_copy:
        mpoly = copy.deepcopy(mpoly)

    # ------------------------------------------------------------------
    # 5.2 If no correction factor is applied
    # ------------------------------------------------------------------
    if mpoly["lmdegree"] == 0:
        return bestfit, mpoly

    # ------------------------------------------------------------------
    # 5.3 Extract good pixel flux and errors
    # ------------------------------------------------------------------
    good_flux, good_flux_err = (
        SignalLog["data"][goodPixels],
        SignalLog["err"][goodPixels],
    )

    # ------------------------------------------------------------------
    # 5.4 Validate flux and noise positivity
    # ------------------------------------------------------------------
    if (np.max(good_flux) <= 0) or (np.min(good_flux_err) <= 0):
        raise ValueError(
            "The spectrum flux or error has negative value, please "
            "check the spectrum quality."
        )

    # ------------------------------------------------------------------
    # 5.5 Initialize model spectrum without correction factor
    # ------------------------------------------------------------------
    if deep_copy:
        bestfitp = bestfit
    else:
        # bestfit from uly_fit_weight already includes mpoly["poly"], so
        # remove it here
        bestfitp = bestfit / mpoly["poly"]

    # ------------------------------------------------------------------
    # 5.6 Solve Legendre polynomial coefficients mpoly["mpolcoefs"]
    # ------------------------------------------------------------------
    # Obs_flux ≈ np.dot(mpoly["leg_array"], mpoly["mpolcoefs"]) *
    # bestfitp
    cmul = mpoly["leg_array"] * bestfitp
    coefs_pol, inv = mregress(
        cmul[goodPixels, :],
        good_flux,
        measure_errors=good_flux_err,
    )

    # ------------------------------------------------------------------
    # 5.7 Apply penalization to coefficients if polpen != 0
    # ------------------------------------------------------------------
    if polpen != 0.0:
        penal_pol = np.abs(coefs_pol) / (polpen * inv["sigma"])
        indices = np.where(penal_pol[1:] < 1)[0]
        if len(indices) > 0:
            pen = 1 + indices
            coefs_pol[pen] *= penal_pol[pen] ** 2

    # ------------------------------------------------------------------
    # 5.8 Update correction factor and compute its minimum/maximum
    # ------------------------------------------------------------------
    mpoly["poly"] = mpoly["leg_array"][:, 0 : mpoly["lmdegree"] + 1] @ np.transpose(
        coefs_pol
    ).reshape(-1, 1)
    minmaxpol = [np.min(mpoly["poly"][goodPixels]), np.max(mpoly["poly"][goodPixels])]
    # 5.8.1 The maximum of the correction factor must be greater than 0
    if minmaxpol[1] <= 0:
        raise ValueError(
            "Correction factor (Pseudo-continuum) all flux values are less "
            "than 0, please check the spectrum quality."
        )
    # 5.8.2 If the correction factor minimum is below 0, treat the
    # spectrum as low-quality and abort parameter inference (unless
    # reduction is allowed)
    if (minmaxpol[0] < 0) & (minmaxpol[1] > 0) & (not allow_polynomial_reduction):
        raise ValueError(
            "Correction factor (Pseudo-continuum) is less "
            "than 0, please check the spectrum quality."
        )
    # 5.8.3 If the correction factor minimum is below 0 and
    # allow_polynomial_reduction is True, iteratively reduce
    # polynomial degree
    if (minmaxpol[0] < 0) & (minmaxpol[1] > 0) & (allow_polynomial_reduction):
        lmd = mpoly["lmdegree"]
        while (np.min(mpoly["poly"]) <= 0) & (lmd > 0):
            lmd = lmd - 1
            coefs_pol, inv = mregress(
                cmul[goodPixels, 0 : lmd + 1],
                good_flux,
                measure_errors=good_flux_err,
            )
            if polpen != 0:
                penal_pol = np.abs(coefs_pol) / (polpen * inv["sigma"])
                indices = np.where(penal_pol[1:] < 1)[0]
                if len(indices) > 0:
                    pen = 1 + indices
                    coefs_pol[pen] *= penal_pol[pen] ** 2
            mpoly["poly"] = mpoly["leg_array"][:, 0 : lmd + 1] @ np.transpose(
                coefs_pol
            ).reshape(-1, 1)

    # ------------------------------------------------------------------
    # 5.9 Apply correction factor to the model spectrum
    # ------------------------------------------------------------------
    bestfitp = bestfitp * mpoly["poly"]

    # ------------------------------------------------------------------
    # 5.10 Update Legendre polynomial coefficients mpoly["mpolcoefs"]
    # ------------------------------------------------------------------
    mpoly["mpolcoefs"] = coefs_pol

    # ------------------------------------------------------------------
    # 5.11 Return the model spectrum corrected by the correction factor
    # (bestfitp) together with the updated Legendre polynomial
    # coefficients mpoly["mpolcoefs"]
    # ------------------------------------------------------------------
    return bestfitp, mpoly


# ======================================================================
# 6. Weight calculation function
# ======================================================================
def uly_fit_weight(
    SignalLog: dict,
    goodpixels: ArrayLike,
    tmp: ArrayLike,
    mpoly: dict,
    cmp: dict,
    adegree: int = -1,
    deep_copy: bool = False,
) -> tuple[ArrayLike, dict, ArrayLike]:
    r"""Determine weights for each component (LASP uses a cmp).

    Parameters
    ----------
    SignalLog : dict
        Structure containing observed spectrum.
    goodpixels : np.ndarray
        Indices of valid pixels.
    tmp : np.ndarray
        shape (flux_dim, 1)
        model spectrum.
    mpoly : dict
        Legendre polynomial dictionary structure.
    cmp : dict or list of dict
        Component structure(s). For single-component fits, this is a
        single component dictionary. For multi-component fits, this
        should be a list of component dictionaries.
    adegree : int
        Additive polynomial degree. LASP sets to -1 (no additive).
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
    bestfit : np.ndarray
        shape (flux_dim, 1)
        Weighted model spectrum.
    cmp : dict
        Updated dictionary with weight information.
    addcont : np.ndarray or None
        Additive spectrum array (None for adegree = -1).

    Notes
    -----
    - LASP sets adegree=-1 (no additive polynomial).
    - For PyLASP, cmp is a single component for now (n_cmp,
      number_cmp = 1); support for multiple components (n_cmp,
      number_cmp != 1) will be added gradually.

    Examples
    --------
    >>> obs_flux = np.array([10., 12., 11., 13., 12.])
    >>> obs_err = np.full_like(obs_flux, 0.5)
    >>> SignalLog = {"data": obs_flux, "err": obs_err}
    >>> goodpixels = np.arange(obs_flux.size)
    >>> tmp = (obs_flux * 0.9).reshape(-1, 1)
    >>> mpoly = {"poly": np.ones_like(tmp)}
    >>> cmp = {}
    >>> bestfit, cmp_out, addcont = uly_fit_weight(
    ...     SignalLog=SignalLog,
    ...     goodpixels=goodpixels,
    ...     tmp=tmp,
    ...     mpoly=mpoly,
    ...     adegree=-1,
    ...     cmp=cmp,
    ... )
    >>> print(bestfit.shape)
    (5, 1)
    >>> print(cmp_out["weight"].shape)
    (1,)
    >>> print(addcont is None)
    True

    """

    # ------------------------------------------------------------------
    # 6.1 Deep copy to prevent external cmp dictionary from being
    #     overwritten
    # ------------------------------------------------------------------
    if deep_copy:
        cmp = copy.deepcopy(cmp)

    # ------------------------------------------------------------------
    # 6.2 Get number of pixels and a cmp, good flux and errors
    # ------------------------------------------------------------------
    npix, number_cmp, good_flux, good_flux_err = (
        SignalLog["data"].shape[0],
        1,
        SignalLog.get("data")[goodpixels],
        SignalLog.get("err")[goodpixels],
    )

    # ------------------------------------------------------------------
    # 6.3 Obs flux ≈ correction factor * model spectrum
    # ------------------------------------------------------------------
    c = mpoly["poly"].reshape(-1, 1) * tmp

    # ------------------------------------------------------------------
    # 6.4 Additive polynomial
    # ------------------------------------------------------------------
    if adegree != -1:
        npoly = adegree + 1
        if adegree >= 0:
            c1 = np.repeat(mpoly["poly"].reshape(-1, 1), npoly, axis=1)
            x = 2.0 * np.arange(npix) / npix - 1.0
            for j in range(adegree + 1):
                # c1[:, j] *= eval_legendre(j, x)
                c1[:, j] = c1[0, j] * eval_legendre(j, x)
            c = np.hstack([c1, c])
    else:
        adegree = -1

    # ------------------------------------------------------------------
    # 6.5 Prepare good pixel data
    # ------------------------------------------------------------------
    a = c[goodpixels, :]
    for j in range(0, (adegree + 1) + number_cmp - 1 + 1):
        a[:, j] /= good_flux_err

    # ------------------------------------------------------------------
    # 6.6 Compute additive polynomial and weights
    # ------------------------------------------------------------------
    if adegree == -1:
        cmp["weight"] = uly_fit_fractsolve(
            a=a,
            b=good_flux / good_flux_err,
        )
        bestfit = c * cmp["weight"]
        addcont = np.zeros_like(bestfit)
    else:
        wght = uly_fit_fractsolve(
            a=a,
            b=good_flux / good_flux_err,
            npoly=npoly,
            # bounds=cmp["lim_weig"],
        )
        bestfit = c @ wght.reshape(-1, 1)
        cmp["weight"] = wght[adegree + 1 :]
        addcont = c[:, 0 : adegree + 1] @ wght[0 : adegree + 1]

    # ------------------------------------------------------------------
    # 6.7 Check invalid weights for single component and set them to 0
    # (since multiple components are not yet supported, cmp["weight"]
    # is a single scaling factor)
    # ------------------------------------------------------------------
    # This section is directly adapted from the IDL implementation.
    # A more efficient Pythonic implementation can be added later, e.g.:
    # if (np.isfinite(cmp["weight"])) | (np.abs(cmp["weight"]) > 1e36) | (np.abs(cmp["weight"]) < 1e-36):
    #   cmp["weight"] = 0
    nan_idx = np.where(~np.isfinite(cmp["weight"]) | (np.abs(cmp["weight"]) > 1e36))[0]
    if len(nan_idx) > 0:
        cmp["weight"] = 0
    zeros_idx = np.where(np.abs(cmp["weight"]) < 1e-36)[0]
    if len(zeros_idx) > 0:
        cmp["weight"] = 0

    # ------------------------------------------------------------------
    # 6.8 Return model spectrum after multiplying by weights,
    # together with the updated cmp and additive polynomial
    # ------------------------------------------------------------------
    if adegree == -1:
        return bestfit, cmp, None
    else:
        return bestfit, cmp, addcont


# ======================================================================
# 7. Main fitting function
# ======================================================================
def uly_fit_conv_weight_poly(
    cmp: dict,
    SignalLog: dict,
    goodPixels: ArrayLike,
    voff: float,
    adegree: int = -1,
    par_losvd: ArrayLike | None = None,
    mpoly: dict | None = None,
    polpen: float = 0.0,
    sampling_function: str | None = None,
    allow_polynomial_reduction: bool = False,
    deep_copy: bool = False,
) -> tuple[ArrayLike, dict]:
    r"""Fit linear coefficients and return best-fit result.

    Parameters
    ----------
    cmp : dict or list of dict
        Component structure(s). For single-component fits, this is a
        single component dictionary. For multi-component fits, this
        should be a list of component dictionaries.
    SignalLog : dict
        Structure containing observed spectrum.
    goodPixels : np.ndarray
        Indices of valid pixels.
    voff : float
        Velocity offset between the spectrum to analyse and the
        model (km/s).
    adegree : int
        Additive polynomial degree. LASP sets to -1 (no additive).
    par_losvd : np.ndarray, optional
        LOSVD parameters array [cz, sigma, h3, h4, h5, h6]. LASP uses
        only cz and sigma for resolution reduction.
    mpoly : dict, optional
        Legendre polynomial dictionary structure.
    polpen : float, optional
        Bias level for multiplicative polynomial. Can reduce impact
        of unimportant terms in multiplicative polynomial.
        Default 0. (no penalty).
    sampling_function : str, optional
        Interpolation method: 'splinf', 'cubic', 'slinear', 'quadratic',
        'linear'. Default is 'linear'.
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
    bestfit : np.ndarray
        shape (flux_dim, 1)
        Best-fit model spectrum: convolved, weighted, correction factor.
    mpoly : dict
        Updated Legendre polynomial dictionary.

    Raises
    ------
    ValueError
        - Weight is negative.

    Notes
    -----
    - For PyLASP, cmp is a single component for now (n_cmp,
      number_cmp = 1); support for multiple components (n_cmp,
      number_cmp != 1) will be added gradually.

    Examples
    --------
    No minimal runnable example is provided here, because this function
    requires a fully configured spectral model (model coefficients, cmp
    structure, etc.) to generate the model spectrum.

    """

    # ------------------------------------------------------------------
    # 7.1 Deep copy to prevent external mpoly dictionary from being
    #     overwritten
    # ------------------------------------------------------------------
    if deep_copy:
        mpoly = copy.deepcopy(mpoly)

    # ------------------------------------------------------------------
    # 7.2 Extract spectrum dimensions and valid pixel data
    # ------------------------------------------------------------------
    npix = SignalLog.get("data").shape[0]

    # ------------------------------------------------------------------
    # 7.3 Initialize polynomial dictionary if not provided
    # ------------------------------------------------------------------
    if mpoly is None:
        mpoly = {
            "lmdegree": 50,
            "mpolcoefs": np.ones(1),
            "poly": np.ones(shape=(npix, 1)),
        }

    # ------------------------------------------------------------------
    # 7.4 Call uly_tgm_eval to generate model spectrum
    # ------------------------------------------------------------------
    para_values = np.array([param["value"] for param in cmp["para"]])
    models = uly_tgm_eval(
        eval_data=cmp["eval_data"],
        para=para_values,
        sampling_function=sampling_function,
    )

    # ------------------------------------------------------------------
    # 7.5 Degrade model spectrum resolution to match observed
    # ------------------------------------------------------------------
    if par_losvd is not None:
        # 7.5.1 Compute Gaussian kernel
        losvd_mu, losvd_sigma = par_losvd[0], par_losvd[1]
        vel, sigma_pix = voff + losvd_mu, np.max([losvd_sigma, 0.1])
        dx1, dx2 = (
            np.ceil(np.abs(voff) + np.abs(losvd_mu) + 5.0 * losvd_sigma),
            (npix - 1.0) / 2.0,
        )
        dx = np.min([dx1, dx2])
        x = dx - np.arange(int(2 * dx + 1))
        w = (x - vel) / sigma_pix
        w2 = w * w
        losvd = np.where(
            np.abs(w) > 5.0,
            0.0,
            np.exp(-0.5 * w2) / (np.sqrt(2.0 * np.pi) * sigma_pix),
        )

        # 7.5.2 Non-Gaussian kernel
        nherm = len(par_losvd) - 2
        if nherm > 0:
            poly = 1.0 + par_losvd[2] / np.sqrt(3.0) * (w * (2.0 * w2 - 3.0))
            if nherm > 1:
                poly += par_losvd[3] / np.sqrt(24.0) * (w2 * (4.0 * w2 - 12.0) + 3.0)
            if nherm > 2:
                poly += (
                    par_losvd[4] / np.sqrt(60.0) * (w * (w2 * (4.0 * w2 - 20.0) + 15.0))
                )
            if nherm > 3:
                poly += (
                    par_losvd[5]
                    / np.sqrt(720.0)
                    * (w2 * (w2 * (8.0 * w2 - 60.0) + 90.0) - 15.0)
                )
            losvd *= poly

        # 7.5.3 Normalize kernel
        losvd /= np.sum(losvd)
        # 7.5.4 Convolve to degrade model spectrum resolution to match
        # observed spectrum
        models = convol(models, losvd)

    # ------------------------------------------------------------------
    # 7.6 Apply weight to model spectrum
    # ------------------------------------------------------------------
    bestfitw, cmp, addcont = uly_fit_weight(
        SignalLog,
        goodPixels,
        models,
        mpoly,
        adegree=adegree,
        cmp=cmp,
        deep_copy=deep_copy,
    )
    # 7.6.1 Weights must be positive; otherwise flux becomes non-positive
    posw = np.where(cmp["weight"] > 0)[0]
    if len(posw) == 0:
        raise ValueError("All weights are not positive.")

    # ------------------------------------------------------------------
    # 7.7 Apply correction factor to model spectrum
    # ------------------------------------------------------------------
    bestfit, mpoly = uly_fit_mulpol(
        bestfitw,
        mpoly=mpoly,
        SignalLog=SignalLog,
        goodPixels=goodPixels,
        allow_polynomial_reduction=allow_polynomial_reduction,
        polpen=polpen,
        deep_copy=deep_copy,
    )

    # ------------------------------------------------------------------
    # 7.8 Set the constant-term Legendre coefficient to 1
    # ------------------------------------------------------------------
    # 7.8.1 Rescale polynomial and weight by the first Legendre
    # coefficient. Enforce the first Legendre coefficient to be 1
    # (for single component, this has almost no impact on the inferred
    # parameters)
    if deep_copy:
        mpoly = copy.deepcopy(mpoly)
        cmp = copy.deepcopy(cmp)
    mpoly["poly"] /= mpoly["mpolcoefs"][0]
    cmp["weight"] *= mpoly["mpolcoefs"][0]
    # 7.8.2 Normalize Legendre coefficients so that the first term
    # is 1
    mpoly["mpolcoefs"] /= mpoly["mpolcoefs"][0]

    # ------------------------------------------------------------------
    # 7.9 Output final model spectrum and Legendre polynomial structure
    # ------------------------------------------------------------------
    return bestfit, mpoly
