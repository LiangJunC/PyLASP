# -*- coding: utf-8 -*-
# @Time    : 08/12/2024 12.16
# @Author  : ljc
# @FileName: xrebin.py
# @Software: PyCharm
# Update:  2025/11/10 16:04:40


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of spectral resampling for LASP-CurveFit.

1.1 Purpose
-----------
Implement spectral data resampling using "cumulative-interpolate-
differentiate" method based on IDL xrebin.pro core algorithm.

1.2 Functions
-------------
1) xrebin: Called by 'WRS/uly_spect_logrebin.py'.

1.3 Explanation
---------------
The xrebin function performs spectral resampling using "cumulative-
interpolate-differentiate" method. Complete resampling includes 5
steps, where steps 1 and 5 are calculated in WRS/uly_spect_logrebin.py.
Steps:
    1) WRS/uly_spect_logrebin.py calculates input/output wavelength bin
       edges (xin, xout).
    2) Accumulate input flux values (cumulative sum).
    3) Interpolate cumulative spectrum onto new wavelength grid.
    4) Differentiate interpolated result to get differential spectrum.
    5) Pass to 'WRS/uly_spect_logrebin.py' to determine whether to
       convert the "cumulative-interpolate-differentiate" differential
       spectrum to "integrate-interpolate-derivative" spectrum.

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
from scipy.interpolate import interp1d, CubicSpline
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Wavelength resampling function
# ======================================================================
def xrebin(
    xin: ArrayLike,
    yin: ArrayLike,
    xout: ArrayLike,
    sampling_function: str = None,
) -> ArrayLike:
    r"""Resample spectrum using "cumulative-interpolate-differentiate".

    Parameters
    ----------
    xin : np.ndarray
        shape (n+1,)
        Input wavelength bin edges. n+1 edges define n bins.
    yin : np.ndarray
        shape (n,) or (n, 1)
        Input flux. n flux values correspond to n bins.
    xout : np.ndarray
        shape (m+1,)
        Output wavelength bin edges. m+1 edges define m bins.
    sampling_function : str, optional
        Interpolation method. Options: "splinf", "cubic", "slinear",
        "quadratic", "linear". Default: "linear".

    Returns
    -------
    yout : np.ndarray
        shape (m,) or (m, 1)
        Resampled flux. m flux values correspond to m output bins.

    Raises
    ------
    TypeError
        - If 'xin' 'yin' and 'xout' are not ArrayLike.
    ValueError
        - If 'xin' and 'xout' are not 1-D arrays, 'yin' is interpreted
          as multiple spectra and its dimensions become inconsistent
          with the wavelength dimension.

    Notes
    -----
    - Default interpolation method is linear for best balance between
      accuracy and computational efficiency.
    - Invalid sampling_function values will automatically fall back to
      linear interpolation.
    - For LAMOST spectra, interpolation method choice has minimal
      impact on fitting results.

    Examples
    --------
    >>> xin = np.linspace(4000, 5000, 1001)
    >>> yin = np.sin(xin[:-1] / 100) + 2
    >>> xout = np.linspace(4000, 5000, 501)
    >>> yout = xrebin(xin, yin, xout, sampling_function="linear")
    >>> print(yout.shape)
    (500,)

    """

    # ------------------------------------------------------------------
    # 4.1 Input validation
    # ------------------------------------------------------------------
    # 4.1.1 Type check
    if not isinstance(xin, ArrayLike):
        raise TypeError(
            f"xin must be a numpy array, got {type(xin).__name__}. "
            f"Use np.array() to convert."
        )
    if not isinstance(yin, ArrayLike):
        raise TypeError(
            f"yin must be a numpy array, got {type(yin).__name__}. "
            f"Use np.array() to convert."
        )
    if not isinstance(xout, ArrayLike):
        raise TypeError(
            f"xout must be a numpy array, got {type(xout).__name__}. "
            f"Use np.array() to convert."
        )

    # 4.1.2 Dimension check
    if xin.ndim != 1 or xout.ndim != 1:
        raise ValueError("'xin' and 'xout' must be 1-D arrays of bin edges.")
    if yin.ndim == 1:
        yin = yin.reshape(-1, 1)
    elif yin.ndim == 2:
        if yin.shape[1] != 1:
            raise ValueError(
                f"yin must be a single spectrum. Got shape {yin.shape} with"
                f"{yin.shape[1]} spectra. Only (n,) or (n, 1) shapes are supported."
            )
    else:
        raise ValueError("yin must be 1D or 2D array.")

    # 4.1.3 Wavelength-flux dimension consistency check
    if yin.shape[0] + 1 != len(xin):
        raise ValueError("xin must be one element longer than yin.")

    # ------------------------------------------------------------------
    # 4.2 Accumulate flux
    # ------------------------------------------------------------------
    integr = np.insert(np.nancumsum(yin, dtype=np.float64), 0, 0)

    # ------------------------------------------------------------------
    # 4.3 Interpolate to New Wavelength Grid
    # ------------------------------------------------------------------
    if sampling_function == "splinf":
        y2 = CubicSpline(xin, integr, bc_type="natural")
        integr_interp = y2(xout)
    elif sampling_function == "cubic":
        cubic_interp = interp1d(xin, integr, kind="cubic", fill_value="extrapolate")
        integr_interp = cubic_interp(xout)
    else:
        if sampling_function == "slinear":
            kind = "slinear"
        elif sampling_function == "quadratic":
            kind = "quadratic"
        # 4.3.1 Linear interpolation by default
        else:
            kind = "linear"
        interp_func = interp1d(xin, integr, kind=kind, fill_value="extrapolate")
        integr_interp = interp_func(xout)

    # ------------------------------------------------------------------
    # 4.4 Differentiate to get resampled spectrum
    # ------------------------------------------------------------------
    yout = (np.roll(integr_interp, -1) - integr_interp)[: len(integr_interp) - 1]
    # yout = np.diff(integr_interp)

    # ------------------------------------------------------------------
    # 4.5 Return resampled spectrum
    # ------------------------------------------------------------------
    return yout
