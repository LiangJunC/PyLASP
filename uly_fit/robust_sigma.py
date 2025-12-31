# -*- coding: utf-8 -*-
# @Time    : 2025/1/4 15:05
# @Author  : ljc
# @FileName: robust_sigma.py
# @Software: PyCharm
# Update:  2025/11/26 20:43:07


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of robust sigma calculation for LASP-CurveFit.

1.1 Purpose
-----------
Calculate robust standard deviation using biweight method, converted
from IDL robust_sigma.pro implementation.

1.2 Functions
-------------
1) robust_sigma: Calculate robust standard deviation using biweight
   estimator with median absolute deviation (MAD) scaling.

1.3 Explanation
---------------
This module provides robust dispersion estimation for PyLASP.
Steps:
    1) Compute central value (zero or median).
    2) Calculate median absolute deviation (MAD).
    3) Compute biweight weights for data points.
    4) Calculate weighted variance and standard deviation.
    5) Return robust standard deviation estimate.

1.4 Notes
---------
- This is a Python-specific rewrite and optimization, not a complete
  port of all IDL features, applicable to LASP-CurveFit.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
import numpy as np


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Robust sigma calculation function
# ======================================================================
def robust_sigma(y: ArrayLike, zero_bool: bool = True) -> float:
    r"""Calculate robust standard deviation using biweight estimator.

    Parameters
    ----------
    y : np.ndarray
        Input data array for robust dispersion calculation.
    zero_bool : bool, optional
        If True, use zero as central value; if False, use median.
        Default is True.

    Returns
    -------
    sigma : float
        Robust standard deviation. Returns 0.0 if calculation fails
        or if variance is negative.

    Raises
    ------
    TypeError
        - This distribution is too Weird.

    Notes
    -----
    - Uses biweight estimator with MAD (median absolute deviation) scaling.
    - MAD is normalized by 0.6745 to match Gaussian standard deviation.
    - Falls back to mean absolute deviation (normalized by 0.80) if MAD
      is too small.
    - Requires at least 3 valid points with biweight < 1.0.
    - Resistant to outliers and heavy-tailed distributions.

    Examples
    --------
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    >>> sigma = robust_sigma(y, zero_bool=False)
    >>> print(f"{sigma:.4f}")
    1.6814

    """

    # ------------------------------------------------------------------
    # 4.1 Set numerical threshold
    # ------------------------------------------------------------------
    eps = 1.0e-20

    # ------------------------------------------------------------------
    # 4.2 Determine central value
    # ------------------------------------------------------------------
    y0 = 0.0 if zero_bool else np.median(y)

    # ------------------------------------------------------------------
    # 4.3 Calculate median absolute deviation
    # ------------------------------------------------------------------
    mad = np.median(np.abs(y - y0)) / 0.6745

    # ------------------------------------------------------------------
    # 4.4 Fall back to mean absolute deviation if needed
    # ------------------------------------------------------------------
    if mad < eps:
        mad = np.mean(np.abs(y - y0)) / 0.80
    if mad < eps:
        return 0.0

    # ------------------------------------------------------------------
    # 4.5 Calculate biweight values
    # ------------------------------------------------------------------
    u = (y - y0) / (6.0 * mad)
    uu = u * u
    q = np.where(uu <= 1.0)[0]

    # ------------------------------------------------------------------
    # 4.6 Validate sufficient valid points
    # ------------------------------------------------------------------
    if len(q) < 3:
        raise ValueError("Robust_sigma: This distribution is too Weird.")

    # ------------------------------------------------------------------
    # 4.7 Calculate robust variance
    # ------------------------------------------------------------------
    n = np.sum(np.isfinite(y))
    numerator = np.sum((y[q] - y0) ** 2 * (1 - uu[q]) ** 4)
    den1 = np.sum((1 - uu[q]) * (1 - 5 * uu[q]))
    sigma = n * numerator / (den1 * (den1 - 1))

    # ------------------------------------------------------------------
    # 4.8 Return robust standard deviation
    # ------------------------------------------------------------------
    return np.sqrt(sigma) if sigma > 0 else 0.0
