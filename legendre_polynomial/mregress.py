# -*- coding: utf-8 -*-
# @Time    : 12/12/2024 11.21
# @Author  : ljc
# @FileName: mregress.py
# @Software: PyCharm
# Update:  2025/11/26 19:45:03


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of multiple LR for LASP-CurveFit.

1.1 Purpose
-----------
Implement multiple linear regression to calculate regression
coefficients based on IDL mregress.pro core algorithm.

1.2 Functions
-------------
1) mregress: Called by 'uly_fit/uly_fit_conv_weight_poly.py' to solve
   linear regression coefficients.

1.3 Explanation
---------------
The mregress function performs weighted least squares multiple
linear regression to solve coefficient matrix A in linear system
X·A = y.
Mathematical Principle:
    For linear system y = X·A, the explicit solution of weighted
    least squares is:
        A = (X^T·W·X)^(-1)·X^T·W·y
    where:
        X: Independent variable matrix (Npoints × Nterms)
        W: Weight diagonal matrix with diagonal elements
           1/(measure_errors^2)
        y: Dependent variable vector (Npoints × 1)
        A: Regression coefficient vector (Nterms × 1)
Steps:
    1) Data validation: Check types and dimensions of input data X,
       y, measure_errors.
    2) Calculate weight matrix W.
    3) Weighted processing: Calculate X^T·W·X.
    4) Matrix inversion: Calculate (X^T·W·X)^(-1).
    5) Weighted processing: Calculate W·y.
    6) Calculate regression coefficient A:
       (X^T·W·X)^(-1)·X^T·W·y.

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

# import scipy as sc
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Multiple linear regression function
# ======================================================================
def mregress(
    x: ArrayLike,
    y: ArrayLike,
    measure_errors: ArrayLike | None = None,
    inv: dict | None = None,
) -> tuple[ArrayLike, dict]:
    r"""Perform weighted least squares multiple linear regression.

    Parameters
    ----------
    x : np.ndarray
        shape (Npoints, Nterms)
        Independent variable data matrix, where Npoints is the number
        of sample points, Nterms is the number of coefficients
        (independent variables) to solve.
    y : np.ndarray
        shape (Npoints,)
        Dependent variable data vector, must contain Npoints elements
        matching the first dimension of x.
    measure_errors : np.ndarray, optional
        shape (Npoints,)
        Vector containing standard measurement errors for each point
        y[i], vector length must match x and y. If not provided, unit
        weights are used.
    inv : dict, optional
        Named variable to receive dictionary containing covariance
        matrix and other reusable intermediate calculation results.

    Returns
    -------
    A : np.ndarray
        shape (Nterms,)
        Regression coefficient vector solved from equation X·A = y.
    inv : dict
        Dictionary containing detailed fitting information for further
        analysis or diagnosis. Dictionary contains the following keys:
        - 'a': Inverse of covariance matrix
        - 'ww': Weight vector
        - 'wx': Weighted independent variable matrix
        - 'sx': Standard deviation vector of independent variables
        - 'sigma': Standard error of regression coefficients
        - 'status': Calculation status code
                    0: Successfully completed calculation
                    1: Negative variance occurred, set to zero

    Raises
    ------
    TypeError
        If 'x', 'y', or 'measure_errors' are not numpy arrays.
    ValueError
        If X and Y have incompatible dimensions, or if matrix
        inversion fails due to singular array.

    Notes
    -----
    - IDL and Python may have numerical precision differences when
      calculating matrix pseudo-inverse, but the impact on final
      parameter inference for spectral fitting is minimal.
    - This implementation uses numpy.linalg.pinv to calculate
      pseudo-inverse, which is more stable but slightly slower than
      scipy.linalg.lu_solve.
    - When diagonal elements of covariance matrix are negative, they
      are set to zero and the status flag is set to 1.
    - The inv dictionary returned by the function contains detailed
      fitting information for further analysis.
    - The number of degrees of freedom is defined here as Npoints - 1
      for consistency with the original IDL mregress.pro implementation,
      rather than the more conventional Npoints - Nterms.

    Examples
    --------
    >>> X = np.array([[2, 4, 6], [3, 5, 9], [10, 11, 3]])
    >>> y = np.array([6, 8, 11])
    >>> measure_errors = np.array([1, 1, 1])
    >>> a, inv = mregress(X, y, measure_errors)
    >>> print(a.shape)
    (3,)

    """

    # ------------------------------------------------------------------
    # 4.1 Input validation
    # ------------------------------------------------------------------
    # 4.1.1 Type check
    if not isinstance(x, ArrayLike):
        raise TypeError(
            f"x must be a numpy array, got {type(x).__name__}. "
            f"Use np.array() to convert."
        )
    if not isinstance(y, ArrayLike):
        raise TypeError(
            f"y must be a numpy array, got {type(y).__name__}. "
            f"Use np.array() to convert."
        )
    # 4.1.2 Check y is 1D
    if y.ndim != 1:
        raise ValueError(
            f"y must be a 1D array of shape (Npoints,), got shape {y.shape}."
        )

    # 4.1.3 Handle measurement errors parameter
    if measure_errors is None:
        measure_errors = np.ones_like(y)
    else:
        if not isinstance(measure_errors, ArrayLike):
            raise TypeError(
                f"measure_errors must be a numpy array, got "
                f"{type(measure_errors).__name__}. Use np.array() to convert."
            )
        if measure_errors.ndim != 1:
            raise ValueError(
                f"measure_errors must be a 1D array, got shape {measure_errors.shape}."
            )
        if measure_errors.shape[0] != y.shape[0]:
            raise ValueError("measure_errors must have the same length as y.")

    # 4.1.4 Get array dimension information and verify consistency
    sx, sy = x.shape, y.shape

    Npoints = sx[0]
    if sx[0] != sy[0]:
        raise ValueError("X and Y have incompatible dimensions.")

    # 4.1.5 Ensure x is a 2D array
    if x.ndim == 1:
        x = x.reshape(Npoints, 1)

    # 4.1.6 Determine number of regression terms and degrees of freedom
    Nterms, nfree = 1 if len(sx) == 1 else sx[1], Npoints - 1

    # 4.1.7 Check validity of inv dictionary
    if inv is not None:
        if inv.get("wx") is None or inv.get("wx").shape[0] != Npoints:
            inv = None

    # ------------------------------------------------------------------
    # 4.2 Weighted least squares core calculation
    # ------------------------------------------------------------------
    with np.errstate(under="ignore"):
        if inv is None:
            # ----------------------------------------------------------
            # 4.2.1 Weight calculation and normalization
            # ----------------------------------------------------------
            weights = 1 / (measure_errors**2)
            # 4.2.1.1 Normalize weights to avoid numerical issues
            sw = np.sum(weights) / Npoints
            weights = weights / sw

            # ----------------------------------------------------------
            # 4.2.2 Weighted processing: calculate X^T·W·X
            # ----------------------------------------------------------
            wgt = np.tile(weights, (Nterms, 1)).T
            wx = wgt * x
            # 4.2.2.1 To improve inversion stability
            sigmax = np.sqrt(np.sum(x * wx, axis=0) / nfree)
            ar = np.dot(wx.T, x) / (nfree * np.outer(sigmax, sigmax))

            # ----------------------------------------------------------
            # 4.2.3 Matrix inversion: calculate (X^T·W·X)^(-1)
            # ----------------------------------------------------------
            try:
                # 4.2.3.1 Calculate pseudo-inverse of the matrix
                ar = np.linalg.pinv(ar)
                # ar = sc.linalg.lu_solve(sc.linalg.lu_factor(ar), np.eye(ar.shape[0]))
                status = 0
            except np.linalg.LinAlgError:
                raise ValueError("Inversion failed due to singular array.")

            # ----------------------------------------------------------
            # 4.2.4 Calculate standard error
            # ----------------------------------------------------------
            sigma = np.diag(ar) / (sw * nfree * sigmax**2)
            # 4.2.4.1 Check and handle negative value
            neg = np.where(sigma < 0)[0].tolist()
            if len(neg) > 0:
                sigma[neg] = 0
                status = 1
                # warnings.warn("Negative variance detected in coefficient covariance matrix.")
            # 4.2.4.2 Standard error = sqrt(variance)
            sigma = np.sqrt(sigma)

            # ----------------------------------------------------------
            # 4.2.5 Save intermediate calculation results
            # ----------------------------------------------------------
            inv = {
                "a": ar,
                "ww": weights,
                "wx": wx,
                "sx": sigmax,
                "sigma": sigma,
                "status": status,
            }

        # --------------------------------------------------------------
        # 4.2.6 Weighted processing: calculate W·y
        # --------------------------------------------------------------
        sigmay = np.sqrt(np.sum(inv["ww"] * (y**2)) / nfree)

        # --------------------------------------------------------------
        # 4.2.7 Solve regression coefficient A: (X^T·W·X)^(-1)·X^T·W·y
        # --------------------------------------------------------------
        A = np.dot(inv["a"], np.dot(inv["wx"].T, y) / (inv["sx"] * sigmay * nfree)) * (
            sigmay / inv["sx"]
        )

    # ------------------------------------------------------------------
    # 4.2.8 Return regression coefficients and detailed information
    # ------------------------------------------------------------------
    return A, inv
