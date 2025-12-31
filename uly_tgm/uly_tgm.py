# -*- coding: utf-8 -*-
# @Time    : 10/12/2024 14.52
# @Author  : ljc
# @FileName: uly_tgm.py
# @Software: PyCharm
# Update:  2025/11/18 22:07:41


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of model component definition for LASP-CurveFit.

1.1 Purpose
-----------
Define a model (Teff - log g - [Fe/H]) component structure, converted
from IDL uly_tgm.pro implementation.

1.2 Functions
-------------
1) uly_tgm: Define model (Teff - log g - [Fe/H]) component with initial
   parameter guesses and constraints. Used in 'uly_fit/ulyss.py' and
   'uly_fit/ulyss_pytorch.py'.

1.3 Explanation
---------------
This module provides component structure definition for PyLASP.
Steps:
    1) Initialize a model component with a model file path.
    2) Set parameter initial guesses (Teff, log g, [Fe/H]).
    3) Define parameter limits and constraints.
    4) Configure component weight ranges.
    5) Return component dictionary structure.

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
import os
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Model component definition function
# ======================================================================
def uly_tgm(
    model_file: str,
    lsf_file: str | None = None,
    t_limits: list | None = None,
    l_limits: list | None = None,
    z_limits: list | None = None,
    t_guess: float | None = None,
    l_guess: float | None = None,
    z_guess: float | None = None,
    fixpar: ArrayLike | None = None,
    lim_weight: tuple = (0, np.inf),
) -> dict:
    r"""Define a model (Teff - log g - [Fe/H]) component structure.

    Parameters
    ----------
    model_file : str
        Path to model FITS file containing polynomial coefficients.
    lsf_file : str, optional
        Path to LSF FITS file containing a relative LSF to be injected
        in the template. Default is None.
    t_limits : list, optional
        Effective temperature inference range in Kelvin.
        If None, defaults to [-np.inf, np.inf].
    l_limits : list, optional
        Surface gravity (log g) inference range.
        If None, defaults to [-np.inf, np.inf].
    z_limits : list, optional
        Metallicity ([Fe/H]) inference range in dex.
        If None, defaults to [-np.inf, np.inf].
    t_guess : float or list, optional
        Initial guess for Teff in Kelvin. If None, defaults to 7500.0 K.
    l_guess : float or list, optional
        Initial guess for log g. If None, defaults to 3.0.
    z_guess : float or list, optional
        Initial guess for [Fe/H] in dex. If None, defaults to -0.5.
    fixpar : np.ndarray, optional
        shape (3,)
        Parameter fixing flags. If set, corresponding parameters are
        fixed during fitting. Array order: [Teff, log g, [Fe/H]].
    lim_weight : tuple, optional
        Component weight (w_{i}) range (lower, upper).
        f = w_{1} * f_{1} + ... + w_{number_cmp} * f_{number_cmp}.

    Returns
    -------
    cmp : dict
        Model component structure containing:
        - init_fun: Initialization function name
        - init_data: Data for initialization function
        - eval_fun: Evaluation function name
        - eval_data: Data for evaluation function
        - para: Stellar parameter structure list
        - start: Starting wavelength
        - step: Wavelength step
        - npix: Number of pixels
        - sampling: Wavelength sampling mode
        - mask: Pixel mask array
        - weight: Component weight
        - e_weight: Weight error
        - l_weight: Flux ratio weight
        - lim_weig: Weight limits

    Raises
    ------
    ValueError
        - If model_file is None.

    Notes
    -----
    - Single component mode: Only one set of initial parameter guesses.
    - Multi-component mode: Multiple sets of initial parameter guesses.
    - LASP uses CFI-based initialization with a single component (unless
      CFI cannot provide initial values).
    - For PyLASP, cmp is a single component for now (n_cmp,
      number_cmp = 1); support for multiple components (n_cmp,
      number_cmp != 1) will be added gradually.

    Examples
    --------
    >>> from file_paths import TGM_MODEL_FILE
    >>> cmp = uly_tgm(
    ...     model_file=TGM_MODEL_FILE(),
    ...     t_guess=None,
    ...     l_guess=None,
    ...     z_guess=None,
    ... )
    >>> print(cmp['para'][1]['guess'])
    3.0

    """

    # ------------------------------------------------------------------
    # 4.1 Set model file path
    # ------------------------------------------------------------------
    if model_file is None:
        raise ValueError("model_file must be specified.")
    model_file = os.path.join(model_file)

    # ------------------------------------------------------------------
    # 4.2 Initialize parameter guesses
    # ------------------------------------------------------------------
    if t_guess is None:
        t_guess = np.log(7500.0)
    else:
        t_guess = np.log(np.array(t_guess))
    if l_guess is None:
        l_guess = 3.0
    else:
        l_guess = np.array(l_guess)
    if z_guess is None:
        z_guess = -0.5
    else:
        z_guess = np.array(z_guess)
    if t_limits is None:
        t_limits = [-np.inf, np.inf]
    else:
        t_limits = np.log(t_limits)
    if l_limits is None:
        l_limits = [-np.inf, np.inf]
    if z_limits is None:
        z_limits = [-np.inf, np.inf]

    # ------------------------------------------------------------------
    # 4.3 Define model component structure
    # ------------------------------------------------------------------
    cmp = {
        "init_fun": "uly_tgm_init",
        "init_data": {"model": model_file, "lsf_file": lsf_file},
        "eval_fun": None,
        "eval_data": None,
        "para": None,
        "start": 0.0,
        "step": 0.0,
        "npix": 0,
        "sampling": -1,
        "mask": None,
        "weight": 0.0,
        "e_weight": 0.0,
        "l_weight": 0.0,
        "lim_weig": np.finfo(np.float64).max * np.array([0, 1]),
    }

    # ------------------------------------------------------------------
    # 4.4 Define stellar parameter structures
    # ------------------------------------------------------------------
    params = [
        {
            "name": "Teff",
            "unit": "K",
            "guess": t_guess,
            "step": 0.005,
            "limits": t_limits,
            "limited": [1, 1],
            "fixed": 0,
            "value": t_guess,
            "error": 0.0,
            "dispf": "exp",
        },
        {
            "name": "log g",
            "unit": "cm/s2",
            "guess": l_guess,
            "step": 0.01,
            "limits": l_limits,
            "limited": [1, 1],
            "fixed": 0,
            "value": l_guess,
            "error": 0.0,
        },
        {
            "name": "[Fe/H]",
            "unit": "dex",
            "guess": z_guess,
            "step": 0.01,
            "limits": z_limits,
            "limited": [1, 1],
            "fixed": 0,
            "value": z_guess,
            "error": 0.0,
        },
    ]

    # ------------------------------------------------------------------
    # 4.5 Apply fixed parameter flags if provided
    # ------------------------------------------------------------------
    if fixpar is not None:
        for i, fixed in enumerate(fixpar):
            params[i]["fixed"] = fixed

    # ------------------------------------------------------------------
    # 4.6 Set component weight range
    # ------------------------------------------------------------------
    if lim_weight is not None:
        cmp["lim_weig"] = lim_weight

    # ------------------------------------------------------------------
    # 4.7 Assign parameter list to component structure
    # ------------------------------------------------------------------
    cmp["para"] = params

    # ------------------------------------------------------------------
    # 4.8 Return component structure
    # ------------------------------------------------------------------
    return cmp
