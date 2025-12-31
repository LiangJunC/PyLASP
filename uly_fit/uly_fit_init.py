# -*- coding: utf-8 -*-
# @Time    : 10/12/2024 20.05
# @Author  : ljc
# @FileName: uly_fit_init.py
# @Software: PyCharm
# Update:  2025/11/26 21:15:56


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of component initialization for LASP-CurveFit.

1.1 Purpose
-----------
Initialize fitting components by calling component-specific
initialization functions, converted from IDL uly_fit_init.pro
implementation.

1.2 Functions
-------------
1) uly_fit_init: Called by 'uly_fit/ulyss.py' 'uly_fit/ulyss_pytorch.py'.

1.3 Explanation
---------------
This module provides component initialization wrapper for PyLASP.
Steps:
    1) Extract wavelength range from input parameters.
    2) Check component initialization status.
    3) Initialize component structure if not already initialized.
    4) Call component-specific initialization function.
    5) Return initialized component structure.

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
from uly_tgm.uly_tgm_init import uly_tgm_init
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray
C_light = 299792.458


# ======================================================================
# 4. Component initialization wrapper function
# ======================================================================
def uly_fit_init(
    cmp: dict,
    lamrange: ArrayLike | None = None,
    velscale: float | None = None,
) -> dict:
    r"""Initialize fitting component by calling initialization function.

    Parameters
    ----------
    cmp : dict
        Component dictionary structure containing model information,
        initial parameter values, and fitting configuration. Single
        component mode: cmp is a dictionary, not a list.
    lamrange : np.ndarray, optional
        shape (2,)
        Wavelength range in Angstroms [lambda_min, lambda_max].
    velscale : float, optional
        Velocity scale in km/s corresponding to wavelength step size.
        Formula: velscale = ln_step * C_light
        Default is None.

    Returns
    -------
    cmp : dict
        Initialized component structure with updated wavelength grid
        and evaluation data.

    Notes
    -----
    - If component has init_fun attribute, calls component-specific
      initialization function (e.g., uly_tgm_init).
    - If component lacks init_fun attribute, performs basic wavelength
      grid initialization.
    - Single component mode is used in LASP-CurveFit workflow.
    - Speed of light is taken as C_light = 299792.458 km/s.

    Examples
    --------
    >>> from uly_tgm.uly_tgm import uly_tgm
    >>> from file_paths import TGM_MODEL_FILE
    >>> cmp = uly_tgm(model_file=TGM_MODEL_FILE())
    >>> cmp = uly_fit_init(
    ...     cmp,
    ...     lamrange=np.array([4000.0, 6000.0]),
    ...     velscale=69.029764
    ... )
    >>> print(cmp['npix'])
    1760

    """

    # ------------------------------------------------------------------
    # 4.1 Extract wavelength range
    # ------------------------------------------------------------------
    wr = np.array(lamrange)

    # ------------------------------------------------------------------
    # 4.2 Initialize component structure
    # ------------------------------------------------------------------
    if cmp.get("init_fun", None) is None:
        cmp["start"] = np.log(wr[0])
        cmp["step"] = velscale / C_light
        cmp["npix"] = 1 + int(np.round(np.log(wr[1] / wr[0]) / cmp["step"]))
        # cmp["npix"] = 1 + np.floor(np.log(wr[1] / wr[0]) / cmp["step"])
        cmp["sampling"] = 1
        cmp["eval_data"] = cmp
    else:
        cmp = uly_tgm_init(cmp, lamrange=wr, velscale=velscale)

    # ------------------------------------------------------------------
    # 4.3 Return initialized component structure
    # ------------------------------------------------------------------
    return cmp
