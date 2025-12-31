# -*- coding: utf-8 -*-
# @Time    : 07/12/2024 23.17
# @Author  : ljc
# @FileName: uly_spect_alloc.py
# @Software: PyCharm
# Update:  2025/11/26 21:42:35


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python port to allocate spectrum structures for LASP-CurveFit.

1.1 Purpose
-----------
Allocate spectrum structure based on IDL uly_spect_alloc.pro core
algorithm.

1.2 Functions
-------------
1) uly_spect_alloc: Called by 'uly_read_lms/uly_spect_exact.py',
   'uly_tgm_eval/uly_tgm_eval.py' and 'uly_tgm/uly_tgm_init.py'.

1.3 Explanation
---------------
The uly_spect_alloc function allocates the spectrum structure. This
structure is used throughout the PyLASP to store and manage spectral
data, metadata, and processing parameters.
Steps:
    1) Initialize default spectrum structure.
    2) Update with spectrum parameter if provided.
    3) Update individual fields with provided parameters.
    4) Return initialized spectrum structure.

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
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Spectrum structure allocation function
# ======================================================================
def uly_spect_alloc(
    title: str = None,
    data: ArrayLike = None,
    start: float = None,
    step: float = None,
    sampling: int = None,
    err: ArrayLike = None,
    wavelen: ArrayLike = None,
    goodpix: list = None,
    header: dict = None,
    dof_factor: float = None,
    spectrum: dict = None,
) -> dict:
    r"""Allocate and update spectrum dictionary structure.

    Parameters
    ----------
    title : str, optional
        Spectrum filename.
    data : np.ndarray, optional
        Flux within specified wavelength range.
    start : float, optional
        Starting wavelength.
    step : float, optional
        Wavelength step size.
    sampling : int, optional
        Wavelength sampling flag:
        0 – uniformly spaced in linear wavelength,
        1 – uniformly spaced in ln wavelength,
        2 – non-uniform in both linear and ln (wavelength stored in
            linear scale).
    err : np.ndarray, optional
        Error array, not used in LASP (but note some observed spectra
        have flux errors).
    wavelen : np.ndarray, optional
        Wavelength array, used when sampling=2, not set in LASP.
    goodpix : list, optional
        Unmasked good pixel index values.
    header : dict, optional
        Spectrum header.
    dof_factor : float, optional
        Degrees of freedom factor. Ratio of actual pixels to independent
        measurements. Increases when spectrum is resampled to smaller
        pixels. Default: 1.0.
    spectrum : dict, optional
        Existing spectrum structure to initialize a new structure from.

    Returns
    -------
    spectrum : dict
        Initialized spectrum structure containing the
        following keys:
        - title: Spectrum filename
        - hdr: Spectrum header
        - data: Flux within specified wavelength range
        - err: Flux error array (not used in LASP)
        - wavelen: Wavelength array (used when sampling=2)
        - goodpix: Unmasked good pixel index values
        - start: Starting air wavelength
        - step: Wavelength step size
        - sampling: Wavelength sampling flag
          (0: uniform linear, 1: uniform ln, 2: non-uniform linear)
        - dof_factor: Degrees of freedom factor

    Raises
    ------
    ValueError
        - If wavelen is given and sampling is not NaN and not 2.

    Notes
    -----
    - This function is primarily for compatibility with IDL-based
      LASP-MPFit.
    - The spectrum structure is a Python dictionary that replaces
      IDL's structure data type.
    - Default sampling method is 1 (ln logarithmic wavelength).
    - dof_factor is set to 1 by default in LASP.

    Examples
    --------
    >>> import numpy as np
    >>> spectrum = uly_spect_alloc()
    >>> print(spectrum['sampling'])
    1
    >>> data = np.random.rand(1000)
    >>> spectrum = uly_spect_alloc(data=data, start=8.2, step=0.0002)
    >>> print(spectrum['data'].shape)
    (1000,)

    """

    # ------------------------------------------------------------------
    # 4.1 Initialize default spectrum dictionary structure
    # ------------------------------------------------------------------
    ini_spectrum = {
        "title": None,
        "hdr": None,
        "data": None,
        "err": None,
        "wavelen": None,
        "goodpix": None,
        "start": 1.0,
        "step": 1.0,
        "sampling": 1,
        "dof_factor": 1.0,
    }

    # ------------------------------------------------------------------
    # 4.2 Update from spectrum parameter if provided
    # ------------------------------------------------------------------
    if spectrum is not None:
        for key in [
            "title",
            "hdr",
            "data",
            "err",
            "start",
            "step",
            "sampling",
            "dof_factor",
            "wavelen",
            "goodpix",
        ]:
            value = spectrum.get(key)
            if value is not None:
                ini_spectrum[key] = value

    # ------------------------------------------------------------------
    # 4.3 Override defaults with explicit keyword arguments
    # ------------------------------------------------------------------
    # 4.3.1 Handle wavelen / sampling consistency
    if wavelen is not None:
        if sampling is not None and sampling != 2:
            raise ValueError(
                "Inconsistent input: if 'wavelen' is provided, 'sampling' must be 2."
            )
        if sampling is None:
            sampling = 2
        ini_spectrum["wavelen"] = wavelen

    # 4.3.2 Bulk update for simple scalar/array fields
    override_map = {
        "title": title,
        "hdr": header,
        "data": data,
        "err": err,
        "goodpix": goodpix,
        "start": start,
        "step": step,
        "sampling": sampling,
        "dof_factor": dof_factor,
    }

    for key, value in override_map.items():
        if value is not None:
            ini_spectrum[key] = value

    # ------------------------------------------------------------------
    # 4.4 Return allocated spectrum structure
    # ------------------------------------------------------------------
    return ini_spectrum
