# -*- coding: utf-8 -*-
# @Time    : 07/12/2024 23.21
# @Author  : ljc
# @FileName: uly_spect_get.py
# @Software: PyCharm
# Update:  2025/11/26 21:51:51


# ======================================================================
# 1. Introduction
# ======================================================================
"""Python implementation of spectrum-field retrieval for LASP-CurveFit.

1.1 Purpose
-----------
Retrieve information from a spectrum structure, converted from the IDL
uly_spect_get.pro implementation.

1.2 Functions
-------------
1) uly_spect_get: Called by 'uly_read_lms/extract.py',
'uly_fit/ulyss.py', 'uly_fit/ulyss_pytorch.py',
'uly_tgm/uly_tgm_init.py' and 'uly_fit/uly_fit_a_cmp.py'.

1.3 Explanation
---------------
The uly_spect_get function retrieves various types of information
from a spectrum dictionary structure, including wavelength range,
good pixel array, FITS header, and pixel mask array.

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
# 4. Spectrum information retrieval function
# ======================================================================
def uly_spect_get(
    SignalIn: dict,
    waverange_bool: bool = None,
    goodpix_bool: bool = None,
    hdr_bool: bool = None,
    mask_bool: bool = None,
) -> tuple[ArrayLike | None, ArrayLike | None, str | None, ArrayLike | None]:
    """Retrieve information from spectrum dictionary structure.

    This function processes a spectrum structure returned by
    uly_spect_read_lss and retrieves the requested information
    based on specified flags.

    Parameters
    ----------
    SignalIn : dict
        Input spectrum dictionary structure.
    waverange_bool : bool, optional
        If not None, return wavelength range as a 2-element array
        containing [minimum_wavelength, maximum_wavelength].
    goodpix_bool : bool, optional
        If not None, return good pixel array from spectrum dictionary
        structure. If not set in the structure, assumes all pixels
        are valid.
    hdr_bool : bool, optional
        If not None, return FITS header from spectrum dictionary
        structure. If header does not exist, returns None.
    mask_bool : bool, optional
        If not None, return array containing pixel mask where 1
        indicates good pixel and 0 indicates bad pixel. Mask length
        matches spectrum vector length. If input spectrum structure
        does not contain good pixel array, assumes all pixels are valid.

    Returns
    -------
    waverange : np.ndarray or None
        Wavelength range array [wave_min, wave_max]. None if not
        requested.
    goodpix : np.ndarray or None
        Array of good pixel indices. None if not requested.
    hdr : str or None
        FITS header string. None if header not available. None if
        not requested.
    mask : np.ndarray or None
        Pixel mask array with 1 for good pixels and 0 for bad pixels.
        None if not requested.

    Raises
    ------
    ValueError
        - If the input spectrum structure is None.
        - If the input spectrum structure does not contain flux data
          in the 'data' field.
        - If waverange_bool is True but the spectrum structure does not
          contain the 'sampling' field.
        - If sampling < 2 but 'start' or 'step' is missing when
          computing the wavelength range.
        - If sampling == 2 but the 'wavelen' array is missing.
        - If the sampling type is invalid (not 0, 1, or 2).

    Examples
    --------
    >>> signal = {
    ...     'data': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ...     'start': 4000,
    ...     'step': 1.0,
    ...     'sampling': 0,
    ...     'goodpix': np.array([0, 1, 2, 4]),
    ...     'hdr': 'SIMPLE = T'
    ... }
    >>> waverange, goodpix, hdr, mask = uly_spect_get(
    ...     signal, waverange_bool=True, goodpix_bool=True, hdr_bool=True, mask_bool=True
    ... )
    >>> print(waverange)
    [4000. 4004.]
    >>> print(goodpix)
    [0 1 2 4]
    >>> print(mask)
    [1 1 1 0 1]

    """

    # ------------------------------------------------------------------
    # 4.1 Get flux and flux dimension
    # ------------------------------------------------------------------
    if SignalIn is None:
        raise ValueError("Input spectrum structure cannot be None.")
    else:
        flux = SignalIn.get("data", None)
    if flux is None:
        raise ValueError("Input spectrum structure does not contain data.")
    else:
        ndim = flux.ndim
        if ndim in (1, 2):
            npix = flux.shape[0]
        if ndim == 3:
            npix = flux.shape[1]

    # ------------------------------------------------------------------
    # 4.2 Calculate and return wavelength range if requested
    # ------------------------------------------------------------------
    if waverange_bool:
        sampling = SignalIn.get("sampling", None)
        if sampling is None:
            raise ValueError("Input spectrum structure does not contain 'sampling'.")
        if sampling < 2:
            start, step = SignalIn.get("start", None), SignalIn.get("step", None)
            if start is None or step is None:
                raise ValueError("Input spectrum is missing wavelength start or step.")
            else:
                # 4.2.1 Linear sampling
                waverange = np.array([start, start + (npix - 1) * step])
            # 4.2.2 Logarithmic sampling (ln scale)
            if sampling == 1:
                waverange = np.exp(waverange)

        # 4.2.3 Irregular sampling
        elif sampling == 2:
            wavelen = SignalIn.get("wavelen", None)
            if wavelen is None:
                raise ValueError("Input spectrum structure does not contain wavelen.")
            else:
                waverange = np.array([wavelen[0], wavelen[-1]])
        # 4.2.4 Invalid sampling
        else:
            raise ValueError("Invalid sampling type.")
    else:
        waverange = None

    # ------------------------------------------------------------------
    # 4.3 Retrieve and return good pixel list if requested
    # ------------------------------------------------------------------
    if goodpix_bool:
        goodpix = SignalIn.get("goodpix", np.arange(npix))
    else:
        goodpix = None

    # ------------------------------------------------------------------
    # 4.4 Retrieve and return FITS header if requested
    # ------------------------------------------------------------------
    if hdr_bool:
        hdr = SignalIn.get("hdr", None)
    else:
        hdr = None

    # ------------------------------------------------------------------
    # 4.5 Generate and return pixel mask if requested
    # ------------------------------------------------------------------
    if mask_bool:
        mask = np.zeros(npix, dtype=np.uint8)
        # 4.5.1 Set mask based on good pixel list if available
        goodpix_mask = SignalIn.get("goodpix", None)
        if goodpix_mask is not None:
            mask[goodpix_mask] = 1
        # 4.5.2 Assume all pixels are valid if good pixel list not set
        else:
            mask += 1
    else:
        mask = None

    # ------------------------------------------------------------------
    # 4.6 Return requested information
    # ------------------------------------------------------------------
    return waverange, goodpix, hdr, mask
