# -*- coding: utf-8 -*-
# @Time    : 10/12/2024 16.20
# @Author  : ljc
# @FileName: uly_spect_extract.py
# @Software: PyCharm
# Update:  2025/11/26 21:47:53


# ======================================================================
# 1. Introduction
# ======================================================================
"""Python implementation of spectrum extraction for LASP-CurveFit.

1.1 Purpose
-----------
Extract a portion of spectrum and return a spectrum structure,
converted from IDL uly_spect_extract.pro implementation.

1.2 Functions
-------------
1) uly_spect_extract: Called by 'uly_read_lms/uly_spect_read_lms.py',
'uly_fit/ulyss.py', 'uly_fit/ulyss_pytorch.py' and
'uly_tgm/uly_tgm_init.py' to extract a portion of spectrum.

1.3 Explanation
---------------
The uly_spect_extract function extracts a portion of a spectrum based
on wavelength range and returns a new spectrum dictionary structure.
Steps:
    1) Determine whether to create a new structure or overwrite the
       existing one.
    2) Trim wavelength range according to specified bounds.
    3) Update mask information, flux, and flux error.
    4) Return extracted spectrum structure.

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
from uly_read_lms.uly_spect_get import uly_spect_get
from uly_read_lms.uly_spect_alloc import uly_spect_alloc
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Spectrum extraction function
# ======================================================================
def uly_spect_extract(
    SignalIn: dict, waverange: ArrayLike = None, overwrite: bool = False
) -> dict:
    """Extract a portion of spectrum and return a spectrum structure.

    Parameters
    ----------
    SignalIn : dict
        Input spectrum dictionary structure containing spectral data,
        wavelength information, etc.
    waverange : np.ndarray, optional
        shape (2,) or (2, 1)
        Specified wavelength range to extract. Must be a array with 2
        elements: Extract specified wavelength range [wave_min,
        wave_max].
    overwrite : bool, optional
        If True, SignalOut will be the same variable as SignalIn,
        otherwise a new spectrum structure is created. Default is False.

    Returns
    -------
    SignalOut : dict
        Extracted spectrum (spectrum structure, see uly_spect_alloc).
        Dictionary contains the following keys:
        - data: Flux within specified wavelength range
        - err: Flux error array (not used in LASP)
        - start: Starting air wavelength
        - step: Wavelength step size
        - sampling: Wavelength sampling flag
          (0: uniform linear, 1: uniform ln, 2: non-uniform linear)
        - goodpix: Unmasked good pixel index values
        - Other metadata from original spectrum

    Raises
    ------
    ValueError
        - If required spectral fields (flux, wavelength info) are
          missing from the input spectrum.
        - If waverange is not an ArrayLike object containing exactly
          two elements.
        - If the requested wavelength range lies completely outside
          the available spectrum.
        - If no spectral points fall inside the requested wavelength
          range.

    Notes
    -----
    - For long-slit spectrum extraction (not applicable to LASP), refer
      to the original IDL uly_spect_extract.pro code.

    Examples
    --------
    >>> signal = {
    ...     'data': np.array([1., 2., 3., 4., 5.]),
    ...     'start': 4000.0,
    ...     'step': 1.0,
    ...     'sampling': 0
    ... }
    >>> extracted = uly_spect_extract(signal, waverange=np.array([4000, 4002]))
    >>> print(extracted['data'])
    [1. 2. 3.]

    """

    # ------------------------------------------------------------------
    # 4.1 Initialize output spectrum structure
    # ------------------------------------------------------------------
    # 4.1.1 Determine whether to overwrite or create new structure
    SignalOut = SignalIn if overwrite else uly_spect_alloc(spectrum=SignalIn)
    # 4.1.2 Obtain spectral field information
    flux, err, goodpix, wavelen, start, step, sampling = (
        SignalOut.get("data", None),
        SignalOut.get("err", None),
        SignalOut.get("goodpix", None),
        SignalOut.get("wavelen", None),
        SignalOut.get("start", None),
        SignalOut.get("step", None),
        SignalOut.get("sampling", None),
    )
    # 4.1.3 Basic sanity checks on required fields
    if flux is None:
        raise ValueError("No spectral data found in the spectrum.")
    if sampling is None:
        raise ValueError("No wavelength sampling found in the spectrum.")
    if (sampling == 2) and (wavelen is None):
        raise ValueError(
            "No wavelength information found in the spectrum when sampling = 2."
        )
    if (sampling < 2) and ((start is None) or (step is None)):
        raise ValueError(
            "No wavelength information found in the spectrum when sampling < 2."
        )
    # 4.1.4 Determine number of pixels from flux array
    ndim = flux.ndim
    if ndim in (1, 2):
        npix = flux.shape[0]
    if ndim == 3:
        npix = flux.shape[1]

    # ------------------------------------------------------------------
    # 4.2 Trim wavelength range
    # ------------------------------------------------------------------
    if waverange is not None:
        # --------------------------------------------------------------
        # 4.2.1 Validate wavelength range format
        # --------------------------------------------------------------
        if (not isinstance(waverange, ArrayLike)) and (
            len(np.asarray(waverange).ravel()) != 2
        ):
            raise ValueError(
                "Waverange must be an array-like with exactly two elements."
            )

        # --------------------------------------------------------------
        # 4.2.2 Regular (linear / logarithmic) sampling
        # --------------------------------------------------------------
        if sampling < 2:
            wr = waverange
            # 4.2.2.1 Convert wavelength range to ln logarithmic scale
            if sampling == 1:
                wr = np.log(waverange)
            if (wr[0] >= (start + step * (npix - 1))) or (wr[-1] <= start):
                raise ValueError("Exceeds the boundary of the wavelength range.")

            # 4.2.2.2 Calculate number of pixels to mask on side
            # 4.2.2.2.1 Left side
            nummin = int(np.floor((wr[0] - start) / step + 0.01))
            if nummin < 0:
                # In IDL : nummin = 1
                nummin = 0
            # 4.2.2.2.2 Right side
            nummax = int(np.ceil((wr[1] - start) / step - 0.01))
            if nummax > npix:
                nummax = npix - 1

            # 4.2.2.3 Update spectrum starting position
            SignalOut["start"] += nummin * step

            # 4.2.2.4 Update mask information
            if goodpix is not None:
                msk = uly_spect_get(SignalOut, mask_bool=True)[3][nummin : nummax + 1]
                SignalOut["goodpix"] = np.where(msk == 1)[0]

            # 4.2.2.5 Trim flux and flux error arrays
            if ndim == 1:
                SignalOut["data"] = flux[nummin : nummax + 1]
            if ndim == 3:
                SignalOut["data"] = flux[:, nummin : nummax + 1, :]
            if err is not None:
                SignalOut["err"] = err[nummin : nummax + 1]

        # --------------------------------------------------------------
        # 4.2.3 Irregular sampling (sampling == 2)
        # --------------------------------------------------------------
        elif sampling == 2:
            if wavelen is None:
                raise ValueError("sampling=2 requires 'wavelen' in the spectrum.")
            if (waverange[0] >= max(wavelen)) or (waverange[-1] <= min(wavelen)):
                raise ValueError("Exceeds the boundary of the wavelength range.")

            # 4.2.3.1 Determine wavelength bounds for extraction
            lmn, lmx = (
                np.max([np.min(wavelen), waverange[0]]),
                np.min([np.max(wavelen), waverange[1]]),
            )

            # 4.2.3.2 Find indices within wavelength range
            extr = np.where((wavelen >= lmn) & (wavelen <= lmx))[0]
            if len(extr) == 0:
                raise ValueError(
                    "No spectral points fall inside the requested wavelength range."
                )

            # 4.2.3.3 Update wavelength information
            SignalOut["wavelen"] = wavelen[extr]
            if goodpix is not None:
                mask = np.zeros(SignalOut["data"].size, dtype=int)
                mask[goodpix] = 1
                SignalOut["goodpix"] = np.where(mask[extr] == 1)[0]

            # 4.2.3.4 Trim flux and flux error arrays
            SignalOut["data"] = flux[extr]
            if err is not None:
                SignalOut["err"] = err[extr]

    # ------------------------------------------------------------------
    # 4.3 Return extracted spectrum structure
    # ------------------------------------------------------------------
    return SignalOut
