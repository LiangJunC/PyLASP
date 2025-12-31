# -*- coding: utf-8 -*-
# @Time    : 08/12/2024 11.13
# @Author  : ljc
# @FileName: uly_spect_logrebin.py
# @Software: PyCharm
# Update:  2025/11/26 22:17:21


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of spectral resampling for LASP-CurveFit.

1.1 Purpose
-----------
Perform wavelength resampling on model spectra to match the flux
dimension of the observed spectrum.

1.2 Functions
-------------
1) uly_spect_logrebin: Called by 'uly_fit/ulyss.py',
   'uly_fit/ulyss_pytorch.py' and 'uly_tgm_eval/uly_tgm_eval.py'.

1.3 Explanation
---------------
During parameter inference, each optimizer iteration produces an
updated set of stellar parameters, which are fed into the spectral
emulator to generate a model spectrum. uly_spect_logrebin then
resamples this model spectrum from its native wavelength grid onto the
observed spectrum’s wavelength grid, ensuring one-to-one alignment of
data points.
Steps:
    1) uly_spect_logrebin builds input/output wavelength bin edges
       (borders, NewBorders) on a logarithmic grid.
    2) It calls WRS/xrebin.py, which uses a "cumulative-interpolate-
       differentiate" scheme to resample the flux onto NewBorders.
    3) If div_flat is True, the resampled flux is divided by flat to
       compensate for the missing wavelength-interval factors and make
       it equivalent to an "integrate-interpolate-derivative" scheme.
    4) The output structure is updated with the new log-wavelength
       sampling (start, step, sampling=1, dof_factor, etc.).

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
from WRS.xrebin import xrebin
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Logarithmic-grid wavelength resampling function
# ======================================================================
def uly_spect_logrebin(
    SignalIn: dict,
    vsc: float | None = None,
    waverange: ArrayLike | None = None,
    sampling_function: str | None = None,
    div_flat: bool = True,
    exact: bool = False,
    overwrite: bool = False,
) -> dict:
    r"""Perform wavelength resampling on input spectrum structure.

    Parameters
    ----------
    SignalIn : dict
        Input data in dictionary structure, containing flux, wavelength,
        sampling info, etc.
    vsc : float, optional
        Velocity scale per pixel (km/s), calculated as:
        vsc = ln_step * c = log10_step * c * ln(10).
    waverange : np.ndarray, optional
        shape (2,)
        Interpolation wavelength range.
    sampling_function : str, optional
        Interpolation method. Options: "splinf", "cubic", "slinear",
        "quadratic", "linear". Default: "linear".
    div_flat : bool, optional
        Whether to correct the difference between "cumulative-
        interpolate-differentiate" and "integrate-interpolate-
        derivative" in 'WRS/xrebin.py'. Default: True.
    exact : bool, optional
        Whether to force output to align exactly with waverange and vsc.
        Default: False.
    overwrite : bool, optional
        Whether to overwrite the input spectrum structure.
        Default: False.

    Returns
    -------
    SignalOut : dict
        Spectrum structure containing logarithmically resampled data.

    Raises
    ------
    TypeError
        If 'SignalIn' is not a dict-like spectrum structure, or
        'waverange' is not a ArrayLike.
    ValueError
        If 'SignalIn' does not contain a 'data' field, if 'waverange'
        is not strictly increasing, if the requested wavelength range
        is invalid, or if the sampling mode is not supported.

    Notes
    -----
    - "logarithmic wavelength sampling" refers to resampling the model
      spectrum onto an $\ln \lambda$ grid that is uniformly spaced in
      $\ln \lambda$ but defined in linear wavelength units ($\AA$).
    - For observed spectra: No resampling is performed, function returns
      input structure directly.
    - For model spectra: Full resampling is performed to match observed
      wavelength grid.
    - Because 'WRS/xrebin.py' omits the wavelength interval in both
      cumulative sum and differencing, dividing by flat cancels the
      missing interval factors.
    - Since the shape difference between model and observed spectra can
      be corrected using pseudo-continuum, the div_flat setting has
      minimal impact on parameter inference in LASP.

    Examples
    --------
    >>> signal = {
    ...     'data': np.random.rand(7500, 1),
    ...     'start': 3800,
    ...     'step': 0.2,
    ...     'sampling': 0,
    ...     'title': 'test',
    ...     'hdr': 'test'
    ... }
    >>> signal_out = uly_spect_logrebin(signal, vsc=69.0)
    >>> signal_out = uly_spect_logrebin(signal, vsc=69.0, waverange=np.array([4200, 5700]))

    """

    # ------------------------------------------------------------------
    # 4.1 Input validation
    # ------------------------------------------------------------------
    # 4.1.1 Check if input data is a spectrum structure with flux data
    if not isinstance(SignalIn, dict):
        raise TypeError("SignalIn must be a dict-like spectrum structure.")
    if "data" not in SignalIn:
        raise ValueError("SignalIn must contain a 'data' field.")
    # 4.1.2 Check if waverange is a ArrayLike containing two elements
    if waverange is not None:
        if not isinstance(waverange, ArrayLike):
            raise TypeError("waverange must be a ArrayLike.")
        if len(waverange) != 2:
            raise ValueError("waverange must be a ArrayLike with two elements.")
        if waverange[0] > waverange[1]:
            raise ValueError("waverange[0] must be smaller than waverange[1].")

    # 4.1.3 Get basic information
    # sampling = 0: linear, uniform $\lambda$
    # sampling = 1: logarithmic, uniform $\ln \lambda$
    # sampling = 2: linear and logarithmic non-uniform
    flux, err_, goodpix = (
        SignalIn.get("data", None),
        SignalIn.get("err", None),
        SignalIn.get("goodpix", None),
    )
    if flux is None:
        raise ValueError("SignalIn must contain a 'data' field.")
    dim, npix, step, start, sampling, C, wavelen, title, hdr = (
        flux.shape,
        flux.shape[0],
        SignalIn.get("step", 1),
        SignalIn.get("start", 0),
        SignalIn.get("sampling", 0),
        299792.458,
        SignalIn.get("wavelen", None),
        SignalIn.get("title", None),
        SignalIn.get("hdr", None),
    )

    # ------------------------------------------------------------------
    # 4.2 Check if resampling is needed
    # ------------------------------------------------------------------
    # 4.2.1 If input spectrum has logarithmic equally-spaced wavelength
    if sampling == 1:
        # 4.2.1.1 Flag whether output can be directly copied
        cpout = False
        # 4.2.1.2 Check if vsc is consistent with C * step
        if vsc is not None:
            # 4.2.1.2.1 If vsc is consistent with C * step
            if abs(1.0 - vsc / (C * step)) * npix < 0.001:
                cpout = True
        # 4.2.1.3 If vsc is not set
        elif vsc is None:
            cpout = True

        # 4.2.1.4 Check wavelength alignment
        if exact and cpout and waverange is not None:
            # 4.2.1.4.1 Check starting wavelength alignment
            nshift = (start - np.log(waverange[0])) / step % 1
            if abs(nshift) > 0.001:
                cpout = False
            # 4.2.1.4.2 Check ending wavelength alignment
            if cpout and len(waverange) == 2:
                # Note: Fix IDL right-side nshift bug
                # nshift = (start + (npix - 1) * step - np.log(waverange[0])) / step % 1
                nshift = (start + (npix - 1) * step - np.log(waverange[1])) / step % 1
                if abs(nshift) > 0.001:
                    cpout = False

        # 4.2.1.5 If no resampling needed, return input structure
        if cpout:
            return SignalIn if overwrite else SignalIn.copy()

    # ------------------------------------------------------------------
    # 4.3 Calculate wavelength bins
    # ------------------------------------------------------------------
    # 4.3.1 Input spectrum has linear equally-spaced wavelength
    if sampling == 0:
        # 4.3.1.1 Calculate linear and logarithmic wavelength bins
        # Linear wavelength [4199.2 4199.4 4199.6 ... 5700.0 5700.2],
        # bins are [4199.1 4199.3 4199.5 ... 5700.1 5700.3]
        borders = start + np.array([-0.5, *(np.arange(npix) + 0.5)]) * step
        bordersLog = np.log(borders)

        # 4.3.1.2 Calculate logRange
        if vsc is None:
            wrange = np.log([start, start + step * (npix - 1)])
            logScale = (wrange[1] - wrange[0]) / (npix - 1)
        else:
            logScale = vsc / C
        logRange = (
            np.array([bordersLog[0], bordersLog[-1]]) + np.array([0.5, -0.5]) * logScale
        )

        # 4.3.1.3 Get intersection boundaries of logRange and waverange
        # Add integer multiples of logScale to calculate left boundary
        if waverange is not None:
            nshift = np.ceil(
                np.max([0, (logRange[0] - np.log(waverange[0]))]) / logScale - 1e-7
            )
            logRange[0] = np.log(waverange[0]) + logScale * nshift
            if len(waverange) == 2:
                logRange[1] = np.min([np.log(waverange[1]), logRange[1]])
            if logRange[1] < logRange[0]:
                raise ValueError("waverange is not valid.")

        # 4.3.1.4 Calculate resampling wavelength
        nout, nin = (
            np.round((logRange[1] - logRange[0]) / logScale + 1),
            np.round((np.exp(logRange)[1] - np.exp(logRange)[0]) / step + 1),
        )
        logStart, dof_factor = logRange[0], nout / nin
        NewBorders = np.exp(logStart + (np.arange(nout + 1) - 0.5) * logScale)
        if NewBorders[0] > borders[npix]:
            raise ValueError("start value is not in the valid range.")

        # 4.3.1.5 Calculate flat
        flat = np.exp(logStart + np.arange(nout) * logScale) * logScale / step

    # 4.3.2 Input spectrum has logarithmic equally-spaced wavelength
    elif sampling == 1:
        # 4.3.2.1 Calculate logarithmic and linear wavelength bins
        bordersLog = start + np.array([-0.5, *(np.arange(npix) + 0.5)]) * step
        borders = np.exp(bordersLog)

        # 4.3.2.2 Calculate logRange
        if vsc is None:
            logScale = step
        else:
            logScale = vsc / C
        logRange = (
            np.array([bordersLog[0], bordersLog[-1]]) + np.array([0.5, -0.5]) * logScale
        )

        # 4.3.2.3 Get intersection boundaries of logRange and waverange
        # Add integer multiples of logScale to calculate left boundary
        if waverange is not None:
            nshift = np.ceil(
                np.max([0, (logRange[0] - np.log(waverange[0]))]) / logScale - 1e-7
            )
            logRange[0] = np.log(waverange[0]) + logScale * nshift
            if len(waverange) == 2:
                logRange[1] = np.min([np.log(waverange[1]), logRange[1]])
            if logRange[1] < logRange[0]:
                raise ValueError("waverange is not valid.")

        # 4.3.2.4 Calculate resampling wavelength
        nout, nin = (
            np.round((logRange[1] - logRange[0]) / logScale + 1),
            np.round((np.exp(logRange)[1] - np.exp(logRange)[0]) / step + 1),
        )
        logStart, dof_factor = logRange[0], nout / nin
        NewBorders = np.exp(logStart + (np.arange(nout + 1) - 0.5) * logScale)
        if NewBorders[0] > borders[npix]:
            raise ValueError("start value is not in the valid range.")

        # 4.3.2.5 Calculate flat
        flat = logScale / step

    # 4.3.3 Input spectrum has non-uniform wavelength
    elif sampling == 2:
        # 4.3.3.1 Calculate linear wavelength and logarithmic bins
        if wavelen is None:
            raise ValueError("wavelen is required for non-uniform wavelength.")
        borders = (wavelen[:-1] + wavelen[1:]) / 2
        borders = np.concatenate(
            ([2 * wavelen[0] - borders[0]], borders, [2 * wavelen[-1] - borders[-1]])
        )

        # 4.3.3.2 Calculate logRange
        if vsc is None:
            if waverange is not None:
                nw = np.searchsorted(wavelen, waverange, side="right") - 1
                nw = np.clip(nw, 0, npix - 1)
                if nw[1] <= nw[0]:
                    # fallback to full range if window is too narrow
                    nw = np.array([0, npix - 1])
            else:
                nw = np.array([0, npix - 1])
            logScale = np.log(wavelen[nw][1] / wavelen[nw][0]) / (nw[1] - nw[0])
        else:
            logScale = vsc / C
        logRange = np.log([wavelen[0], wavelen[-1]]) + np.array([0.5, -0.5]) * logScale

        # 4.3.3.3 Get intersection boundaries of logRange and waverange
        # Add integer multiples of logScale to calculate left boundary
        if waverange is not None:
            nshift = np.ceil(
                np.max([0, (logRange[0] - np.log(waverange[0]))]) / logScale - 1e-7
            )
            logRange[0] = np.log(waverange[0]) + logScale * nshift
            if len(waverange) == 2:
                logRange[1] = np.min([np.log(waverange[1]), logRange[1]])
            if logRange[1] < logRange[0]:
                raise ValueError("waverange is not valid.")

        # 4.3.3.4 Calculate resampling wavelength
        nout, nin = np.round((logRange[1] - logRange[0]) / logScale + 1), len(wavelen)
        logStart, dof_factor = logRange[0], nout / nin
        NewBorders = np.exp(logStart + (np.arange(nout + 1) - 0.5) * logScale)
        if NewBorders[0] > borders[npix]:
            raise ValueError("start value is not in the valid range.")

        # 4.3.3.5 Calculate flat
        # If resampling a vector of 1s, the resampled result is still 1,
        # that is flat = xrebin(borders, [1, 1, ..., 1], NewBorders)
        flat = xrebin(
            borders,
            np.ones(dim[0], dtype=np.float64),
            NewBorders,
            sampling_function=sampling_function,
        )

    # 4.3.4 If sampling mode is not specified, raise error
    else:
        raise ValueError(f"Invalid sampling mode: {sampling}.")

    # --------------------------------------------------------------------------
    # 4.4 Resample spectrum data
    # --------------------------------------------------------------------------
    # 4.4.1 Initialize output spectrum structure
    SignalOut = SignalIn if overwrite else SignalIn.copy()

    # 4.4.2 Flux resampling
    if div_flat:
        SignalOut["data"] = (
            xrebin(
                borders,
                flux,
                NewBorders,
                sampling_function=sampling_function,
            )
            / flat
        )
    else:
        SignalOut["data"] = xrebin(
            borders,
            flux,
            NewBorders,
            sampling_function=sampling_function,
        )

    # 4.4.3 Flux error resampling
    # In LASP, err=1 is set for observed spectra, but note that LAMOST
    # spectra provide inverse variance. Model spectra have not flux
    # error, i.e., None.
    # 4.4.3.1 Get flux error
    if (err_ is not None) and (isinstance(err_, np.ndarray)):
        n_err = err_.shape[0]
    else:
        n_err = 0

    # 4.4.3.2 Flux squared error resampling
    if n_err == npix:
        err = xrebin(
            borders,
            err_**2,
            NewBorders,
            sampling_function=sampling_function,
        )
        if div_flat:
            SignalOut["err"] = err**0.5 / flat
        else:
            SignalOut["err"] = err**0.5
        # 4.4.3.2.1 If dof_factor > 1 (resampling grid is denser)
        if dof_factor > 1:
            SignalOut["err"] = SignalOut["err"] / dof_factor**0.5
        # 4.4.3.2.2 If SignalIn["dof_factor"] > 1, update spectrum error
        dof_in = SignalIn.get("dof_factor", 1.0)
        if dof_in > 1:
            d = dof_factor
            if d * dof_in < 1:
                d = 1 / dof_in
            SignalOut["err"] = SignalOut["err"] / d**0.5

    # 4.4.4 Good pixel resampling
    if goodpix is not None:
        maskI = np.zeros(npix, dtype=np.uint8)
        maskI[goodpix] = 1
        maskO = (
            xrebin(
                borders,
                maskI,
                NewBorders,
                sampling_function=sampling_function,
            )
            / flat
        )
        SignalOut["goodpix"] = np.where(abs(maskO - 1) < 0.1)[0].tolist()

    # ------------------------------------------------------------------
    # 4.5 Update output data fields
    # ------------------------------------------------------------------
    (
        SignalOut["title"],
        SignalOut["hdr"],
        SignalOut["start"],
        SignalOut["step"],
        SignalOut["sampling"],
        SignalOut["dof_factor"],
    ) = (
        title,
        hdr,
        logStart,
        logScale,
        1,
        np.max([1.0, SignalIn.get("dof_factor", 1.0) * dof_factor]),
    )

    # ------------------------------------------------------------------
    # 4.6 Return resampled spectrum
    # ------------------------------------------------------------------
    return SignalOut
