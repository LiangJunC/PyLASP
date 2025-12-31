# -*- coding: utf-8 -*-
# @Time    : 04/12/2024 10.44
# @Author  : ljc
# @FileName: uly_tgm_eval.py
# @Software: PyCharm
# Update:  2025/11/26 22:09:51


# ======================================================================
# 1. Introduction
# ======================================================================
"""Python implementation of the spectral emulator for LASP-CurveFit.

1.1 Purpose
-----------
Generate model spectra at specified Teff, log g, and [Fe/H], converted
from IDL uly_tgm_eval.pro implementation.

1.2 Functions
-------------
1) uly_tgm_model_param: Generate model polynomial parameter combinations
   at specified Teff, log g, and [Fe/H].
2) uly_tgm_eval: Generate one model spectrum at specified Teff, log g,
   and [Fe/H], and resample to the observed wavelength grid. Called by
   'uly_fit/uly_fit_conv_weight_poly.py'.

1.3 Explanation
---------------
This module provides functions to generate spectra for PyLASP.
Steps:
    1) Generate polynomial parameter combinations for specified
       stellar parameters.
    2) Compute model spectrum using polynomial coefficients.
    3) Interpolate between temperature regimes (warm, hot, cold).
    4) Resample model spectrum to match the observed wavelength grid.
    5) Return model spectrum ready for fitting.

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
from uly_read_lms.uly_spect_alloc import uly_spect_alloc
from WRS.uly_spect_logrebin import uly_spect_logrebin


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. Model polynomial parameter generation function
# ======================================================================
def uly_tgm_model_param(
    version: int,
    teff: float,
    gravi: float,
    fehi: float,
) -> ArrayLike:
    """Generate model polynomial parameter combinations.

    Parameters
    ----------
    version : int
        Model version number.
    teff : float
        Effective temperature in log10 scale.
    gravi : float
        Surface gravity log g.
    fehi : float
        Metallicity [Fe/H].

    Returns
    -------
    param : np.ndarray
        Model polynomial parameter combinations at specified Teff,
        log g, and [Fe/H]. Shape is (np_s, 3) where np_s depends
        on the model version.

    Raises
    ------
    ValueError
        - The model version is not supported.

    Notes
    -----
    - Due to floating-point precision differences between IDL and
      Python, computational results may show minor discrepancies.
    - Teff is logarithmic (log10).
    - For version 1 and 2: warm, hot, cold model spectrum emulator
      coefficients are all 14501×26 matrices, parameter matrix is 26×3.
      For version 3: parameter matrix is 26×3.
    - Polynomial coefficients may have all-zero terms.

    Examples
    --------
    >>> param = uly_tgm_model_param(
    ...     version=2,
    ...     teff=0.25,
    ...     gravi=-0.44,
    ...     fehi=0.0
    ... )
    >>> print(param.shape)
    (23, 3)

    """

    # ------------------------------------------------------------------
    # 4.1 Determine number of parameters based on version
    # ------------------------------------------------------------------
    if version in (1, 2):
        np_s = 23
    elif version == 3:
        np_s = 26
    else:
        raise ValueError(f"Unsupported model version: {version}.")

    # ------------------------------------------------------------------
    # 4.2 Initialize parameter matrix
    # ------------------------------------------------------------------
    param = np.zeros((np_s, 3), dtype=float)

    # ------------------------------------------------------------------
    # 4.3 Compute polynomial terms for each parameter set
    # ------------------------------------------------------------------
    for i in range(3):
        teffc = teff
        if (version != 1) & (i == 2):
            teffc = teff + 0.1
        grav = gravi
        feh = fehi
        tt = teffc / 0.2
        tt2 = tt**2 - 1.0
        param[0, i] = 1.0
        param[1, i] = tt
        param[2, i] = feh
        param[3, i] = grav
        param[4, i] = tt**2
        param[5, i] = tt * tt2
        param[6, i] = tt2**2
        param[7, i] = tt * feh
        param[8, i] = tt * grav
        param[9, i] = tt2 * grav
        param[10, i] = tt2 * feh
        param[11, i] = grav**2
        param[12, i] = feh**2
        param[13, i] = tt * (tt2**2)
        param[14, i] = tt * (grav**2)
        param[15, i] = grav**3
        param[16, i] = feh**3
        param[17, i] = tt * (feh**2)
        param[18, i] = grav * feh
        param[19, i] = (grav**2) * feh
        param[20, i] = grav * (feh**2)

        # 4.3.1 Version 1
        if version == 1:
            param[21, i] = np.exp(tt)
            param[22, i] = np.exp(tt**2)

        # 4.3.2 Version 2
        if version == 2:
            param[21, i] = (
                np.exp(tt)
                - 1
                - tt * (1 + tt / 2 + tt**2 / 6 + tt**3 / 24 + tt**4 / 120)
            )
            param[22, i] = (
                np.exp(tt * 2)
                - 1
                - 2 * tt * (1 + tt + 2 / 3 * tt**2 + tt**3 / 3 + tt**4 * 2 / 15)
            )

        # 4.3.3 Version 3
        if version == 3:
            param[21, i] = tt * tt2 * grav
            param[22, i] = tt2 * tt2 * grav
            param[23, i] = tt2 * tt * feh
            param[24, i] = tt2 * (grav**2)
            param[25, i] = tt2 * (grav**3)

    # ------------------------------------------------------------------
    # 4.4 Return parameter matrix
    # ------------------------------------------------------------------
    return param


# ======================================================================
# 5. Emulator
# ======================================================================
def uly_tgm_eval(
    eval_data: dict,
    para: ArrayLike,
    sampling_function: str = None,
) -> ArrayLike:
    """Generate model spectrum and resample to observed wavelength.

    Parameters
    ----------
    eval_data : dict
        Model polynomial coefficient matrix and wavelength information.
        Contains:
        - spec_coef: Model coefficient matrix
        - version: Model version number
        - start: Starting wavelength of observed spectrum (ln scale)
        - step: Wavelength step of observed spectrum (ln scale)
        - sampling: Wavelength sampling method
        - npix: Number of pixels in observed spectrum
        - mod_start: Starting wavelength of model spectrum
        - mod_step: Wavelength step of model spectrum
        - mod_samp: Wavelength sampling method of model
    para : ArrayLike
        Stellar parameters [ln(Teff), log g, [Fe/H]].
    sampling_function : str, optional
        Interpolation method. Can be "splinf", "cubic", "slinear",
        "quadratic", or "linear". Default is "linear" interpolation.

    Returns
    -------
    tgm_model_evalhc : np.ndarray
        shape = (n,)
        Model spectrum at specified Teff, log g, and [Fe/H], resampled
        to match the observed wavelength grid.

    Raises
    ------
    ValueError
        - Wavelength range is not specified in eval_data.

    Notes
    -----
    - The function handles three stellar temperature regimes:
      warm (T ≤ 9000 K), hot (T ≥ 7000 K), and cold (T ≤ 4550 K).
    - Smooth interpolation is applied in transition regions.
    - Model spectrum is resampled to match the observed wavelength grid
      if sampling differs.
    - Speed of light is taken as c = 299792.458 km/s.

    Examples
    --------
    >>> eval_data = {
    ...     'spec_coef': np.random.rand(26, 7506, 3),
    ...     'version': 2,
    ...     'start': 8.342682,
    ...     'step': 0.00023026,
    ...     'sampling': 1,
    ...     'npix': 1327,
    ...     'mod_start': 8.006368,
    ...     'mod_step': 0.00023026,
    ...     'mod_samp': 1
    ... }
    >>> para = np.array([np.log(5777), 4.44, 0.0])
    >>> model = uly_tgm_eval(eval_data, para, sampling_function='linear')
    >>> print(model.shape)
    (7506, 1)

    """

    # ------------------------------------------------------------------
    # 5.1 Extract model coefficient matrix
    # ------------------------------------------------------------------
    # 5.1.1 Get coefficient matrix with shape (26, 7506, 3)
    spec_coef = eval_data["spec_coef"]

    # ------------------------------------------------------------------
    # 5.2 Calculate model polynomial parameter combinations
    # ------------------------------------------------------------------
    # 5.2.1 Calculate scaled teff, log g
    teff, grav = np.log10(np.exp(para[0])) - 3.7617, para[1] - 4.44
    # 5.2.3 Generate model polynomial parameters
    param = uly_tgm_model_param(
        int(eval_data["version"]),
        teff,
        grav,
        para[2],
    )
    # 5.2.4 Get polynomial coefficient matrix dimension
    np_s = param.shape[0]

    # ------------------------------------------------------------------
    # 5.3 Compute model spectrum using polynomial coefficients
    # ------------------------------------------------------------------
    # 5.3.1 Warm stars (T ≤ 9000 K)
    if teff <= np.log10(9000) - 3.7617:
        t1 = spec_coef[:np_s, :, 0].T
        t1 = np.dot(t1, param[:, 0].reshape(-1, 1))
    # 5.3.2 Hot stars (T ≥ 7000 K)
    if teff >= np.log10(7000) - 3.7617:
        t2 = spec_coef[:np_s, :, 1].T
        t2 = np.dot(t2, param[:, 1].reshape(-1, 1))
    # 5.3.3 Cold stars (T ≤ 4550 K)
    if teff <= np.log10(4550) - 3.7617:
        t3 = spec_coef[:np_s, :, 2].T
        t3 = np.dot(t3, param[:, 2].reshape(-1, 1))

    # ------------------------------------------------------------------
    # 5.4 Interpolate or select appropriate spectrum result
    # ------------------------------------------------------------------
    # 5.4.1 Temperature regime below 7000 K
    if teff <= np.log10(7000) - 3.7617:
        if teff > np.log10(4550) - 3.7617:
            tgm_model_evalhc = t1
        elif teff > np.log10(4000) - 3.7617:
            q = (teff - np.log10(4000) + 3.7617) / (np.log10(4550) - np.log10(4000))
            tgm_model_evalhc = q * t1 + (1.0 - q) * t3
        else:
            tgm_model_evalhc = t3
    # 5.4.2 Temperature regime above 9000 K: use hot spectrum
    elif teff >= np.log10(9000) - 3.7617:
        tgm_model_evalhc = t2
    # 5.4.3 Temperature between 7000 K and 9000 K: interpolate
    else:
        q = (teff - np.log10(7000) + 3.7617) / (np.log10(9000) - np.log10(7000))
        tgm_model_evalhc = q * t2 + (1.0 - q) * t1

    # ------------------------------------------------------------------
    # 5.5 Resample model spectrum to observed wavelength grid
    # ------------------------------------------------------------------
    # 5.5.1 Check if resampling is needed
    C_light, sampling, start, step, npix, mod_samp, mod_start, mod_step, spec_npix = (
        299792.458,
        eval_data.get("sampling", None),
        eval_data.get("start", None),
        eval_data.get("step", None),
        eval_data.get("npix", None),
        eval_data.get("mod_samp", None),
        eval_data.get("mod_start", None),
        eval_data.get("mod_step", None),
        spec_coef.shape[1],
    )

    if (
        (sampling != mod_samp)
        | (start != mod_start)
        | (step != mod_step)
        | (npix != spec_npix)
    ):

        # 5.5.2 Allocate spectrum structure for model
        spec = uly_spect_alloc(
            data=tgm_model_evalhc,
            start=mod_start,
            step=mod_step,
            sampling=mod_samp,
        )

        # 5.5.3 Calculate observed spectrum wavelength range
        if sampling < 2:
            if (start is None) or (step is None) or (npix is None):
                raise ValueError("Wavelength range is not specified in eval_data.")
            # The original IDL computed the upper bound as
            # start + npix * step, which seems incorrect (off
            # by one pixel). The last pixel center should be
            # start + (npix - 1) * step
            wrange = np.array([start, start + (npix - 1) * step])
        if sampling == 0:
            wrange = np.log([start, start + step * (npix - 1)])
            velscale = ((wrange[1] - wrange[0]) / (npix - 1)) * C_light
        # 5.5.4 Convert to linear wavelength if using ln logarithmic sampling
        if sampling == 1:
            wrange = np.exp(wrange)
            # Calculate velocity scale for logarithmic resampling
            velscale = step * C_light
        if sampling == 2:
            wrange = eval_data.get("wavelen", None)
            if wrange is None:
                raise ValueError("Wavelength range is not specified in eval_data.")
            velscale = (np.log(wrange[-1] / wrange[0]) / (npix - 1)) * C_light

        # 5.5.5 Resample model spectrum to observed wavelength grid
        if eval_data["sampling"] == 1:
            # 5.5.5.1 Resample model spectrum wavelength to observed wavelength
            # Example: interpolate from 7506 to 1327 dimensions
            spec = uly_spect_logrebin(
                spec,
                velscale,
                waverange=wrange,
                sampling_function=sampling_function,
                overwrite=True,
            )
        # 5.5.5.2 Current, only support logrebin
        # else:
        #   spec = uly_spect_linrebin(spec, eval_data["step"], sampling_function=sampling_function, waverange=wrange)

        # 5.5.6 Extract resampled model spectrum
        tgm_model_evalhc = spec["data"]

    # ------------------------------------------------------------------
    # 5.6 Optional calibration and LSF convolution (not used in LASP)
    # ------------------------------------------------------------------
    # The following code block is from IDL, not used in LASP.
    # If needed, refer to IDL code
    """
    # 5.6.1 multiply by a black-body spectrum
    calibration = eval_data.get("calibration", None)
    if (calibration is not None) and (calibration == "C"):
        n = tgm_model_evalhc.size
        if sampling == 1:
            wavelength = np.exp(eval_data["start"] + np.arange(n) * eval_data["step"])
        else:
            wavelength = eval_data["start"] + np.arange(n) * eval_data["step"]

        w, c3, c1 = (
            5550.0,
            1.43883e8 / 5550 / np.exp(np.log(para[0])),
            3.74185e19 / 5550**5,
        )
        if c3 < 50:
            bbm = c1 / (np.exp(c3) - 1)
        else:
            bbm = c1 * np.exp(-c3)
        c3, c1 = (
            1.43883e8 / wavelength / np.exp(np.log(para[0])),
            3.74185e19 / wavelength**5 / bbm,
        )
        n1, n2 = np.where(c3 < 50)[0].tolist(), np.where(c3 >= 50)[0].tolist()
        if len(n1) > 0:
            tgm_model_evalhc[n1] = tgm_model_evalhc[n1] * (
                c1[n1] / (np.exp(c3[n1]) - 1)
            )
        if len(n1) < n:
            tgm_model_evalhc[n2] = tgm_model_evalhc[n2] * (c1[n2] / np.exp(-c3[n2]))
    # 5.6.2 Convolve LOSVD in case giving lsf_file
    lsf = eval_data.get("lsf", None)
    if (lsf is not None) and (lsf != "no_lsf"):
        spec = uly_spect_alloc(
            data=tgm_model_evalhc,
            start=eval_data["start"],
            step=eval_data["step"],
            sampling=1,
        )
        tgm_model_evalhc = spec["data"]
    """

    # ------------------------------------------------------------------
    # 5.7 Return wavelength-resampled model spectrum
    # ------------------------------------------------------------------
    return tgm_model_evalhc
