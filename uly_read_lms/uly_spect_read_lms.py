# -*- coding: utf-8 -*-
# @Time    : 03/12/2024 19.52
# @Author  : ljc
# @FileName: uly_spect_read_lms.py
# @Software: PyCharm
# Update:  2025/11/19 20:25:46


"""Python conversion of reading spectra for LASP-CurveFit.

1.1 Purpose
-----------
A demo, read LAMOST FITS file and construct a spectrum dictionary
structure, converted from IDL uly_spect_read.pro implementation.

1.2 Functions
-------------
1) uly_spect_read_lss: Read LAMOST FITS file and return spectrum
   dictionary structure containing LAMOST spectral information.
2) uly_spect_read_lms: Processes the LAMOST structure from
   uly_spect_read_lss and returns an updated structure restricted
   to the specified wavelength range.

1.3 Explanation
---------------
This module provides functions to read and process LAMOST spectroscopic
data for PyLASP.
Steps:
    1) Read LAMOST FITS file.
    2) Convert log10 vacuum wavelength to ln air wavelength.
    3) Extract LAMOST data within specified wavelength range.
    4) Set good pixel indices.
    5) Return spectrum dictionary structure containing.

1.4 Notes
---------
- If you need to use your own spectral data, please set
  uly_spect_read_lss and uly_spect_read_lms to read your
  own spectral FITS file.
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
from astropy.io import fits
from uly_read_lms.uly_spect_extract import uly_spect_extract
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
ArrayLike = np.ndarray


# ======================================================================
# 4. LAMOST FITS file reading function (A demo)
# ======================================================================
def uly_spect_read_lss(
    file_in: str,
    public: bool = False,
    flux_median: bool = False,
    err_bool: bool = False,
    mask_bool: bool = False,
) -> dict:
    """Read LAMOST FITS file and return spectrum dictionary structure.

    Parameters
    ----------
    file_in : str
        Path to LAMOST FITS file.
    public : bool, optional
        Whether the LAMOST spectrum is public or internal data.
        - False: Internal data (default)
        - True: Public data
    flux_median : bool, optional
        Whether to divide by spectrum flux median, which can improve
        optimization efficiency and potentially avoid numerical
        optimization errors.
        - False: Do not divide by flux median (default)
        - True: Divide by flux median
        Note: When median is unstable, consider using other quantiles.
    err_bool : bool, optional
        Whether to extract flux error from LAMOST FITS file. Default is
        False.
    mask_bool : bool, optional
        Whether to extract mask from LAMOST FITS file. Default is False.

    Returns
    -------
    SignalIn : dict
        Spectrum dictionary structure containing LAMOST spectral
        information.

    Notes
    -----
    - LASP uses uly_spect_read_lss function to process spectra and
      return flux dictionary structure. For other data reading methods,
      refer to the original IDL code.
    - vacuum=True indicates LAMOST wavelengths are vacuum wavelengths.
    - The dof_factor is set to 1 for LASP.

    Examples
    --------
    >>> from file_paths import TEST_DATA_DIR
    >>> signal = uly_spect_read_lss(
    ...     file_in=TEST_DATA_DIR() + "spec-57035-HD111424N200151V01_sp14-102.fits",
    ...     public=True,
    ...     flux_median=True
    ... )
    >>> print(signal['sampling'])
    1
    >>> print(signal['data'].shape)
    (3909,)

    """

    # ------------------------------------------------------------------
    # 4.1 Read LAMOST FITS file and extract data
    # ------------------------------------------------------------------
    f = fits.open(file_in)

    # ------------------------------------------------------------------
    # 4.2 Extract LAMOST flux data based on data type
    # ------------------------------------------------------------------
    h = f[0].header
    # 4.2.1 If LAMOST spectrum is internal data
    if public is False:
        spec = f[0].data[0]
        if err_bool:
            ivar = f[0].data[1]
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf)
        else:
            err = None
        if mask_bool:
            # Note: In LAMOST data, mask = 0 marks a valid flux pixel;
            #       however, in our subsequent processing we use mask = 1
            #       to denote valid pixels.
            mask = (f[0].data[3] == 0).astype(np.uint8)
        else:
            mask = None
        if flux_median is True:
            m = np.max([np.nanmedian(spec), 1])
            spec = spec / m
            if err_bool:
                err = err / m

    # 4.2.2 If LAMOST spectrum is public data
    if public is True:
        spec = f[1].data["FLUX"][0]
        if err_bool:
            ivar = f[1].data["IVAR"][0]
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf)
        else:
            err = None
        if mask_bool:
            # Note: In LAMOST data, mask = 0 marks a valid flux pixel;
            #       however, in our subsequent processing we use mask = 1
            #       to denote valid pixels.
            mask = (f[1].data["ANDMASK"][0] == 0).astype(np.uint8)
        else:
            mask = None
        if flux_median is True:
            m = np.max([np.nanmedian(spec), 1])
            spec = spec / m
            if err_bool:
                err = err / m

    # 4.2.3 Close file
    f.close()

    # ------------------------------------------------------------------
    # 4.3 Extract wavelength information from FITS header
    # ------------------------------------------------------------------
    # 4.3.1 Note: LAMOST's sampling is 1
    sampling, vacuum, dof_factor, crval, step, crpix = (
        h["DC-FLAG"],
        h["VACUUM"],
        1,
        np.double(h["CRVAL1"]),
        np.double(h["CD1_1"]),
        np.double(h["CRPIX1"]),
    )
    start = crval - (crpix - 1) * step

    # ------------------------------------------------------------------
    # 4.4 Convert LAMOST's log10 vacuum wavelength to ln air wavelength
    # ------------------------------------------------------------------
    if (sampling == 1) and (start <= 5):
        start *= np.log(10)
        step *= np.log(10)
    if (sampling == 1) and vacuum:
        start -= 0.00028

    # ------------------------------------------------------------------
    # 4.5 Construct and return spectrum dictionary
    # ------------------------------------------------------------------
    SignalIn = {
        "title": file_in,
        "hdr": h,
        "data": spec,
        "err": err,
        "wavelen": None,
        "mask": mask,
        "start": start,
        "step": step,
        "sampling": sampling,
        "dof_factor": dof_factor,
    }

    # ------------------------------------------------------------------
    # 4.6 Return LAMOST spectrum dictionary
    # ------------------------------------------------------------------
    return SignalIn


# ======================================================================
# 5. LAMOST data processing function
# ======================================================================
def uly_spect_read_lms(
    lmin: list,
    lmax: list,
    file_in: str,
    public: bool = False,
    flux_median: bool = False,
    err_bool: bool = False,
    mask_bool: bool = False,
) -> dict:
    """Return the spectrum within the specified wavelength range.

    Parameters
    ----------
    lmin : list
        Minimum wavelength value for fitting range. Currently supports
        single value (e.g., lmin=[4200]). Segmented lists not
        supported (implementation is not difficult, but requires
        modifying code or MASK settings).
    lmax : list
        Maximum wavelength value for fitting range. Currently supports
        single value (e.g., lmax=[5700]). Segmented lists not supported.
    file_in : str
        Path to LAMOST FITS file.
    public : bool, optional
        Whether the LAMOST spectrum is public or internal data.
        - False: Internal data
        - True: Public data
    flux_median : bool, optional
        Whether to divide by spectrum flux median, which can improve
        optimization efficiency and potentially avoid numerical
        optimization errors.
        - False: Do not divide by flux median
        - True: Divide by flux median
    err_bool : bool, optional
        Whether to extract flux error from LAMOST FITS file. Default is
        False.
    mask_bool : bool, optional
        Whether to extract mask from LAMOST FITS file. Default is False.

    Returns
    -------
    SignalOut : dict
        Updated spectrum structure in specified wavelength range.

    Raises
    ------
    TypeError
        - If lmin or lmax is not a list containing exactly one element.
    ValueError
        - lmin and lmax leave too few valid pixels after masking.

    Examples
    --------
    >>> from file_paths import TEST_DATA_DIR
    >>> inspectr = uly_spect_read_lms(
    ...     lmin=[4200],
    ...     lmax=[5700],
    ...     file_in=TEST_DATA_DIR() + "spec-57035-HD111424N200151V01_sp14-102.fits",
    ...     public=True,
    ...     flux_median=True
    ... )
    >>> print(inspectr["data"].shape)
    (1328,)

    """

    # ------------------------------------------------------------------
    # 5.1 Read input spectrum structure
    # ------------------------------------------------------------------
    SignalOut = uly_spect_read_lss(
        file_in=file_in,
        public=public,
        flux_median=flux_median,
        err_bool=err_bool,
        mask_bool=mask_bool,
    )

    # ------------------------------------------------------------------
    # 5.2 Update the spectrum structure to the specified wavelength
    #     range
    # ------------------------------------------------------------------
    if not all(isinstance(param, list) and len(param) == 1 for param in [lmin, lmax]):
        raise TypeError(
            "Both 'lmin' and 'lmax' must be lists containing exactly one element, e.g., [4200] and [5700]."
        )
    SignalOut = uly_spect_extract(
        SignalIn=SignalOut, waverange=[lmin[0], lmax[0]], overwrite=True
    )

    # ------------------------------------------------------------------
    # 5.3 Extract spectrum data
    # ------------------------------------------------------------------
    flux, flux_err, goodpix = (
        SignalOut.get("data", None),
        SignalOut.get("err", None),
        SignalOut.get("goodpix", None),
    )
    ntot = len(flux)

    # ------------------------------------------------------------------
    # 5.4 Mask invalid flux values (NaN or infinity)
    # ------------------------------------------------------------------
    # 5.4.1 Detect invalid flux values
    good_flux_index, nans_flux_index = (
        np.where(np.isfinite(flux))[0],
        np.where(~np.isfinite(flux))[0],
    )
    cnt_flux, nnans_flux = (
        len(good_flux_index),
        len(nans_flux_index),
    )

    if nnans_flux > 0:
        # 5.4.1.1 If goodpix is None, treat finite values as good pixels
        if goodpix is None:
            goodpix = good_flux_index
        else:
            maskI = np.zeros(ntot, dtype=np.uint8)
            maskI[goodpix] = 1
            maskI[good_flux_index] += 1
            goodpix = np.where(maskI == 2)[0]

        # 5.4.1.2 Fill invalid flux values using the next valid value
        next_ = nans_flux_index + 1
        if next_[len(next_) - 1] == ntot:
            next_[len(next_) - 1] = nans_flux_index[len(next_) - 1]
        SignalOut["data"][nans_flux_index] = flux[next_]

        # 5.4.2 Re-check for remaining invalid flux values
        nans = np.where(~np.isfinite(SignalOut["data"]))[0]
        nnans = len(nans)
        if nnans > 0:
            # 5.4.2.1 Fill invalid flux values using the previous
            #         valid value
            prev = nans - 1
            if prev[0] < 0:
                # IDL used nans[1], which may be out-of-bounds when
                # nnans = 1. Using nans[0] avoids IndexError; any
                # remaining NaN is handled below.
                prev[0] = nans[0]
            SignalOut["data"][nans] = SignalOut["data"][prev]

        # 5.4.3 Final check: if any NaN values remain
        nans = np.where(~np.isfinite(SignalOut["data"]))[0]
        nnans = len(nans)
        if nnans > 0:
            SignalOut["data"][nans] = 0
    else:
        goodpix = np.arange(ntot)

    # ------------------------------------------------------------------
    # 5.5 Mask invalid flux error values (NaN or infinity)
    # ------------------------------------------------------------------
    # 5.5.1 Detect invalid flux error values (if err_bool is True)
    if err_bool:
        good_flux_err_index, nans_flux_err_index = (
            np.where(np.isfinite(flux_err))[0],
            np.where(~np.isfinite(flux_err))[0],
        )
        cnt_flux_err, nnans_flux_err = (
            len(good_flux_err_index),
            len(nans_flux_err_index),
        )
        if nnans_flux_err > 0:
            # 5.5.1.1 If goodpix is None, treat finite error values as
            #         good pixels
            maskI = np.zeros(ntot, dtype=np.uint8)
            maskI[goodpix] = 1
            maskI[good_flux_err_index] += 1
            goodpix = np.where(maskI == 2)[0]

            # 5.5.1.2 Fill invalid error values using the next valid
            #         error value
            next_ = nans_flux_err_index + 1
            if next_[len(next_) - 1] == ntot:
                next_[len(next_) - 1] = nans_flux_err_index[len(next_) - 1]
            SignalOut["err"][nans_flux_err_index] = flux_err[next_]

            # 5.5.2 Re-check for remaining invalid error values
            nans = np.where(~np.isfinite(SignalOut["err"]))[0]
            nnans = len(nans)
            if nnans > 0:
                # 5.5.2.1 Fill invalid error values using the previous
                #         valid error value
                prev = nans - 1
                if prev[0] < 0:
                    # IDL used nans[1], which may be out-of-bounds when
                    # nnans = 1. Using nans[0] avoids IndexError; any
                    # remaining NaN is handled below.
                    prev[0] = nans[0]
                SignalOut["err"][nans] = SignalOut["err"][prev]

            # 5.5.3 Final check: if any NaN values remain in the
            #       error array
            nans = np.where(~np.isfinite(SignalOut["err"]))[0]
            nnans = len(nans)
            if nnans > 0:
                SignalOut["err"][nans] = 0

    # ------------------------------------------------------------------
    # 5.6 Ensure that there are enough valid pixels after masking
    # ------------------------------------------------------------------
    if len(goodpix) <= 3:
        raise ValueError("lmin and lmax leave too few valid pixels after masking.")
    if goodpix[0] == 0:
        SignalOut["goodpix"] = goodpix[1:]
    if goodpix[-1] == ntot - 1:
        SignalOut["goodpix"] = goodpix[:-1]

    # ------------------------------------------------------------------
    # 5.7 Return spectrum data within the specified wavelength range
    # ------------------------------------------------------------------
    return SignalOut
