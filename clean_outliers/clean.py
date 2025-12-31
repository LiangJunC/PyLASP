# -*- coding: utf-8 -*-
# @Time    : 2025/11/20 22:06
# @Author  : ljc
# @FileName: clean.py
# @Software: Pycharm
# Update:  2025/11/26 15:48:05


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Python implementation of Clean strategy for LASP-CurveFit-CPU.

1.1 Purpose
-----------
Implement the Clean strategy in LASP-CurveFit-CPU using NumPy to clip
outliers in flux residuals between model spectra and observed spectra
to be measured.

1.2 Functions
-------------
1) clean_outliers: Single spectrum processing function implementing
   three-layer outlier Clean strategy.

1.3 Explanation
---------------
The clean_outliers function clips outliers in flux residuals between
model spectra and observed spectra to be measured. Clipping is divided
into 3 layers, each with different purposes.

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
# 4. Single spectrum outlier cleaning with three-layer strategy
# ======================================================================
def clean_outliers(
    j: int,
    npix: int,
    bestfit: ArrayLike,
    flux_err: ArrayLike,
    resc: ArrayLike,
    goodPixels: ArrayLike,
    goodPixels0: ArrayLike,
    first_stage_outer_number: int,
    second_stage_outer_number: int,
    third_stage_outer_number: int,
    quiet: bool,
) -> tuple[float, ArrayLike, int, int, int]:
    r"""Perform outlier clipping based on Clean strategy.

    Parameters
    ----------
    j : int
        The number of times curve_fit is reused to infer stellar
        parameters.
    npix : int
        Number of pixels per spectrum.
    bestfit : np.ndarray
        shape (npix,)
        Model spectrum flux (best-fit model).
    flux_err : np.ndarray
        shape (npix,)
        Uncertainty of spectrum flux to be measured, used for
        normalizing gradient calculation.
    resc : np.ndarray
        shape (npix,)
        Flux residuals between model spectrum and observed spectrum to
        be measured.
    goodPixels : np.ndarray
        shape (n_good,)
        Indices of good pixels from previous iteration.
    goodPixels0 : np.ndarray
        shape (n_good0,)
        Indices of good pixels determined before Clean, should be
        determined based on actual conditions.
    first_stage_outer_number : int
        Number of outliers clipped in first layer.
    second_stage_outer_number : int
        Number of outliers clipped in second layer.
    third_stage_outer_number : int
        Number of outliers clipped in third layer.
    quiet : bool
        Whether to use quiet mode, outputs statistics of each clipping
        layer when False.

    Returns
    -------
    rbst_sig : float
        Standard deviation of flux residuals.
    goodPixels : np.ndarray
        shape (n_good_final,)
        Indices of good pixels after three-layer clipping.
    first_stage_outer_number : int
        Number of outliers clipped in first layer.
    second_stage_outer_number : int
        Number of outliers clipped in second layer.
    third_stage_outer_number : int
        Number of outliers clipped in third layer.

    Notes
    -----
    - First-layer clipping removes obvious outliers (identify cosmic
      rays, erroneous data points, etc.).
    - Second-layer clipping further removes outliers from good pixels
      identified in the first layer (remove "edge" pixels related to
      primary outliers).
    - Third-layer clipping further removes outliers from good pixels
      identified in the second layer (completely track and remove
      entire spectral features such as emission lines).

    Examples
    --------
    >>> npix = 3800
    >>> np.random.seed(666)
    >>> bestfit = np.random.randn(npix)
    >>> flux_err = np.ones(npix)
    >>> resc = np.random.randn(npix)
    >>> goodPixels0 = np.arange(npix)
    >>> goodPixels = np.arange(npix)
    >>> rbst_sig, goodPixels, m, m2, m3 = clean_outliers(
    ...     j=0,
    ...     npix=npix,
    ...     bestfit=bestfit,
    ...     flux_err=flux_err,
    ...     resc=resc,
    ...     goodPixels=goodPixels,
    ...     goodPixels0=goodPixels0,
    ...     first_stage_outer_number=0,
    ...     second_stage_outer_number=0,
    ...     third_stage_outer_number=0,
    ...     quiet=True,
    ... )
    >>> print(np.round(rbst_sig, 2), goodPixels[:3], m, m2, m3)
    1.01 [0 1 2] 0 0 0

    """

    # ------------------------------------------------------------------
    # 4.1 First layer clipping: remove obvious outliers
    # ------------------------------------------------------------------
    # 4.1.1 Calculate gradient spectrum
    # 4.1.1.1 Set threshold and factor
    facsh = 0.5 if j == 0 else 0.2
    # 4.1.1.2 Calculate model spectrum shifted by one pixel
    bestfit_left, bestfit_right = np.roll(bestfit, -1), np.roll(bestfit, 1)
    # 4.1.1.3 Calculate gradient spectrum
    modelgrd = (
        np.maximum(np.abs(bestfit - bestfit_left), np.abs(bestfit - bestfit_right))
        * facsh
        / flux_err
    )
    # 4.1.2 Calculate standard deviation of flux residuals
    # 4.1.2.1 Calculate standard deviation for good pixels
    rbst_sig = np.std(resc[goodPixels], ddof=1)
    # 4.1.2.2 Create mask for 3 sigma clipping
    mask_condition = (np.abs(resc[goodPixels]) - modelgrd[goodPixels]) > (3 * rbst_sig)
    tmp = np.where(mask_condition)[0]
    m = len(tmp)
    clip_level = 3

    # 4.1.3 Return if no outliers detected
    if m == 0:
        return (
            rbst_sig,
            goodPixels,
            first_stage_outer_number,
            second_stage_outer_number,
            third_stage_outer_number,
        )

    # 4.1.4 Adjust clipping level based on outlier percentage
    # 4.1.4.1 Set more relaxed 4 sigma threshold
    if m > 0.03 * len(goodPixels):
        mask_condition = (np.abs(resc[goodPixels]) - modelgrd[goodPixels]) > (
            4 * rbst_sig
        )
        tmp = np.where(mask_condition)[0]
        m = len(tmp)
        clip_level = 4
        # 4.1.4.2 Set more relaxed 5 sigma threshold
        if m > 0.03 * len(goodPixels):
            mask_condition = (np.abs(resc[goodPixels]) - modelgrd[goodPixels]) > (
                5 * rbst_sig
            )
            tmp = np.where(mask_condition)[0]
            m = len(tmp)
            clip_level = 5
            # 4.1.4.3 Set more relaxed 7 sigma threshold
            if m > 0.03 * len(goodPixels):
                mask_condition = (np.abs(resc[goodPixels]) - modelgrd[goodPixels]) > (
                    7 * rbst_sig
                )
                tmp = np.where(mask_condition)[0]
                m = len(tmp)
                clip_level = 7

    # 4.1.5 Create mask and set mask to 0 for m removed flux points
    # 4.1.5.1 Initialize mask with all pixels
    mask = np.zeros(npix)
    mask[goodPixels0] = 1
    # 4.1.5.2 Set mask to 0 for outliers detected at clip_level
    if m > 0:
        mask_condition = (np.abs(resc[goodPixels0]) - modelgrd[goodPixels0]) > (
            clip_level * rbst_sig
        )
        tmp = np.where(mask_condition)[0]
        tmp = np.array(goodPixels0)[tmp]
        mask[tmp] = 0

    # ------------------------------------------------------------------
    # 4.2 Second layer clipping: remove edge outliers
    # ------------------------------------------------------------------
    # 4.2.1 Initialize second layer outlier counter
    m2 = 0
    # 4.2.2 Check neighbors of outliers from first layer
    if m > 0:
        # 4.2.2.1 Get neighboring pixel indices
        near = np.concatenate([tmp - 1, tmp + 1])
        near = near[(near >= 0) & (near < npix)]
        # 4.2.2.2 Identify neighboring outliers
        mask_condition = (np.abs(resc[near]) - modelgrd[near] > 2 * rbst_sig) & (
            mask[near] == 1
        )
        nnnn = np.where(mask_condition)[0]
        m2 = len(nnnn)
        # 4.2.2.3 Set mask to 0 for neighboring outliers
        if m2 > 0:
            mask[near[nnnn]] = 0

    # ------------------------------------------------------------------
    # 4.3 Third layer clipping: iteratively track spectral features
    # ------------------------------------------------------------------
    # 4.3.1 Initialize third layer outlier counter
    m3 = 0
    if m > 0:
        # 4.3.2 Iteratively refine mask (maximum 20 iterations)
        for k in range(1, 21):
            # 4.3.2.1 Recalculate standard deviation of valid pixels
            r_sig = np.std(resc[mask == 1], ddof=1)
            # 4.3.2.2 Move forward: check outliers in forward direction
            # 4.3.2.2.1 Calculate shifted flux residuals and masks
            ss, mask_shifted_back = np.roll(resc, 1), np.roll(mask, 1)
            # 4.3.2.2.2 Check outlier conditions
            cond1, cond2, cond3, cond4 = (
                (mask == 1) & (mask_shifted_back == 0),
                (ss * resc > 0),
                (np.abs(resc) - modelgrd > r_sig),
                (np.abs(resc) <= np.abs(ss)),
            )
            # 4.3.2.2.3 Identify forward outliers
            nnn1 = np.where(cond1 & cond2 & cond3 & cond4)[0]
            mmm1 = len(nnn1)
            # 4.3.2.3 Move backward: check outliers in backward direction
            # 4.3.2.3.1 Calculate shifted flux residuals and masks
            ss, mask_shifted_forward = np.roll(resc, -1), np.roll(mask, -1)
            # 4.3.2.3.2 Check outlier conditions
            cond1, cond2, cond3, cond4 = (
                (mask == 1) & (mask_shifted_forward == 0),
                (ss * resc > 0),
                (np.abs(resc) - modelgrd > r_sig),
                (np.abs(resc) <= np.abs(ss)),
            )
            # 4.3.2.3.3 Identify backward outliers
            nnn2 = np.where(cond1 & cond2 & cond3 & cond4)[0]
            mmm2 = len(nnn2)
            # 4.3.2.4 Set mask to 0 for anomalous points
            if mmm1 > 0:
                mask[nnn1] = 0
            if mmm2 > 0:
                mask[nnn2] = 0
            # 4.3.2.5 Check for convergence: exit loop if no outliers
            if (mmm1 <= 0) and (mmm2 <= 0):
                break
            # 4.3.2.6 Update third layer outlier counter
            m3 += mmm1 + mmm2

    # ------------------------------------------------------------------
    # 4.4 Update good pixel list and output statistics
    # ------------------------------------------------------------------
    # 4.4.1 Update good pixel list and outlier counters
    if m > 0:
        # 4.4.1.1 Output statistics if quiet mode is False
        if not quiet:
            print(
                f"(6). Number of clipped outliers: {m}+{m2}+{m3}"
                f" out of {len(goodPixels0)}\n"
                "--------------------------------------------------------------------"
            )
        # 4.4.1.2 Update good pixel list
        goodPixels = np.where(mask == 1)[0]
        (
            first_stage_outer_number,
            second_stage_outer_number,
            third_stage_outer_number,
        ) = (m, m2, m3)

    # ------------------------------------------------------------------
    # 4.5 Return final results
    # ------------------------------------------------------------------
    return (
        rbst_sig,
        goodPixels,
        first_stage_outer_number,
        second_stage_outer_number,
        third_stage_outer_number,
    )
