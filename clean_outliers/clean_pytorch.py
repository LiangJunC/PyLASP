# -*- coding: utf-8 -*-
# @Time    : 2025/3/3 21:37
# @Author  : ljc
# @FileName: clean_pytorch.py
# @Software: PyCharm
# Update:  2025/11/26 20:20:28


# ======================================================================
# 1. Introduction
# ======================================================================
r"""PyTorch implementation of Clean strategy for LASP-Adam-GPU.

1.1 Purpose
-----------
Implement the Clean strategy in LASP-Adam-GPU using PyTorch to clip
outliers in flux residuals between model spectra and observed spectra
to be measured.

1.2 Functions
-------------
1) clean_outliers: Batch processing function implementing three-layer
   outlier Clean strategy.

1.3 Explanation
---------------
The clean_outliers function clips outliers in flux residuals between
model spectra and observed spectra to be measured. Clipping is divided
into 3 layers, each with different purposes.
1.3.1 First Layer Clipping
Steps:
    1) Calculate standard deviation (rbst_sig) of flux residuals (loss)
       between model spectra and observed spectra to be measured.
    2) Identify outliers using flux residual thresholds (3 sigma, 4
       sigma, 5 sigma, 7 sigma).
    3) If detected points exceed 3%, increase threshold to avoid over-
       clipping.
1.3.2 Second Layer Clipping
Steps:
    1) Check if neighbors (1 pixel left and right) of good pixels from
       first layer are bad.
    2) Use lower threshold (2 sigma) to determine if good pixels from
       first layer are slightly outlying.
    3) If neighbors are bad and good points from first layer are
       slightly outlying, clip the point.
1.3.3 Third Layer Clipping
Steps:
    1) Track spectral features iteratively (maximum 20 iterations).
    2) Recalculate standard deviation of flux residuals in each
    iteration.
    3) Find points satisfying five specific conditions.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
from config.config import default_set, set_all_seeds
import torch

# 2.1 Set random seed
set_all_seeds()
# 2.2 Call GPU and specify data type
dtype, device = default_set()
# 2.3 Set default data type
torch.set_default_dtype(dtype)


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
TensorLike = torch.Tensor


# ======================================================================
# 4. Batch outlier cleaning with three-layer strategy
# ======================================================================
def clean_outliers(
    j: int,
    bestfit: TensorLike,
    loss: TensorLike,
    npix: int,
    goodPixels0: TensorLike,
    goodPixels: TensorLike | None = None,
    noise: float = 1.0,
    quiet: bool = True,
) -> TensorLike:
    r"""Perform outlier clipping based on three-layer Clean strategy.

    Parameters
    ----------
    j : int
        Iteration number of Adam optimizer.
    bestfit : torch.Tensor
        shape (group_size, npix) or (group_size, npix, 1)
        model spectrum flux (best-fit model).
    loss : torch.Tensor
        shape (group_size, npix) or (group_size, npix, 1)
        Flux residuals between model and observed spectrum to be
        measured.
    npix : int
        Number of pixels per spectrum.
    goodPixels0 : torch.Tensor
        shape (group_size, npix)
        Good pixel mask determined before Clean, 0 indicates bad pixel,
        1 indicates good pixel. goodPixels0 should be determined based
        on actual conditions, default is all 1s.
    goodPixels : torch.Tensor, optional
        shape (group_size, npix) or (group_size, npix, 1)
        Mask from previous iteration, used to multiply with current
        iteration mask in cumulative mode. If None, create initial mask
        of all 1s.
    noise : float
        Uncertainty of spectrum flux to be measured, used for
        normalizing gradient calculation.
    quiet : bool, optional
        Whether to use quiet mode, outputs statistics of each clipping
        layer when False.

    Returns
    -------
    refining_mask : torch.Tensor
        shape (group_size, npix)
        Mask table containing 0s and 1s (0: bad pixel, 1: good pixel),
        representing final mask after three-layer clipping.

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
    >>> bestfit = torch.randn(10, 3800, dtype=dtype, device=device)
    >>> loss = torch.randn(10, 3800, dtype=dtype, device=device)
    >>> goodPixels0 = torch.ones(10, 3800, dtype=dtype, device=device)
    >>> final_mask = clean_outliers(
    ...     j=0,
    ...     bestfit=bestfit,
    ...     loss=loss,
    ...     npix=3800,
    ...     goodPixels0=goodPixels0,
    ...     quiet=True,
    ... )
    >>> print(final_mask.shape)
    torch.Size([10, 3800])

    """

    # ------------------------------------------------------------------
    # 4.1 Input data preprocessing
    # ------------------------------------------------------------------
    if bestfit.dim() == 3:
        bestfit = bestfit.squeeze(-1)
    if loss.dim() > 2:
        loss = loss.squeeze(-1)
    if isinstance(npix, int) and bestfit.shape[1] != npix:
        npix = bestfit.shape[1]

    # 4.1.1 Get batch size
    batch_size = bestfit.shape[0]

    # ------------------------------------------------------------------
    # 4.2 Initialize for current iteration j
    # ------------------------------------------------------------------
    if goodPixels is not None:
        if goodPixels.dim() == 3:
            goodPixels = goodPixels.squeeze(-1)
    else:
        goodPixels = torch.ones_like(bestfit, dtype=dtype, device=device)
    current_iter_mask = torch.ones((batch_size, npix), dtype=dtype, device=device)
    # 4.2.1 Set threshold, factor, standard deviation
    threshold, facsh, rbst_sig = (
        0.03,
        0.5 if j == 0 else 0.2,
        torch.std(loss * goodPixels, dim=1, keepdim=True, unbiased=True),
    )

    # ------------------------------------------------------------------
    # 4.3 First layer clipping: remove obvious outliers
    # ------------------------------------------------------------------
    # 4.3.1 Calculate model spectrum shifted by one pixel
    bestfit_left, bestfit_right = (
        torch.roll(bestfit, -1, dims=1),
        torch.roll(bestfit, 1, dims=1),
    )
    # 4.3.2 Calculate gradient spectrum
    modelgrd = (
        torch.maximum(
            torch.abs(bestfit - bestfit_left), torch.abs(bestfit - bestfit_right)
        )
        * facsh
        / noise
    )

    # 4.3.3 Create masks for different clipping levels
    mask3, mask4, mask5, mask7 = (
        (torch.abs(loss) - modelgrd) <= (3 * rbst_sig),
        (torch.abs(loss) - modelgrd) <= (4 * rbst_sig),
        (torch.abs(loss) - modelgrd) <= (5 * rbst_sig),
        (torch.abs(loss) - modelgrd) <= (7 * rbst_sig),
    )
    # 4.3.4 Calculate outlier percentages at different thresholds
    npix_goodPixels = torch.sum(goodPixels, dim=1)
    outlier_percent3, outlier_percent4, outlier_percent5, outlier_percent7 = (
        1.0 - torch.sum(mask3.float() * goodPixels, dim=1) / npix_goodPixels,
        1.0 - torch.sum(mask4.float() * goodPixels, dim=1) / npix_goodPixels,
        1.0 - torch.sum(mask5.float() * goodPixels, dim=1) / npix_goodPixels,
        1.0 - torch.sum(mask7.float() * goodPixels, dim=1) / npix_goodPixels,
    )

    # 4.3.5 Select clipping level based on outlier percentage
    # 4.3.5.1 Level 3
    condition3 = (outlier_percent3 <= threshold).unsqueeze(1).to(device)
    current_iter_mask = (
        torch.where(condition3, mask3.float(), current_iter_mask) * goodPixels0
    )

    # 4.3.5.2 Level 4
    condition4 = (
        ((outlier_percent3 > threshold) & (outlier_percent4 <= threshold))
        .unsqueeze(1)
        .to(device)
    )
    current_iter_mask = (
        torch.where(condition4, mask4.float(), current_iter_mask) * goodPixels0
    )

    # 4.3.5.3 Level 5
    condition5 = (
        (
            (outlier_percent3 > threshold)
            & (outlier_percent4 > threshold)
            & (outlier_percent5 <= threshold)
        )
        .unsqueeze(1)
        .to(device)
    )
    current_iter_mask = (
        torch.where(condition5, mask5.float(), current_iter_mask) * goodPixels0
    )

    # 4.3.5.4 Level 7
    condition7 = (
        (
            (outlier_percent3 > threshold)
            & (outlier_percent4 > threshold)
            & (outlier_percent5 > threshold)
        )
        .unsqueeze(1)
        .to(device)
    )
    current_iter_mask = (
        torch.where(condition7, mask7.float(), current_iter_mask) * goodPixels0
    )

    # 4.3.6 Output first layer clipping statistics
    if not quiet:
        # 4.3.6.1 Calculate number of clipped points for each sample
        total_outliers = torch.sum(current_iter_mask == 0, dim=1)
        new_outliers = torch.sum((goodPixels == 1) & (current_iter_mask == 0), dim=1)
        # 4.3.6.2 Print number of clipped points for each sample
        for i in range(batch_size):
            print(
                f"First layer clipping: Sample {i + 1}: Total clipped {total_outliers[i].item()} points, newly added {new_outliers[i].item()} points"
            )

    # ------------------------------------------------------------------
    # 4.4 Second layer clipping: remove edge outliers
    # ------------------------------------------------------------------
    # 4.4.1 Calculate masks for adjacent pixels
    mask_left, mask_right = (
        torch.roll(current_iter_mask, 1, dims=1),
        torch.roll(current_iter_mask, -1, dims=1),
    )
    # 4.4.2 Identify positions with neighboring outliers
    neighbor_outliers = (mask_left == 0) | (mask_right == 0)
    # 4.4.3 Identify points to be clipped in second layer
    neighbor_condition = (
        (current_iter_mask == 1.0)
        & ((torch.abs(loss) - modelgrd) > 2 * rbst_sig)
        & neighbor_outliers
    )

    # 4.4.4 Update mask
    current_iter_mask = current_iter_mask * (~neighbor_condition).float()

    # 4.4.5 Output statistics for first two layers of clipping
    if not quiet:
        # 4.4.5.1 Calculate number of clipped points for each sample
        total_outliers = torch.sum(current_iter_mask == 0, dim=1)
        new_outliers = torch.sum((goodPixels == 1) & (current_iter_mask == 0), dim=1)
        # 4.4.5.2 Print number of clipped points for each sample
        for i in range(batch_size):
            print(
                f"First two layers clipping: Sample {i + 1}: Total clipped {total_outliers[i].item()} points, newly added {new_outliers[i].item()} points"
            )

    # ------------------------------------------------------------------
    # 4.5 Third layer clipping: iteratively track spectral features
    # ------------------------------------------------------------------
    # 4.5.1 Initialize refining mask
    refining_mask = current_iter_mask.clone()
    # 4.5.2 Iteratively refine mask (maximum 20 iterations)
    for k in range(20):
        # 4.5.2.1 Save mask from previous iteration
        prev_refining_mask = refining_mask.clone()
        # 4.5.2.2 Calculate standard deviation of valid pixels
        r_sig = torch.std(loss * prev_refining_mask, dim=1, keepdim=True, unbiased=True)
        # 4.5.2.3 Calculate shifted masks and flux residuals
        mask_forward, loss_forward, mask_backward, loss_backward = (
            torch.roll(refining_mask, 1, dims=1),
            torch.roll(loss, 1, dims=1),
            torch.roll(refining_mask, -1, dims=1),
            torch.roll(loss, -1, dims=1),
        )

        # 4.5.2.4 Check outlier conditions
        cond1_forward, cond2_forward, cond3, cond4_forward = (
            (refining_mask == 1) & (mask_forward == 0),
            loss_forward * loss > 0,
            torch.abs(loss) - modelgrd > r_sig,
            torch.abs(loss) <= torch.abs(loss_forward),
        )
        cond1_backward, cond2_backward, cond4_backward = (
            (refining_mask == 1) & (mask_backward == 0),
            loss_backward * loss > 0,
            torch.abs(loss) <= torch.abs(loss_backward),
        )

        # 4.5.2.5 Determine outliers
        forward_outliers, backward_outliers = (
            cond1_forward & cond2_forward & cond3 & cond4_forward,
            cond1_backward & cond2_backward & cond3 & cond4_backward,
        )

        # 4.5.2.6 Update refining mask
        refining_mask = (
            refining_mask * (~(forward_outliers | backward_outliers)).float()
        )

        # 4.5.2.7 Check for convergence
        if torch.all(refining_mask == prev_refining_mask):
            break

    # ------------------------------------------------------------------
    # 4.6 Output statistics for three-layer clipping
    # ------------------------------------------------------------------
    if not quiet:
        # 4.6.1 Calculate number of clipped points for each sample
        total_outliers = torch.sum(refining_mask == 0, dim=1)
        new_outliers = torch.sum((goodPixels == 1) & (refining_mask == 0), dim=1)
        # 4.6.2 Print number of clipped points for each sample
        for i in range(batch_size):
            print(
                f"All three layers clipping: Sample {i + 1}: Total clipped {total_outliers[i].item()} points, newly added {new_outliers[i].item()} points"
            )

    # ------------------------------------------------------------------
    # 4.7 Return final MASK
    # ------------------------------------------------------------------
    return refining_mask
