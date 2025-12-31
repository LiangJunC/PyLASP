# -*- coding: utf-8 -*-
# @Time    : 2025/1/2 11:19
# @Author  : ljc
# @FileName: xrebin_pytorch.py
# @Software: PyCharm
# Update:  2025/11/26 22:21:10


# ============================================================================
# 1. Introduction
# ============================================================================
r"""PyTorch implementation of spectral resampling for LASP-Adam-GPU.

1.1 Purpose
-----------
Implement spectral data resampling using "cumulative-interpolate-
differentiate" method based on IDL xrebin.pro core algorithm. This
PyTorch implementation is optimized for GPU acceleration and supports
only linear interpolation, applicable to LASP-Adam-GPU.

1.2 Functions
-------------
1) xrebin: Batch processing function for spectral resampling. Used in
   'uly_fit/uly_fit_conv_poly_pytorch.py'.

1.3 Explanation
---------------
The xrebin function resamples input spectral data using "cumulative-
interpolate-differentiate" method. Complete resampling includes 5 steps,
where steps 1 and 5 are calculated in uly_tgm_group/ulyss_pytorch.py
and uly_tgm_group/tutorial_LASP_Adam_GPU.ipynb.
Steps:
    1) Calculate input/output wavelength bin edges (xin, xout) in
       uly_fit/ulyss_pytorch.py, saved as pt file by
       data_to_pt/data_to_pt.py.
    2) Accumulate input flux values (cumulative sum).
    3) Interpolate cumulative spectrum onto new wavelength grid.
    4) Differentiate interpolated result to get differential spectrum.
    5) Pass to uly_tgm_group/tutorial_LASP_Adam_GPU.ipynb to convert
       "cumulative-interpolate-differentiate" differential spectrum to
       "integrate-interpolate-derivative" spectrum.

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
# 4. Wavelength resampling function
# ======================================================================
def xrebin(xin: TensorLike, yin: TensorLike, xout: TensorLike) -> TensorLike:
    r"""Resample spectrum using 'cumulative-interpolate-differentiate'.

    Parameters
    ----------
    xin : torch.Tensor
        shape (group_size, n+1)
        Input wavelength bin edges. n+1 edges define n bins.
    yin : torch.Tensor
        shape (group_size, n)
        Input flux. n flux values correspond to n bins.
    xout : torch.Tensor
        shape (group_size, m+1)
        Output wavelength bin edges. m+1 edges define m bins.

    Returns
    -------
    yout : torch.Tensor
        shape (group_size, m)
        Resampled flux. m flux values correspond to m output bins.

    Notes
    -----
    - This function implements the core resampling algorithm using
      cumulative sum, linear interpolation, and differentiation.
    - The output flux from this function represents the "cumulative-
      interpolate-differentiate" result, which needs to be converted
      using the flat factor to obtain the "integrate-interpolate-
      derivative" result.
    - Only linear interpolation is supported in this implementation.
    - All spectra in the batch share the same wavelength grid xin
      and xout (i.e., rows are identical).

    Examples
    --------
    >>> xin = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype, device=device)
    >>> yin = torch.tensor([[1.0, 2.0, 1.0]], dtype=dtype, device=device)
    >>> xout = torch.tensor([[1.2, 2.1, 3.2, 3.5]], dtype=dtype, device=device)
    >>> yout = xrebin(xin, yin, xout)
    >>> print(yout.shape)
    torch.Size([1, 3])

    """

    # ------------------------------------------------------------------
    # 4.1 Find position indices of xout points in xin
    # ------------------------------------------------------------------
    indices = torch.searchsorted(xin, xout)

    # ------------------------------------------------------------------
    # 4.2 Accumulate input flux values
    # ------------------------------------------------------------------
    flux = torch.cat([torch.zeros_like(yin[:, :1]), torch.cumsum(yin, dim=1)], dim=1)

    # ------------------------------------------------------------------
    # 4.3 Gather left and right boundary points for interpolation
    # ------------------------------------------------------------------
    w1_left, w1_right, flux_left, flux_right = (
        torch.gather(xin, 1, indices - 1),
        torch.gather(xin, 1, indices),
        torch.gather(flux, 1, indices - 1),
        torch.gather(flux, 1, indices),
    )

    # ------------------------------------------------------------------
    # 4.4 Linear interpolation to new wavelength grid
    # ------------------------------------------------------------------
    flux_interpolated = flux_left + (flux_right - flux_left) * (xout - w1_left) / (
        w1_right - w1_left
    )

    # ------------------------------------------------------------------
    # 4.5 Differentiate to obtain resampled spectrum
    # ------------------------------------------------------------------
    yout = (torch.roll(flux_interpolated, -1, dims=1) - flux_interpolated)[:, :-1]

    # ------------------------------------------------------------------
    # 4.6 Return resampled spectrum
    # ------------------------------------------------------------------
    return yout
