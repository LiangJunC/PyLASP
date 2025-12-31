# -*- coding: utf-8 -*-
# @Time    : 2025/1/4 16:05
# @Author  : ljc
# @FileName: convol_pytorch.py
# @Software: PyCharm
# Update:  2025/11/26 20:41:10


# ======================================================================
# 1. Introduction
# ======================================================================
r"""PyTorch implementation of resolution degradation for LASP-Adam-GPU.

1.1 Purpose
-----------
Implement batch convolution method to degrade model spectral resolution
based on IDL convol core algorithm. This PyTorch implementation is
optimized for GPU acceleration and batch processing, applicable to
LASP-Adam-GPU.

1.2 Functions
-------------
1) convol: Batch processing function for spectral resolution
   degradation. Used in 'uly_fit/uly_fit_conv_poly_pytorch.py'.

1.3 Explanation
---------------
The convol function performs dynamic convolution on input batch high
resolution spectra to degrade their resolution, matching the resolution
of observed spectra.
Steps:
    1) Calculate convolution kernel size for each spectrum based on
       parameters (mu, sigma).
    2) Deduplicate kernel sizes and group by different sizes for
       efficiency.
    3) Generate Gaussian convolution kernels for each group and
       normalize.
    4) Pad input spectrum edges (replicate padding).
    5) Use grouped convolution to batch degrade resolution.
    6) Merge all group results and return.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
from config.config import default_set, set_all_seeds
import torch
import torch.nn.functional as F

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
# 4. Batch spectral resolution degradation function
# ======================================================================
def convol(
    models: TensorLike, kernal_mu: TensorLike, kernal_stds: TensorLike
) -> TensorLike:
    r"""Batch degrade model spectral resolution using Gaussian kernels.

    Parameters
    ----------
    models : torch.Tensor
        shape (batch_size, spectrum_length)
        Batch high-resolution spectra.
    kernal_mu : torch.Tensor
        shape (batch_size, 1)
        Gaussian convolution kernel mean (mu) for each sample, used to
        calculate radial velocity (RV).
    kernal_stds : torch.Tensor
        shape (batch_size, 1)
        Gaussian convolution kernel standard deviation (sigma) for each
        sample, contains instrumental broadening and rotation effects.

    Returns
    -------
    result : torch.Tensor
        shape (batch_size, spectrum_length, 1)
        Batch degraded resolution spectra.

    Notes
    -----
    - Dynamic convolution kernels are generated for each spectrum and
      iteration, with parameters varying across samples.
    - Spectra with identical kernel sizes are grouped together for
      efficient batch processing using grouped convolution.
    - Minimum standard deviation is clamped to 0.1 for numerical
      stability.
    - Replicate padding is applied to minimize edge effects.

    Examples
    --------
    >>> models = torch.randn(100, 200, device=device, dtype=dtype)
    >>> kernal_mu = torch.randn(100, 1, device=device, dtype=dtype)
    >>> kernal_stds = torch.ones(100, 1, device=device, dtype=dtype) * 2.0
    >>> result = convol(models, kernal_mu, kernal_stds)
    >>> print(result.shape)
    torch.Size([100, 200, 1])

    """

    # ------------------------------------------------------------------
    # 4.1 Calculate convolution kernel size
    # ------------------------------------------------------------------
    kernal_stds = torch.clamp(kernal_stds, min=0.1)
    dx = torch.ceil(torch.abs(kernal_mu) + 5.0 * kernal_stds)
    kernel_sizes = (2 * dx + 1).long().view(-1)

    # ------------------------------------------------------------------
    # 4.2 Deduplicate and group kernel sizes
    # ------------------------------------------------------------------
    unique_sizes = torch.unique(kernel_sizes)
    result = torch.zeros_like(models)
    for size in unique_sizes:
        # 4.2.1 Find sample indices corresponding to current kernel size
        indices = torch.where(kernel_sizes == size)[0]
        current_batch_size = len(indices)
        if current_batch_size == 0:
            continue
        # 4.2.2 Get flux, kernel mean, standard deviation and radius
        current_models, current_mu, current_std, current_dx = (
            models[indices, :],
            kernal_mu[indices, :],
            kernal_stds[indices, :],
            dx[indices, :],
        )

        # --------------------------------------------------------------
        # 4.3 Generate gaussian convolution kernel
        # --------------------------------------------------------------
        x = current_dx - torch.arange(size, device=device)
        w = (x - current_mu) / current_std
        w2 = w * w
        mask = torch.abs(w) <= 5.0
        kernal = torch.exp(-0.5 * w2) / (
            torch.sqrt(torch.tensor(2.0 * torch.pi)) * current_std
        )
        kernal = kernal * mask
        kernal = kernal / torch.sum(kernal, dim=1, keepdim=True)

        # --------------------------------------------------------------
        # 4.4 Adjust tensor shapes for grouped convolution
        # --------------------------------------------------------------
        current_models = current_models.view(1, current_batch_size, -1)
        kernal = kernal.view(current_batch_size, 1, -1)

        # --------------------------------------------------------------
        # 4.5 Pad
        # --------------------------------------------------------------
        pad_size = (size - 1) // 2
        padded = F.pad(current_models, (pad_size, pad_size), mode="replicate")

        # --------------------------------------------------------------
        # 4.6 Grouped convolution
        # --------------------------------------------------------------
        conv_result = F.conv1d(padded, kernal, groups=current_batch_size)

        # --------------------------------------------------------------
        # 4.7 Save convolution results of current group
        # --------------------------------------------------------------
        result[indices, :] = conv_result.squeeze(0).squeeze(1)

    # ------------------------------------------------------------------
    # 4.8 Return degraded resolution spectrum
    # ------------------------------------------------------------------
    return result.unsqueeze(-1)
