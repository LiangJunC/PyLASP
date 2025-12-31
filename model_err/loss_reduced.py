# -*- coding: utf-8 -*-
# @Time    : 2025/2/25 15:54
# @Author  : ljc
# @FileName: loss_reduced.py
# @Software: PyCharm
# Update:  2025/11/26 20:08:41


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Flux residual calculation for error analysis based on Jacobian and
    Hessian matrices.

1.1 Purpose
-----------
Compute per-spectrum flux-fitting residuals between observations and
model spectra. These residuals are used to build the Jacobian
matrix, approximate the Hessian, and estimate parameter errors for
each sample in LASP-Adam-GPU.

1.2 Functions
-------------
1) loss_reduced: Compute reduced residuals (observed − model) or the
   corresponding summed-squared residuals, depending on whether the
   Jacobian or Hessian is being constructed.

1.3 Explanation
---------------
The loss_reduced function performs the following steps for a group
of observed spectra:

    1) Generate model spectra from polynomial spectral emulator
       coefficients and internal best-fit parameters.
    2) Rebin the model spectra from the wavelength grid to the
       observed wavelength grid using xrebin.
    3) Convolve the rebinned spectra to match the effective
       resolution of the observed spectra.
    4) Fit a multiplicative Legendre polynomial to match the continuum
       using mregress_batch_cholesky.
    5) Compute flux residuals:
       - For Jacobian-based analysis: return per-pixel residuals.
       - For Hessian-based analysis: return the sum of squared residuals
         for each spectrum.

1.4 Notes
---------
- By default, the function is used in Jacobian mode
  (Jacobian=True, Hessian=False), returning per-pixel residuals
  for each spectrum.
- When Hessian=True, the function returns the summed-squared
  residuals per spectrum, which are typically used to approximate the
  Hessian or to compute scalar loss values.
- The goodPixels_final argument can be used to apply a cleaned mask
  (e.g. to exclude outliers). If it is False, the default "No Clean"
  strategy is used (no outlier rejection).
- This module uses 'warnings.filterwarnings("ignore")' to globally
  suppress Python warnings for cleaner large-scale runs. If you
  rely on warnings from NumPy/SciPy or other libraries, please
  comment out or adjust this line in your environment.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
from config.config import default_set, set_all_seeds
import torch
from legendre_polynomial.mregress_pytorch import mregress_batch_cholesky
from resolution_reduction.convol_pytorch import convol
from uly_tgm_eval.uly_tgm_eval_pytorch import uly_tgm_eval
from WRS.xrebin_pytorch import xrebin
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# 2.1 Set random seed
set_all_seeds()
# 2.2 Call GPU and specify data type
dtype, device = default_set()
# 2.3 Set default data type
torch.set_default_dtype(dtype)


# ======================================================================
# 3. Type definitions and constants for better code readability
# ======================================================================
TensorLike = torch.Tensor


# ======================================================================
# 4. Compute per-spectrum flux-fitting residuals
# ======================================================================
def loss_reduced(
    best_params: TensorLike,
    specs: TensorLike,
    spec_coef: TensorLike,
    borders_: TensorLike,
    NewBorders_: TensorLike,
    flat: TensorLike,
    leg_array_: TensorLike,
    goodPixels_final: TensorLike | bool = False,
    Jacobian: bool = True,
    Hessian: bool = False,
) -> TensorLike:
    r"""Compute flux-fitting residuals for each observed spectrum.

    Parameters
    ----------
    best_params : torch.Tensor
        Best-fit internal parameters obtained by Adam, typically of
        shape (group_size, 5), where:
            - best_params[:, 0:3] are internal parameters corresponding
              to (Teff, log g, [Fe/H]).
            - best_params[:, 3] is the first LOSVD parameter (mu).
            - best_params[:, 4] is the second LOSVD parameter (sigma).
    specs : torch.Tensor
        Observed flux spectra, shape (group_size, M).
    spec_coef : torch.Tensor
        Polynomial spectral emulator coefficients.
    borders_ : torch.Tensor
        Integration borders of the model spectra (input grid).
    NewBorders_ : torch.Tensor
        Integration borders of the rebinned spectra, corresponding to
        the observed wavelength grid.
    flat : torch.Tensor
        Scaling factor proportional to
        Δλ₁ / (λ₂''(i+1) − λ₂''(i)), used when mapping model spectra
        onto the observed grid (see LASP-Adam-GPU method, Step 3).
    leg_array_ : torch.Tensor
        Legendre polynomial array evaluated on the observed wavelength
        grid. Typically of shape (group_size, M, L), where L is the
        Legendre polynomial degree.
    goodPixels_final : torch.Tensor or bool, optional
        Final good-pixel mask (e.g. after outlier clipping). If
        False (default), no additional clipping is applied and all
        pixels are used. If a tensor mask is provided, it should have a
        broadcastable shape compatible with (group_size, M - 2).
    Jacobian : bool, optional
        If True (default), return per-pixel residuals
        (observed − model) for each spectrum, suitable for building
        a Jacobian matrix.
    Hessian : bool, optional
        If True, return summed-squared residuals per spectrum
        (i.e. a scalar loss per sample), typically used for
        Hessian-based approximations or scalar loss evaluation.

        Notes
        -----
        In normal usage, exactly one of Jacobian or Hessian should be
        True. If both are True, the Hessian branch will overwrite the
        Jacobian result.

    Returns
    -------
    loss : torch.Tensor
        - If Jacobian=True and Hessian=False:
              Tensor of shape (group_size, M-2) containing per-pixel
              flux residuals (observed − model), where the last two
              pixels are excluded for consistency with the Legendre
              setup.
        - If Hessian=True:
              Tensor of shape (group_size,) containing the
              summed-squared residuals for each spectrum.

    Notes
    -----
    - Residuals are always defined as (observed flux − model flux).
    - The "No Clean" strategy corresponds to goodPixels_final=False,
      meaning all pixels are used. A "Clean" strategy can be emulated by
      passing a 0/1 mask via goodPixels_final.

    Examples
    --------
    No minimal runnable example is provided here, because this function
    requires a fully configured spectral model to generate the model
    spectrum.

    """

    # ------------------------------------------------------------------
    # 4.1 Generate model spectra from polynomial coefficients
    # ------------------------------------------------------------------
    TGM_model_predict_spectra = uly_tgm_eval(
        spec_coef.to(device, dtype=dtype),
        best_params[:, :3],
    )

    # ------------------------------------------------------------------
    # 4.2 Rebin spectra onto the observed wavelength grid
    # ------------------------------------------------------------------
    TGM_model_predict_spectra_xrebin = (
        xrebin(
            borders_,
            TGM_model_predict_spectra,
            NewBorders_,
        )
        / flat
    ).to(device, dtype=dtype)

    # ------------------------------------------------------------------
    # 4.3 Convolve to lower resolution (match observed resolution)
    # ------------------------------------------------------------------
    low_resolution_spec = convol(
        TGM_model_predict_spectra_xrebin,
        best_params[:, 3].reshape(-1, 1),
        best_params[:, 4].reshape(-1, 1),
    ).to(device, dtype=dtype)

    # ------------------------------------------------------------------
    # 4.4 Fit and apply multiplicative Legendre polynomial
    # ------------------------------------------------------------------
    coefs_pol = mregress_batch_cholesky(
        leg_array_[:, :-2, :] * low_resolution_spec[:, :-2],
        specs[:, :-2],
    ).unsqueeze(1)
    poly1 = torch.matmul(coefs_pol, leg_array_.transpose(1, 2)).squeeze(1).unsqueeze(2)
    polynomial_multiply_TGM_model_predict_spectra = low_resolution_spec * poly1

    # ------------------------------------------------------------------
    # 4.5 Compute flux residuals (observed − model)
    # ------------------------------------------------------------------
    # 4.5.1 Jacobian mode: per-pixel residuals
    if Jacobian is True:
        if goodPixels_final is False:
            # 4.5.1.1 No Clean: do not iteratively clip outliers
            loss = (
                specs[:, :-2]
                - polynomial_multiply_TGM_model_predict_spectra.squeeze(-1)[:, :-2]
            )
        else:
            # 4.5.1.2 Clean: apply provided mask
            loss = (
                specs[:, :-2]
                - polynomial_multiply_TGM_model_predict_spectra.squeeze(-1)[:, :-2]
            ) * goodPixels_final
    # 4.5.2 Hessian mode: summed-squared residuals per spectrum
    if Hessian is True:
        if goodPixels_final is False:
            # 4.5.2.1 No Clean: do not iteratively clip outliers
            loss = torch.sum(
                (
                    specs[:, :-2]
                    - polynomial_multiply_TGM_model_predict_spectra.squeeze(-1)[:, :-2]
                )
                ** 2,
                dim=1,
            )
        else:
            # 4.5.2.2 Clean: apply provided mask
            loss = torch.sum(
                (
                    specs[:, :-2]
                    - polynomial_multiply_TGM_model_predict_spectra.squeeze(-1)[:, :-2]
                )
                * goodPixels_final,
                dim=1,
            )

    # ------------------------------------------------------------------
    # 4.6 Return flux residuals for Jacobian / Hessian analysis
    # ------------------------------------------------------------------
    return loss
