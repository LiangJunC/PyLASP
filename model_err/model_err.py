# -*- coding: utf-8 -*-
# @Time    : 2025/2/28 23:05
# @Author  : ljc
# @FileName: model_err.py
# @Software: PyCharm
# Update:  2025/11/26 20:34:29


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Error propagation via Jacobian and Hessian for LASP-Adam-GPU.

1.1 Purpose
-----------
Use central finite differences and standard error-propagation formulae
to estimate the uncertainties of the 5 internal parameters inferred
for each spectrum in LASP-Adam-GPU.

1.2 Functions
-------------
1) Jacobian_matrix: Compute the Jacobian matrix of per-pixel flux
   residuals with respect to the 5 fitted parameters for each spectrum.
2) Jacobian_parameter_err: Use the Jacobian and error-propagation
   formula to estimate parameter errors.
3) Hessian_matrix: Compute the Hessian matrix of the summed-squared
   residuals with respect to the 5 fitted parameters for each spectrum.
4) Hessian_parameter_err: Use the Hessian and error-propagation
   formula to estimate parameter errors.
5) parameter_err: User-facing wrapper that selects Jacobian-based or
   Hessian-based error estimation depending on the flags (Jacobian by
   default).

1.3 Explanation
---------------
- The Jacobian-based approach:
    * Uses per-pixel residuals between observed and model spectra.
    * Approximates the covariance as
      cov(param) = (JᵀJ / σ²)⁻¹,
      where J is the Jacobian and σ² is estimated from the residuals
      via an unbiased estimator.
- The Hessian-based approach:
    * Works on the scalar summed-squared residuals per spectrum.
    * Approximates parametric curvature using the second derivatives
      of the loss with respect to the parameters.
- In both cases, the internal parameters are later converted to
  astrophysical parameters using standard error-propagation
  (e.g. Manfred Drosg et al. 2009).

1.4 Notes
---------
- By default, Jacobian-based uncertainties are used (Jacobian=True,
  Hessian=False).
- Central finite differences are adopted for both Jacobian and Hessian
  computations; the default step size is h = 0.05.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
from config.config import default_set, set_all_seeds
import torch
import numpy as np

# 2.1 Set random seed
set_all_seeds()
# 2.2 Call GPU and specify data type
dtype, device = default_set()
# 2.3 Set default data type
torch.set_default_dtype(dtype)


# ======================================================================
# 3. Type definitions and constants for better code readability
# ======================================================================
ArrayLike = np.ndarray
TensorLike = torch.Tensor
C_light = 299792.458


# ======================================================================
# 4. Jacobian matrix of flux residuals
# ======================================================================
def Jacobian_matrix(
    loss_reduced,
    best_params: TensorLike,
    specs: TensorLike,
    spec_coef: TensorLike,
    borders_: TensorLike,
    NewBorders_: TensorLike,
    flat: TensorLike,
    leg_array_: TensorLike,
    group_size: int,
    n_params: int,
    h: float,
    goodPixels_final: bool,
    Jacobian: bool = True,
    Hessian: bool = False,
) -> TensorLike:
    r"""Compute the Jacobian matrix.

    Parameters
    ----------
    loss_reduced : callable
        Residual function in Jacobian mode, returning per-pixel
        residuals of shape (group_size, M-2).
    best_params : torch.Tensor
        Best-fit internal parameters from Adam, shape
        (group_size, n_params). These are internal variables and
        will later be propagated to astrophysical parameters.
    specs : torch.Tensor
        Observed spectra, shape (group_size, M).
    spec_coef : torch.Tensor
        polynomial coefficients used by the spectral emulator.
    borders_ : torch.Tensor
        Integration borders of the model spectra (input grid) for
        rebinning.
    NewBorders_ : torch.Tensor
        Integration borders corresponding to the target wavelength grid.
    flat : torch.Tensor
        Rebinning scaling factor.
    leg_array_ : torch.Tensor
        Legendre polynomial values.
    group_size : int
        Number of spectra in the current group.
    n_params : int
        Number of fitted parameters per spectrum (n_params = 5).
    h : float
        Step size for central finite differences.
    goodPixels_final : bool
        Final good-pixel mask (for outlier clipping). If False, no
        extra clipping is applied.
    Jacobian : bool
        Flag passed to loss_reduced; for Jacobian mode this should
        be True and Hessian=False so that per-pixel residuals are
        returned.
    Hessian : bool
        Flag passed to loss_reduced. For Jacobian computation this
        should normally be False.

    Returns
    -------
    J : torch.Tensor
        Jacobian matrix of shape
        (group_size, specs.shape[1]-2, n_params), containing
        ∂(residual flux) / ∂(parameter_j) for each spectrum.

    Notes
    -----
    - The central finite-difference formula is used:
          ∂f/∂p_j ≈ [f(p_j + h) − f(p_j − h)] / (2h)
      evaluated at the best-fit parameters.

    """

    # ------------------------------------------------------------------
    # 4.1 Initialize Jacobian with proper shape and dtype/device
    # ------------------------------------------------------------------
    with torch.no_grad():
        J = torch.zeros(
            (group_size, specs.shape[1] - 2, n_params),
            dtype=best_params.dtype,
            device=best_params.device,
        )

        # --------------------------------------------------------------
        # 4.2 Central finite differences for each parameter
        # --------------------------------------------------------------
        for j in range(n_params):
            params_plus = best_params.clone()
            params_minus = best_params.clone()

            params_plus[:, j] += h
            params_minus[:, j] -= h

            derivatives = (
                loss_reduced(
                    params_plus,
                    specs,
                    spec_coef,
                    borders_,
                    NewBorders_,
                    flat,
                    leg_array_,
                    goodPixels_final,
                    Jacobian,
                    Hessian,
                )
                - loss_reduced(
                    params_minus,
                    specs,
                    spec_coef,
                    borders_,
                    NewBorders_,
                    flat,
                    leg_array_,
                    goodPixels_final,
                    Jacobian,
                    Hessian,
                )
            ) / (2 * h)

            J[:, :, j : j + 1] = derivatives.unsqueeze(-1)

    # ------------------------------------------------------------------
    # 4.4 Return Jacobian matrix
    # ------------------------------------------------------------------
    return J


# ======================================================================
# 5. Parameter errors using the Jacobian
# ======================================================================
def Jacobian_parameter_err(
    loss_reduced,
    best_params: TensorLike,
    specs: TensorLike,
    spec_coef: TensorLike,
    borders_: TensorLike,
    NewBorders_: TensorLike,
    flat: TensorLike,
    leg_array_: TensorLike,
    group_size: int,
    n_params: int,
    dof: int,
    h: float,
    goodPixels_final: bool,
    Jacobian: bool,
    Hessian: bool,
    ln_wave_step: float = 0.0002302585092994046,
) -> np.ndarray:
    r"""Estimate parameter errors using the Jacobian and error propagation.

    Parameters
    ----------
    (All shared parameters follow Jacobian_matrix documentation.)
    Only additional parameters are listed below:

    dof : int
        Degrees of freedom used for unbiased variance σ² estimation.
    ln_wave_step: float
       ln(wave step) for rebinning.

    Returns
    -------
    parameter_std : numpy.ndarray
        Standard deviations of the 5 fitted parameters for each
        spectrum, shape (group_size, n_params). The first and
        fourth parameters are further converted to Teff and Rv
        uncertainties via error-propagation.

    Notes
    -----
    - The covariance is approximated as:
          cov ≈ (JᵀJ / σ²)⁻¹
      with σ² estimated via:
          σ² ≈ Σ(residual²) / dof (unbiased variance estimate).
    - Finally, error propagation is applied:
        * param 0: converted to Teff error
          σ_Teff = σ_p0 · 10^(p0 + 3.7617) · ln(10)
        * param 3: converted to radial-velocity error
          using c ≈ 299792.458 km/s and ln-step 0.0002302585.
    """

    # ------------------------------------------------------------------
    # 5.1 Compute Jacobian for all parameters and spectra
    # ------------------------------------------------------------------
    J = Jacobian_matrix(
        loss_reduced,
        best_params,
        specs,
        spec_coef,
        borders_,
        NewBorders_,
        flat,
        leg_array_,
        group_size,
        n_params,
        h,
        goodPixels_final,
        Jacobian,
        Hessian,
    )

    # ------------------------------------------------------------------
    # 5.2 Estimate variance of flux residuals via unbiased estimator
    # ------------------------------------------------------------------
    ssigma_sq = (
        torch.sum(
            loss_reduced(
                best_params,
                specs,
                spec_coef,
                borders_,
                NewBorders_,
                flat,
                leg_array_,
                goodPixels_final,
                Jacobian,
                Hessian,
            )
            ** 2,
            dim=1,
        )
        .unsqueeze(-1)
        .unsqueeze(-1)
        / dof
    )

    # ------------------------------------------------------------------
    # 5.3 Compute covariance matrix: cov ≈ (JᵀJ / σ²)⁻¹
    # ------------------------------------------------------------------
    covariance = torch.linalg.pinv(torch.bmm(J.transpose(1, 2), J) / ssigma_sq)

    # ------------------------------------------------------------------
    # 5.4 Extract diagonal and take sqrt to obtain parameter std
    # ------------------------------------------------------------------
    parameter_std = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2)).reshape(
        group_size, 5
    )

    # ------------------------------------------------------------------
    # 5.5 Error propagation: internal → astrophysical parameters
    # ------------------------------------------------------------------
    best_params = best_params.detach().cpu().numpy()
    parameter_std = parameter_std.detach().cpu().numpy()
    # 5.5.1 Teff errors (param 0)
    #       σ_Teff = σ_p0 · 10^(p0 + 3.7617) · ln(10)
    parameter_std[:, 0] = (
        parameter_std[:, 0] * np.power(10, best_params[:, 0] + 3.7617) * np.log(10)
    )
    # 5.5.2 Radial-velocity errors (param 3)
    #       c ≈ 299792.458 km/s; ln wave step ≈ 0.0002302585092994046
    parameter_std[:, 3] = parameter_std[:, 3] * (C_light * ln_wave_step)

    # ------------------------------------------------------------------
    # 5.6 Return propagated parameter uncertainties
    # ------------------------------------------------------------------
    return parameter_std


# ======================================================================
# 6. Hessian matrix of summed-squared residuals
# ======================================================================
def Hessian_matrix(
    loss_reduced,
    best_params: TensorLike,
    specs: TensorLike,
    spec_coef: TensorLike,
    borders_: TensorLike,
    NewBorders_: TensorLike,
    flat: TensorLike,
    leg_array_: TensorLike,
    group_size: int,
    n_params: int,
    h: float,
    goodPixels_final: bool,
    Jacobian: bool,
    Hessian: bool,
) -> TensorLike:
    r"""Compute Hessian of summed-squared residuals w.r.t. 5 parameters.

    Central finite differences are used to compute second derivatives of
    the summed-squared residuals with respect to the parameters.

    Parameters
    ----------
    (Shared parameters follow Jacobian_matrix. Only new logic is noted.)
    h : float
        Step size for central finite differences.
    Jacobian : bool
        Flag passed to loss_reduced; for Hessian mode this should
        be False and Hessian=True.
    Hessian : bool
        Flag passed to loss_reduced. For Hessian mode this should
        be True.

    Returns
    -------
    hessians : torch.Tensor
        Hessian matrices for all spectra, shape
        (group_size, n_params, n_params).

    """

    # ------------------------------------------------------------------
    # 6.1 Initialize Hessian tensor
    # ------------------------------------------------------------------
    with torch.no_grad():
        hessians = torch.zeros(
            (group_size, n_params, n_params),
            dtype=best_params.dtype,
            device=best_params.device,
        )

        # --------------------------------------------------------------
        # 6.2 Diagonal terms (second derivatives)
        # --------------------------------------------------------------
        for j in range(n_params):
            params_plus = best_params.clone()
            params_minus = best_params.clone()
            params_original = best_params.clone()

            params_plus[:, j] += h
            params_minus[:, j] -= h

            hessians[:, j, j] = (
                loss_reduced(
                    params_plus,
                    specs,
                    spec_coef,
                    borders_,
                    NewBorders_,
                    flat,
                    leg_array_,
                    goodPixels_final,
                    Jacobian,
                    Hessian,
                )
                - 2
                * loss_reduced(
                    params_original,
                    specs,
                    spec_coef,
                    borders_,
                    NewBorders_,
                    flat,
                    leg_array_,
                    goodPixels_final,
                    Jacobian,
                    Hessian,
                )
                + loss_reduced(
                    params_minus,
                    specs,
                    spec_coef,
                    borders_,
                    NewBorders_,
                    flat,
                    leg_array_,
                    goodPixels_final,
                    Jacobian,
                    Hessian,
                )
            ) / (h * h)

        # --------------------------------------------------------------
        # 6.3 Off-diagonal terms (mixed partial derivatives)
        # --------------------------------------------------------------
        for j in range(n_params):
            for k in range(j + 1, n_params):
                params_pp = best_params.clone()
                params_pm = best_params.clone()
                params_mp = best_params.clone()
                params_mm = best_params.clone()

                params_pp[:, j] += h
                params_pp[:, k] += h

                params_pm[:, j] += h
                params_pm[:, k] -= h

                params_mp[:, j] -= h
                params_mp[:, k] += h

                params_mm[:, j] -= h
                params_mm[:, k] -= h

                hessians[:, j, k] = (
                    loss_reduced(
                        params_pp,
                        specs,
                        spec_coef,
                        borders_,
                        NewBorders_,
                        flat,
                        leg_array_,
                        goodPixels_final,
                        Jacobian,
                        Hessian,
                    )
                    - loss_reduced(
                        params_pm,
                        specs,
                        spec_coef,
                        borders_,
                        NewBorders_,
                        flat,
                        leg_array_,
                        goodPixels_final,
                        Jacobian,
                        Hessian,
                    )
                    - loss_reduced(
                        params_mp,
                        specs,
                        spec_coef,
                        borders_,
                        NewBorders_,
                        flat,
                        leg_array_,
                        goodPixels_final,
                        Jacobian,
                        Hessian,
                    )
                    + loss_reduced(
                        params_mm,
                        specs,
                        spec_coef,
                        borders_,
                        NewBorders_,
                        flat,
                        leg_array_,
                        goodPixels_final,
                        Jacobian,
                        Hessian,
                    )
                ) / (4 * h * h)

                hessians[:, k, j] = hessians[:, j, k]

    # ------------------------------------------------------------------
    # 6.4 Return Hessian matrices
    # ------------------------------------------------------------------
    return hessians


# ======================================================================
# 7. Parameter errors using the Hessian
# ======================================================================
def Hessian_parameter_err(
    loss_reduced,
    best_params: TensorLike,
    specs: TensorLike,
    spec_coef: TensorLike,
    borders_: TensorLike,
    NewBorders_: TensorLike,
    flat: TensorLike,
    leg_array_: TensorLike,
    group_size: int,
    n_params: int,
    dof: int,
    h: float,
    goodPixels_final: bool,
    Jacobian: bool,
    Hessian: bool,
    ln_wave_step: float = 0.0002302585092994046,
) -> np.ndarray:
    r"""Estimate parameter errors using the Hessian and error propagation.

    Parameters
    ----------
    (All shared parameters are defined in Jacobian_matrix. Only
    function-specific parameters are listed below.)
    dof : int
        Degrees of freedom per spectrum.
    Jacobian : bool
        Flag passed to loss_reduced; for Hessian mode this should
        be False and Hessian=True.
    Hessian : bool
        Flag passed to loss_reduced. For Hessian mode this should
        be True.
    ln_wave_step: float
       ln(wave step) for rebinning.

    Returns
    -------
    parameter_std : numpy.ndarray
        Standard deviations of the 5 fitted parameters for each
        spectrum, shape (group_size, n_params), after error
        propagation to Teff and Rv.

    """

    # ------------------------------------------------------------------
    # 7.1 Compute Hessian matrices for all spectra
    # ------------------------------------------------------------------
    H = Hessian_matrix(
        loss_reduced,
        best_params,
        specs,
        spec_coef,
        borders_,
        NewBorders_,
        flat,
        leg_array_,
        group_size,
        n_params,
        h,
        goodPixels_final,
        Jacobian,
        Hessian,
    )

    # ------------------------------------------------------------------
    # 7.2 Estimate σ² from scalar loss (summed-squared residuals)
    # ------------------------------------------------------------------
    ssigma_sq = (
        loss_reduced(
            best_params,
            specs,
            spec_coef,
            borders_,
            NewBorders_,
            flat,
            leg_array_,
            goodPixels_final,
            Jacobian,
            Hessian,
        )
        .unsqueeze(-1)
        .unsqueeze(-1)
        / dof
    )

    # ------------------------------------------------------------------
    # 7.3 Approximate covariance from Hessian
    # ------------------------------------------------------------------
    covariance = 2.0 * torch.linalg.pinv(H) * ssigma_sq

    # ------------------------------------------------------------------
    # 7.4 Extract diagonal and take sqrt to obtain parameter std
    # ------------------------------------------------------------------
    parameter_std = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2)).reshape(
        group_size, 5
    )

    # ------------------------------------------------------------------
    # 7.5 Error propagation: internal → astrophysical parameters
    # ------------------------------------------------------------------
    best_params = best_params.detach().cpu().numpy()
    parameter_std = parameter_std.detach().cpu().numpy()
    # 7.5.1 Teff errors (param 0)
    parameter_std[:, 0] = (
        parameter_std[:, 0] * np.power(10, best_params[:, 0] + 3.7617) * np.log(10)
    )
    # 7.5.2 Radial-velocity errors (param 3)
    parameter_std[:, 3] = parameter_std[:, 3] * (C_light * ln_wave_step)

    # ------------------------------------------------------------------
    # 7.6 Return propagated parameter uncertainties
    # ------------------------------------------------------------------
    return parameter_std


# ======================================================================
# 8. User-facing wrapper for parameter errors
# ======================================================================
def parameter_err(
    loss_reduced,
    best_params: TensorLike,
    specs: TensorLike,
    spec_coef: TensorLike,
    borders_: TensorLike,
    NewBorders_: TensorLike,
    flat: TensorLike,
    leg_array_: TensorLike,
    group_size: int,
    n_params: int,
    dof: int,
    h: float,
    goodPixels_final: bool,
    Jacobian: bool = True,
    Hessian: bool = False,
    ln_wave_step: float = 0.0002302585092994046,
) -> np.ndarray:
    r"""Select Jacobian-based or Hessian-based parameter error estimation.

    Parameters
    ----------
    (All shared parameters are defined in Jacobian_matrix. Only
    function-specific parameters are listed below.)
    ln_wave_step: float
       ln(wave step) for rebinning.

    Returns
    -------
    parameter_std : numpy.ndarray
        Parameter uncertainties, shape (group_size, n_params),
        computed either by Jacobian or Hessian method.

    Notes
    -----
    - In normal usage, set exactly one of Jacobian or Hessian to True.
    - Jacobian-based estimation is the default choice.

    """

    # ------------------------------------------------------------------
    # 8.1 Select Jacobian or Hessian-based parameter error estimation
    # ------------------------------------------------------------------
    if Jacobian is True:
        return Jacobian_parameter_err(
            loss_reduced,
            best_params,
            specs,
            spec_coef,
            borders_,
            NewBorders_,
            flat,
            leg_array_,
            group_size,
            n_params,
            dof,
            h,
            goodPixels_final,
            Jacobian,
            Hessian,
            ln_wave_step,
        )
    if Hessian is True:
        return Hessian_parameter_err(
            loss_reduced,
            best_params,
            specs,
            spec_coef,
            borders_,
            NewBorders_,
            flat,
            leg_array_,
            group_size,
            n_params,
            dof,
            h,
            goodPixels_final,
            Jacobian,
            Hessian,
            ln_wave_step,
        )
