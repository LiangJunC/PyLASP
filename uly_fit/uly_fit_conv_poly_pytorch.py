# -*- coding: utf-8 -*-
# @Time    : 2025/11/22 22:01
# @Author  : ljc
# @FileName: uly_fit_conv_poly_pytorch.py
# @Software: Pycharm
# Update:  2025/11/26 21:05:12


# ============================================================================
# 1. Introduction
# ============================================================================
r"""Spectral fitting for observed spectra in LASP-Adam-GPU.

1.1 Purpose
-----------
Provide a high-throughput, GPU-accelerated implementation of PyLASP.
This module reads pre-packed spectra and auxiliary tensors from a
.pt file, fits the stellar parameters and radial velocity for each
spectrum via Adam optimization, and optionally performs iterative
Clean or No Clean strategies.

1.2 Functions
-------------
1) uly_fit_conv_poly:
   Main batch fitting driver. It loops over all spectra in the
   input .pt file, processes them in mini-batches, performs Adam
   optimization to minimize the flux residuals, estimates parameter
   errors, and appends the results to a CSV file.

1.3 Explanation
---------------
This module is designed for large-scale parameter inference on GPU
for stellar spectra.
Steps:
    1) Load spectra, initial parameters, wavelength grid, borders,
       NewBorders, Legendre polynomial array, and model coefficients
       from a pre-built .pt file.
    2) For each mini-batch of spectra:
         a) Normalize each observed spectrum by its median flux and
            rescale the initial parameters into a compact range
            suitable for Adam optimization.
         b) Generate the model spectra at the current parameters
            (Teff, log g, [Fe/H]).
         c) Rebin the model to the observational wavelength grid,
            divide by the flat field vector, and convolve with the
            LOSVD-like broadening parameters.
         d) Fit correction factors using batched linear regression
            and apply it to the broadened model spectra.
         e) Optionally invoke a Clean-like outlier rejection every
            fixed number of iterations to update the good-pixel mask.
         f) Minimize the flux residuals via Adam until the convergence
            criterion is reached or the maximum number of iterations
            is exceeded.
    3) After convergence, de-normalize the best-fit parameters back to
       physical units (Teff, log g, [Fe/H], mu and sigma).
    4) Compute parameter errors using finite-difference derivatives
       and the error propagation formalism provided by parameter_err
       and loss_reduced.
    5) Append the inferred parameters, errors, losses, timing
       information, and iteration counts to a CSV table for further
       analysis.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
from config.config import default_set, set_all_seeds
from uly_tgm_eval.uly_tgm_eval_pytorch import uly_tgm_eval
from WRS.xrebin_pytorch import xrebin
from resolution_reduction.convol_pytorch import convol
from legendre_polynomial.mregress_pytorch import mregress_batch_cholesky
from clean_outliers.clean_pytorch import clean_outliers
from model_err.model_err import parameter_err
from model_err.loss_reduced import loss_reduced
import torch
import numpy as np
import pandas as pd
import time, os

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
# 4. Main fitting function
# ======================================================================
def uly_fit_conv_poly(
    pt_file: str,
    group_size: int,
    max_inter: int,
    save_params_csv_file: str,
    clean: bool = True,
    quiet: bool = True,
) -> None:
    r"""Fit stellar parameters for all spectra stored in a .pt file.

    Parameters
    ----------
    pt_file : str
        Path to the .pt file containing pre-packed spectral data,
        initial parameters, wavelength grids, borders, NewBorders,
        flat, model coefficients, and Legendre polynomial arrays.
    group_size : int
        Number of spectra to process in each mini-batch. Larger values
        increase GPU memory usage but may improve throughput.
    max_inter : int
        Maximum number of Adam optimization iterations per batch.
    save_params_csv_file : str
        Path to the output CSV file where inferred parameters, errors,
        losses, and timing information will be appended.
    clean : bool, optional
        If True, enable iterative Clean-like outlier rejection during
        the fitting process. If False, use all pixels without masking.
    quiet : bool, optional
        Whether to use quiet mode, outputs statistics of each clipping
        layer when False.

    Returns
    -------
    None
        Results are written directly to the CSV file specified by
        save_params_csv_file.

    Notes
    -----
    - The function processes spectra in mini-batches to manage
      GPU memory.
    - Initial parameters are normalized for optimization.
    - Clean mode reduces outlier influence but increases computation
      time.
    - Convergence is determined by consecutive iterations with loss
      changes below tolerance (1e-5) or when loss fails to decrease
      for 50 iterations.
    - Parameter errors are computed via finite-difference Jacobian.

    Examples
    --------
    No minimal runnable example is provided here, because this function
    requires a fully configured spectral model to generate the model
    spectrum.

    """

    # ------------------------------------------------------------------
    # 4.1 Load packed .pt file containing spectra and auxiliary tensors
    # ------------------------------------------------------------------
    pt_data = torch.load(pt_file)
    all_specta_data = pt_data["spectra_data"].to(device, dtype=dtype)
    sample_number = all_specta_data.shape[0]
    (
        total_ini,
        all_wavelength,
        all_borders,
        all_NewBorders,
        flat,
        spec_coef,
        leg_array,
        all_ini_params,
        all_ID,
    ) = (
        int(np.ceil(sample_number / group_size)),
        pt_data["wavelength_data"].to(device, dtype=dtype),
        pt_data["borders"].to(device, dtype=dtype),
        pt_data["NewBorders"].to(device, dtype=dtype),
        pt_data["flat"].to(device, dtype=dtype),
        pt_data["spec_coef"].to(device, dtype=dtype),
        pt_data["leg_array"].to(device, dtype=dtype),
        pt_data["ini_params"].to(device, dtype=dtype),
        pt_data["ID"],
    )

    # ------------------------------------------------------------------
    # 4.2 Loop over all spectrum batches
    # ------------------------------------------------------------------
    for ii in range(total_ini):

        # 4.2.1 Extract data for the ii-th batch
        start, end = ii * group_size, min((ii + 1) * group_size, sample_number)
        batch_size = end - start
        ID, spectra_data, ini_params = (
            all_ID[start:end].reshape(-1),
            all_specta_data[start:end],
            all_ini_params[start:end].clone(),
        )
        borders, NewBorders, leg_array_ = (
            all_borders.repeat(batch_size, 1),
            all_NewBorders.repeat(batch_size, 1),
            leg_array.unsqueeze(0).repeat(batch_size, 1, 1),
        )

        # 4.2.1.1 Normalize flux by median and rescale initial parameters
        spectra_median = torch.median(spectra_data, dim=1, keepdim=True).values
        spectra_data_ = spectra_data / spectra_median

        # 4.2.1.2 Transform initial parameters to log/shifted space
        ini_params[:, 0], ini_params[:, 1], ini_params[:, 2] = (
            torch.log10(ini_params[:, 0]) - 3.7617,
            ini_params[:, 1] - 4.44,
            ini_params[:, 2].clip(min=-2.5),
        )

        # 4.2.1.3 Normalize parameters to a fixed range to improve
        #         optimization stability (empirically chosen scaling).
        #         Guess: the plausible bounds for Teff, log g, [Fe/H],
        #         mu, and sigma are [3000, 30000], [0, 6], [-2.5, 1],
        #         [-15, 15], and [0.1, 5], respectively.
        #         x_norm = 2*(x - 0.5*(x_max + x_min)) / (x_max - x_min)
        # log10(Teff) : 0.215 = (log10(30000) + log10(3000)) / 2 - 3.7617,
        #                   1 = log10(30000) - log10(3000)
        # log g - 4.44 : -1.44 = (0 + 6) / 2 - 4.44,
        #                    6 = 6 - 0
        # [Fe/H] : -0.75 = (-2.5 + 1) / 2,
        #            3.5 = 1 - (-2.5)
        # mu : 0 = (15 - 15) / 2,
        #     30 = 15 - (-15)
        # sigma : 2.55 = (5 + 0.1) / 2,
        #          4.9 = 5 - (0.1)
        time1 = time.time()
        fit_params_tensor_batch = ini_params.clone()
        (
            fit_params_tensor_batch[:, 0],
            fit_params_tensor_batch[:, 1],
            fit_params_tensor_batch[:, 2],
            fit_params_tensor_batch[:, 3],
            fit_params_tensor_batch[:, 4],
        ) = (
            2 * (fit_params_tensor_batch[:, 0] - 0.215) / 1,
            2 * (fit_params_tensor_batch[:, 1] - (-1.44)) / 6,
            2 * (fit_params_tensor_batch[:, 2] - (-0.75)) / 3.5,
            2 * (fit_params_tensor_batch[:, 3] - 0) / 30,
            2 * (fit_params_tensor_batch[:, 4] - 2.55) / 4.9,
        )

        # 4.2.2 Initialize Adam optimizer, convergence criteria and goodPixels
        (
            fit_params_tensor_batch,
            optimizer,
            prev_loss,
            tolerance,
            consecutive_limit,
            consecutive_count,
            loss_min,
            ini_loss_min,
            best_params,
            goodPixels_final,
            goodPixels0,
        ) = (
            fit_params_tensor_batch.requires_grad_(True),
            torch.optim.Adam([fit_params_tensor_batch], lr=0.1, betas=(0.9, 0.999)),
            float("inf"),
            1e-5,
            50,
            0,
            999999999999999,
            0,
            None,
            torch.ones_like(spectra_data_[:, :-2], dtype=torch.float32),
            torch.ones_like(spectra_data_[:, :-2], dtype=torch.float32),
        )

        # 4.2.3 Adam optimization loop to minimize flux residuals
        for iteration in range(max_inter):
            # 4.2.3.1 Zero gradients from previous iteration
            optimizer.zero_grad()
            # 4.2.3.2 Clamp normalized parameters to [-1, 1] range
            with torch.no_grad():
                fit_params_tensor_batch[:, 0].clamp_(-1.0, 1.0)
                fit_params_tensor_batch[:, 1].clamp_(-1.0, 1.0)
                fit_params_tensor_batch[:, 2].clamp_(-1.0, 1.0)
                fit_params_tensor_batch[:, 3].clamp_(-1.0, 1.0)
                fit_params_tensor_batch[:, 4].clamp_(-1.0, 1.0)
            # 4.2.3.3 De-normalize parameters to physical units
            denormalized_params = torch.stack(
                [
                    0.5 * fit_params_tensor_batch[:, 0] + 0.215,
                    3 * fit_params_tensor_batch[:, 1] + (-1.44),
                    1.75 * fit_params_tensor_batch[:, 2] + (-0.75),
                    15 * fit_params_tensor_batch[:, 3] + 0,
                    2.45 * fit_params_tensor_batch[:, 4] + 2.55,
                ],
                dim=1,
            )

            # 4.2.3.4 Generate model spectra
            TGM_model_predict_spectra = uly_tgm_eval(
                spec_coef.to(device, dtype=dtype), denormalized_params[:, :3]
            )
            # 4.2.3.5 Rebin spectra to observational wavelength grid
            TGM_model_predict_spectra_xrebin = (
                xrebin(borders, TGM_model_predict_spectra, NewBorders) / flat
            ).to(device, dtype=dtype)
            # 4.2.3.6 Degrade spectral resolution via convolution
            low_resolution_spec = convol(
                TGM_model_predict_spectra_xrebin,
                denormalized_params[:, 3].reshape(-1, 1),
                denormalized_params[:, 4].reshape(-1, 1),
            ).to(device, dtype=dtype)
            # 4.2.3.7 Fit multiplicative Legendre
            coefs_pol = mregress_batch_cholesky(
                leg_array_[:, :-2, :] * low_resolution_spec[:, :-2],
                spectra_data_[:, :-2],
            ).unsqueeze(1)
            poly1 = (
                torch.matmul(coefs_pol, leg_array_.transpose(1, 2))
                .squeeze(1)
                .unsqueeze(2)
            )
            polynomial_multiply_TGM_model_predict_spectra = low_resolution_spec * poly1

            # 4.2.3.8 Clean mode: iterative outlier rejection
            if clean:
                clean_, clean_num_max = 30, 11
                if (
                    ((iteration % clean_) == 0)
                    & (iteration > 1)
                    & (iteration < (clean_num_max * clean_ + 1))
                ):
                    current_mask = goodPixels_final.clone()
                    goodPixels_final = clean_outliers(
                        iteration,
                        polynomial_multiply_TGM_model_predict_spectra[:, :-2],
                        spectra_data_[:, :-2]
                        - polynomial_multiply_TGM_model_predict_spectra.squeeze(-1)[
                            :, :-2
                        ],
                        spectra_data_[:, :-2].shape[1],
                        goodPixels0=goodPixels0,
                        goodPixels=current_mask,
                        noise=1,
                        quiet=quiet,
                    )

            # 4.2.3.9 Compute loss
            loss = torch.sqrt(
                torch.mean(
                    (
                        (
                            spectra_data_[:, :-2]
                            - polynomial_multiply_TGM_model_predict_spectra.squeeze(-1)[
                                :, :-2
                            ]
                        )
                        * goodPixels_final
                    )
                    ** 2
                )
            )

            # 4.2.3.10 Backpropagation and parameter update
            loss.backward()
            optimizer.step()

            # 4.2.3.11 Check convergence criteria
            loss_diff = abs(prev_loss - loss.item())
            if loss_diff < tolerance:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count >= consecutive_limit:
                best_params, best_loss = fit_params_tensor_batch, loss
                alone_loss = torch.sqrt(
                    torch.mean(
                        (
                            (
                                spectra_data_[:, :-2]
                                - polynomial_multiply_TGM_model_predict_spectra.squeeze(
                                    -1
                                )[:, :-2]
                            )
                            * goodPixels_final
                        )
                        ** 2,
                        dim=1,
                    )
                )
                break
            prev_loss = loss.item()
            if loss < loss_min:
                loss_min, ini_loss_min, best_params, best_loss = (
                    loss,
                    0,
                    fit_params_tensor_batch,
                    loss,
                )
                alone_loss = torch.sqrt(
                    torch.mean(
                        (
                            (
                                spectra_data_[:, :-2]
                                - polynomial_multiply_TGM_model_predict_spectra.squeeze(
                                    -1
                                )[:, :-2]
                            )
                            * goodPixels_final
                        )
                        ** 2,
                        dim=1,
                    )
                )
            else:
                ini_loss_min += 1
            if ini_loss_min > consecutive_limit:
                break

        # 4.2.4 Record inference time for current batch
        time2 = time.time()
        use_time = time2 - time1
        print(
            f"Processing {os.path.relpath(pt_file)}, total {sample_number} samples! "
            f"Inferring samples {start}-{end} completed in: "
            f"{round(use_time, 2)} seconds!"
        )

        # 4.2.5 De-normalize best-fit parameters back to physical units
        with torch.no_grad():
            (
                best_params[:, 0],
                best_params[:, 1],
                best_params[:, 2],
                best_params[:, 3],
                best_params[:, 4],
            ) = (
                0.5 * best_params[:, 0] + 0.215,
                3 * best_params[:, 1] + (-1.44),
                1.75 * best_params[:, 2] + (-0.75),
                15 * best_params[:, 3] + 0,
                2.45 * best_params[:, 4] + 2.55,
            )

        # 4.2.6 Compute parameter errors via error propagation
        parameter_std = parameter_err(
            loss_reduced,
            best_params.detach(),
            spectra_data_.detach(),
            spec_coef.detach(),
            borders.detach(),
            NewBorders.detach(),
            flat.detach(),
            leg_array_.detach(),
            batch_size,
            n_params=best_params.detach().shape[1],
            dof=spectra_data_[:, :-2].detach().shape[1] - best_params.detach().shape[1],
            h=0.05,
            goodPixels_final=goodPixels_final.detach(),
            Jacobian=True,
            Hessian=False,
        )

        # 4.2.7 Save inferred parameters and diagnostics to CSV
        best_params, loss, alone_loss = (
            best_params.detach().cpu().numpy(),
            best_loss.detach().cpu().numpy(),
            alone_loss.detach().cpu().numpy(),
        )
        new_parameters = pd.DataFrame(
            data={
                "ID": ID,
                "pred_Rv": 299792.458
                * (np.exp(0.0002302585092994046 * best_params[:, 3]) - 1),
                "pred_Teff": np.power(10, best_params[:, 0] + 3.7617),
                "pred_logg": best_params[:, 1] + 4.44,
                "pred_FeH": best_params[:, 2],
                "pred_Rv_err": parameter_std[:, 3],
                "pred_Teff_err": parameter_std[:, 0],
                "pred_logg_err": parameter_std[:, 1],
                "pred_FeH_err": parameter_std[:, 2],
                "use_time": [use_time] * batch_size,
                "total_loss": loss,
                "alone_loss": alone_loss,
                "iter": [iteration] * batch_size,
            },
            columns=[
                "ID",
                "pred_Rv",
                "pred_Teff",
                "pred_logg",
                "pred_FeH",
                "pred_Rv_err",
                "pred_Teff_err",
                "pred_logg_err",
                "pred_FeH_err",
                "use_time",
                "total_loss",
                "alone_loss",
                "iter",
            ],
        )
        new_parameters.to_csv(save_params_csv_file, mode="a", header=0, index=False)
