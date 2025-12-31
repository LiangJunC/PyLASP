# -*- coding: utf-8 -*-
# @Time    : 2025/1/4 23:55
# @Author  : ljc
# @FileName: mregress_pytorch.py
# @Software: PyCharm
# Update:  2025/11/26 20:18:44


# ======================================================================
# 1. Introduction
# ======================================================================
r"""PyTorch implementation of multiple LR for LASP-Adam-GPU.

1.1 Purpose
-----------
Implement batch multiple linear regression using PyTorch to calculate
regression coefficients for Legendre polynomial coefficients. Provides
various matrix decomposition methods to accommodate different
computational requirements and numerical stability needs, applicable
to LASP-Adam-GPU.

1.2 Functions
-------------
1) mregress_batch_cholesky: Using Cholesky decomposition (recommended by
   default).
2) mregress_batch_lu: Using LU decomposition, supports batch processing
   for large-scale data.
3) mregress_batch_svd: Using SVD decomposition, highest numerical
   stability.
4) mregress_batch_qr: Using QR decomposition, balances efficiency and
   stability.
5) mregress_batch_inv: Using direct matrix inversion.

1.3 Explanation
---------------
All functions perform weighted least squares multiple linear regression
to solve coefficient matrix A in linear system X·A = y.
1.3.1 Mathematical Principle
For linear system y = X·A, the explicit solution of weighted least
squares is:
    A = (X^T·W·X)^(-1)·X^T·W·y
where:
    X: Independent variable matrix (group_size × Npoints × Nterms)
    W: Weight diagonal matrix with diagonal elements 1/(measure_errors^2)
    y: Dependent variable vector (group_size × Npoints)
    A: Regression coefficient vector (group_size × Nterms)

1.3.2 Solution Steps
Steps:
    1) Data validation: Check types and dimensions of input data X, y,
       measure_errors.
    2) Calculate weight matrix W.
    3) Weighted processing: Calculate X^T·W·X and X^T·W·y.
    4) Matrix inversion: Calculate (X^T·W·X)^(-1).
    5) Calculate regression coefficient A: (X^T·W·X)^(-1)·X^T·W·y.

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
# 4. Batch multiple linear regression using Cholesky decomposition
# ======================================================================
def mregress_batch_cholesky(
    x: TensorLike, y: TensorLike, measure_errors: TensorLike | None = None
) -> TensorLike:
    r"""Perform weighted least squares regression using Cholesky.

    Parameters
    ----------
    x : torch.Tensor
        shape (group_size, npts, nterm)
        Independent variable data matrix, where group_size is the batch
        size, npts is the number of sample points, nterm is the number
        of coefficients (independent variables) to solve.
    y : torch.Tensor
        shape (group_size, npts)
        Dependent variable data matrix, must contain group_size samples,
        each with npts elements.
    measure_errors : torch.Tensor, optional
        Vector containing standard measurement errors for each point
        y[i]. In the current LASP-Adam-GPU pipeline this argument is
        typically passed as None (or 1), but the interface is kept for
        generality.

    Returns
    -------
    a : torch.Tensor
        shape (group_size, nterm)
        Regression coefficient matrix solved from equation X·A = y.

    Notes
    -----
    - Cholesky decomposition version offers fast computation and
      improved numerical stability through automatic diagonal element
      adjustment.
    - Automatically increases diagonal elements (eps) when decomposition
      fails to ensure numerical stability.

    Examples
    --------
    >>> X = torch.randn(10, 100, 5, dtype=dtype, device=device)
    >>> y = torch.randn(10, 100, dtype=dtype, device=device)
    >>> a = mregress_batch_cholesky(X, y)
    >>> print(a.shape)
    torch.Size([10, 5])

    """

    # ------------------------------------------------------------------
    # 4.1 Input validation
    # ------------------------------------------------------------------
    # 4.1.1 Check input dimensions and get shape information
    batch_size, npts, nterm = x.shape
    if y.shape != (batch_size, npts):
        raise ValueError("X and Y have incompatible dimensions.")

    # 4.1.2 Calculate degrees of freedom
    nfree = npts - 1.0

    # ------------------------------------------------------------------
    # 4.2 Weight calculation and normalization
    # ------------------------------------------------------------------
    if measure_errors is None:
        weights = torch.ones_like(y, dtype=dtype, device=device)
    else:
        if measure_errors.shape != (batch_size, npts):
            raise ValueError("Measure errors have incompatible dimensions.")
        else:
            weights = 1 / (measure_errors**2)
    sw = weights.sum(dim=1, keepdim=True) / npts
    weights = weights / sw

    # ------------------------------------------------------------------
    # 4.3  Weighted processing: calculate X^T·W·X and X^T·W·y
    # ------------------------------------------------------------------
    wx = x * weights.unsqueeze(-1)
    # 4.3.1 ar1 and br1 are X^T·W·X and X^T·W·y respectively
    ar1, br1 = (
        torch.bmm(wx.transpose(1, 2), x),
        torch.bmm(wx.transpose(1, 2), y.unsqueeze(2)).squeeze(-1),
    )
    # 4.3.2 To improve inversion stability
    sigmax, sigmay = (
        torch.sqrt((x * wx).sum(dim=1) / nfree),
        torch.sqrt((weights * y.pow(2)).sum(dim=1) / nfree),
    )
    # 4.3.3 ar2 and br2 are normalized X^T·W·X and X^T·W·y respectively
    ar2, br2 = (
        ar1 / (sigmax.unsqueeze(-1) * sigmax.unsqueeze(1) * nfree),
        br1 / (sigmax * sigmay.unsqueeze(1) * nfree),
    )

    # ------------------------------------------------------------------
    # 4.4 Cholesky decomposition and solution: solve A (ar2·A = br2)
    # ------------------------------------------------------------------
    try:
        eps = torch.finfo(ar2.dtype).eps
        ar3 = (
            ar2
            + torch.eye(nterm, device=ar2.device, dtype=ar2.dtype).unsqueeze(0) * eps
        )
        # 4.4.1 Adding machine epsilon to diagonal
        L = torch.linalg.cholesky(ar3)
    except:
        try:
            eps = 1e-6
            ar3 = (
                ar2
                + torch.eye(nterm, device=ar2.device, dtype=ar2.dtype).unsqueeze(0)
                * eps
            )
            # 4.4.2 Adding eps=1e-6 to diagonal
            L = torch.linalg.cholesky(ar3)
        except:
            try:
                eps = 1e-5
                ar3 = (
                    ar2
                    + torch.eye(nterm, device=ar2.device, dtype=ar2.dtype).unsqueeze(0)
                    * eps
                )
                # 4.4.3 Adding eps=1e-5 to diagonal
                L = torch.linalg.cholesky(ar3)
            except:
                eps = 1e-4
                ar3 = (
                    ar2
                    + torch.eye(nterm, device=ar2.device, dtype=ar2.dtype).unsqueeze(0)
                    * eps
                )
                # 4.4.4 Adding eps=1e-4 to diagonal
                L = torch.linalg.cholesky(ar3)
    y1 = torch.linalg.solve_triangular(L, br2.unsqueeze(2), upper=False)
    a = torch.linalg.solve_triangular(L.transpose(-2, -1), y1, upper=True)
    # 4.4.5 Denormalize to calculate final coefficient A
    a = a.squeeze(-1) * (sigmay.unsqueeze(1) / sigmax)

    # ------------------------------------------------------------------
    # 4.5 Return regression coefficients
    # ------------------------------------------------------------------
    return a


# ======================================================================
# 5. Batch multiple linear regression using LU decomposition
# ======================================================================
def mregress_batch_lu(
    x: TensorLike,
    y: TensorLike,
    measure_errors: TensorLike | None = None,
    group_size: int = 100,
) -> TensorLike:
    r"""Perform weighted least squares regression using LU.

    Parameters
    ----------
    x : torch.Tensor
        shape (group_size, npts, nterm)
        Independent variable data matrix, where group_size is the batch
        size, npts is the number of sample points, nterm is the number
        of coefficients (independent variables) to solve.
    y : torch.Tensor
        shape (group_size, npts)
        Dependent variable data matrix, must contain group_size samples,
        each with npts elements.
    measure_errors : torch.Tensor, optional
        Vector containing standard measurement errors for each point
        y[i]. In the current LASP-Adam-GPU pipeline this argument is
        typically passed as None (or 1), but the interface is kept for
        generality.
    group_size : int, default=100
        Batch size for batch processing, used to control memory usage.
        Smaller batch size reduces memory requirements but may increase
        computation time.

    Returns
    -------
    results : torch.Tensor
        shape (group_size, nterm)
        Regression coefficient matrix solved from equation X·A = y.

    Notes
    -----
    - LU decomposition version supports batch processing, suitable for
      large-scale datasets, performs well under memory constraints.
    - Uses torch.linalg.solve which internally employs LU decomposition.

    Examples
    --------
    >>> X = torch.randn(1000, 100, 5, dtype=dtype, device=device)
    >>> y = torch.randn(1000, 100, dtype=dtype, device=device)
    >>> a = mregress_batch_lu(X, y, group_size=100)
    >>> print(a.shape)
    torch.Size([1000, 5])

    """

    # ------------------------------------------------------------------
    # 5.1 Batch calculation preparation
    # ------------------------------------------------------------------
    # 5.1.1 Get total sample size and feature dimensions
    total_size, npts, nterm = x.shape
    num_batches = (total_size + group_size - 1) // group_size
    results = []

    # ------------------------------------------------------------------
    # 5.2 Batch processing loop
    # ------------------------------------------------------------------
    # 5.2.1 Iterate over each batch
    for i in range(num_batches):
        # --------------------------------------------------------------
        # 5.2.1.1 Get current batch data
        # --------------------------------------------------------------
        start_idx = i * group_size
        end_idx = min(start_idx + group_size, total_size)
        batch_x, batch_y = x[start_idx:end_idx], y[start_idx:end_idx]

        # --------------------------------------------------------------
        # 5.2.1.2 Input validation
        # --------------------------------------------------------------
        # 5.2.1.2.1 Check input dimensions and get shape information
        current_batch_size = end_idx - start_idx
        if batch_y.shape != (current_batch_size, npts):
            raise ValueError("X and Y have incompatible dimensions.")

        # 5.2.1.2.2 Calculate degrees of freedom
        nfree = npts - 1.0

        # --------------------------------------------------------------
        # 5.2.1.3 Weight calculation and normalization
        # --------------------------------------------------------------
        if measure_errors is None:
            weights = torch.ones((current_batch_size, npts), dtype=dtype, device=device)
        else:
            if measure_errors.shape != (total_size, npts):
                raise ValueError("Measure errors have incompatible dimensions.")
            else:
                weights = 1 / (measure_errors[start_idx:end_idx] ** 2)
        sw = weights.sum(dim=1, keepdim=True) / npts
        weights = weights / sw

        # --------------------------------------------------------------
        # 5.2.1.4 Weighted processing: calculate X^T·W·X and X^T·W·y
        # --------------------------------------------------------------
        wx = batch_x * weights.unsqueeze(-1)
        # 5.2.1.4.1 ar1 and br1 are X^T·W·X and X^T·W·y respectively
        ar1, br1 = (
            torch.bmm(wx.transpose(1, 2), batch_x),
            torch.bmm(wx.transpose(1, 2), batch_y.unsqueeze(2)).squeeze(-1),
        )
        # 5.2.1.4.2 To improve inversion stability
        sigmax, sigmay = (
            torch.sqrt((batch_x * wx).sum(dim=1) / nfree),
            torch.sqrt((weights * batch_y.pow(2)).sum(dim=1) / nfree),
        )
        # 5.2.1.4.3 ar2 and br2 are normalized X^T·W·X and X^T·W·y
        ar2, br2 = (
            ar1 / (sigmax.unsqueeze(-1) * sigmax.unsqueeze(1) * nfree),
            br1 / (sigmax * sigmay.unsqueeze(1) * nfree),
        )

        # --------------------------------------------------------------
        # 5.2.1.5 LU decomposition and solution: solve A (ar2·A = br2)
        # --------------------------------------------------------------
        # 5.2.1.5.1 Directly use torch.linalg.solve, internally uses LU
        a = torch.linalg.solve(ar2, br2.unsqueeze(2))
        # 5.2.1.5.2 Denormalize to calculate final coefficient A
        a = a.squeeze(-1) * (sigmay.unsqueeze(1) / sigmax)
        results.append(a)

    # ------------------------------------------------------------------
    # 5.3 Return regression coefficients
    # ------------------------------------------------------------------
    return torch.cat(results, dim=0)


# ======================================================================
# 6. Batch multiple linear regression using SVD decomposition
# ======================================================================
def mregress_batch_svd(
    x: TensorLike, y: TensorLike, measure_errors: TensorLike | None = None
) -> TensorLike:
    r"""Perform weighted least squares regression using SVD.

    Parameters
    ----------
    x : torch.Tensor
        shape (group_size, npts, nterm)
        Independent variable data matrix, where group_size is the batch
        size, npts is the number of sample points, nterm is the number
        of coefficients (independent variables) to solve.
    y : torch.Tensor
        shape (group_size, npts)
        Dependent variable data matrix, must contain group_size samples,
        each with npts elements.
    measure_errors : torch.Tensor, optional
        Vector containing standard measurement errors for each point
        y[i]. In the current LASP-Adam-GPU pipeline this argument is
        typically passed as None (or 1), but the interface is kept for
        generality.

    Returns
    -------
    a : torch.Tensor
        shape (group_size, nterm)
        Regression coefficient matrix solved from equation X·A = y.

    Notes
    -----
    - SVD decomposition version has the highest numerical stability,
      can handle non-full-rank matrices, but has relatively slower
      computation speed.

    Examples
    --------
    >>> X = torch.randn(10, 100, 5, dtype=dtype, device=device)
    >>> y = torch.randn(10, 100, dtype=dtype, device=device)
    >>> a = mregress_batch_svd(X, y)
    >>> print(a.shape)
    torch.Size([10, 5])

    """

    # ------------------------------------------------------------------
    # 6.1 Input validation
    # ------------------------------------------------------------------
    # 6.1.1 Check input dimensions and get shape information
    batch_size, npts, nterm = x.shape
    if y.shape != (batch_size, npts):
        raise ValueError("X and Y have incompatible dimensions.")

    # 6.1.2 Calculate degrees of freedom
    nfree = npts - 1.0

    # ------------------------------------------------------------------
    # 6.2 Weight calculation and normalization
    # ------------------------------------------------------------------
    if measure_errors is None:
        weights = torch.ones_like(y, dtype=dtype, device=device)
    else:
        if measure_errors.shape != (batch_size, npts):
            raise ValueError("Measure errors have incompatible dimensions.")
        else:
            weights = 1 / (measure_errors**2)
    sw = weights.sum(dim=1, keepdim=True) / npts
    weights = weights / sw

    # ------------------------------------------------------------------
    # 6.3 Weighted processing: calculate X^T·W·X and X^T·W·y
    # ------------------------------------------------------------------
    wx = x * weights.unsqueeze(-1)
    # 6.3.1 ar1 and br1 are X^T·W·X and X^T·W·y respectively
    ar1, br1 = (
        torch.bmm(wx.transpose(1, 2), x),
        torch.bmm(wx.transpose(1, 2), y.unsqueeze(2)).squeeze(-1),
    )
    # 6.3.2 To improve inversion stability
    sigmax, sigmay = (
        torch.sqrt((x * wx).sum(dim=1) / nfree),
        torch.sqrt((weights * y.pow(2)).sum(dim=1) / nfree),
    )
    # 6.3.3 ar2 and br2 are normalized X^T·W·X and X^T·W·y respectively
    ar2, br2 = (
        ar1 / (sigmax.unsqueeze(-1) * sigmax.unsqueeze(1) * nfree),
        br1 / (sigmax * sigmay.unsqueeze(1) * nfree),
    )

    # ------------------------------------------------------------------
    # 6.4 SVD decomposition and solution: solve A (ar2·A = br2)
    # ------------------------------------------------------------------
    # 6.4.1 Solve using SVD decomposition
    U, S, Vh = torch.linalg.svd(ar2)
    S_inv = 1.0 / (S + 1e-10)
    a = torch.bmm(
        U * S_inv.unsqueeze(1), torch.bmm(U.transpose(1, 2), br2.unsqueeze(2))
    )
    # 6.4.2 Denormalize to calculate final coefficient A
    a = a.squeeze(-1) * (sigmay.unsqueeze(1) / sigmax)

    # ------------------------------------------------------------------
    # 6.5 Return regression coefficients
    # ------------------------------------------------------------------
    return a


# ======================================================================
# 7. Batch multiple linear regression using QR decomposition
# ======================================================================
def mregress_batch_qr(
    x: TensorLike, y: TensorLike, measure_errors: TensorLike | None = None
) -> TensorLike:
    r"""Perform weighted least squares regression using QR.

    Parameters
    ----------
    x : torch.Tensor
        shape (group_size, npts, nterm)
        Independent variable data matrix, where group_size is the batch
        size, npts is the number of sample points, nterm is the number
        of coefficients (independent variables) to solve.
    y : torch.Tensor
        shape (group_size, npts)
        Dependent variable data matrix, must contain group_size samples,
        each with npts elements.
    measure_errors : torch.Tensor, optional
        Vector containing standard measurement errors for each point
        y[i]. In the current LASP-Adam-GPU pipeline this argument is
        typically passed as None (or 1), but the interface is kept for
        generality.

    Returns
    -------
    a : torch.Tensor
        shape (group_size, nterm)
        Regression coefficient matrix solved from equation X·A = y.

    Notes
    -----
    - QR decomposition version avoids explicit matrix inversion,
      providing a good balance between efficiency and numerical
      stability.

    Examples
    --------
    >>> X = torch.randn(10, 100, 5, dtype=dtype, device=device)
    >>> y = torch.randn(10, 100, dtype=dtype, device=device)
    >>> a = mregress_batch_qr(X, y)
    >>> print(a.shape)
    torch.Size([10, 5])

    """

    # ------------------------------------------------------------------
    # 7.1 Input validation
    # ------------------------------------------------------------------
    # 7.1.1 Check input dimensions and get shape information
    batch_size, npts, nterm = x.shape
    if y.shape != (batch_size, npts):
        raise ValueError("X and Y have incompatible dimensions.")

    # 7.1.2 Calculate degrees of freedom
    nfree = npts - 1.0

    # ------------------------------------------------------------------
    # 7.2 Weight calculation and normalization
    # ------------------------------------------------------------------
    if measure_errors is None:
        weights = torch.ones_like(y, dtype=dtype, device=device)
    else:
        if measure_errors.shape != (batch_size, npts):
            raise ValueError("Measure errors have incompatible dimensions.")
        else:
            weights = 1 / (measure_errors**2)
    sw = weights.sum(dim=1, keepdim=True) / npts
    weights = weights / sw

    # ------------------------------------------------------------------
    # 7.3  Weighted processing: calculate X^T·W·X and X^T·W·y
    # ------------------------------------------------------------------
    wx = x * weights.unsqueeze(-1)
    # 7.3.1 ar1 and br1 are X^T·W·X and X^T·W·y respectively
    ar1, br1 = (
        torch.bmm(wx.transpose(1, 2), x),
        torch.bmm(wx.transpose(1, 2), y.unsqueeze(2)).squeeze(-1),
    )
    # 7.3.2 To improve inversion stability
    sigmax, sigmay = (
        torch.sqrt((x * wx).sum(dim=1) / nfree),
        torch.sqrt((weights * y.pow(2)).sum(dim=1) / nfree),
    )
    # 7.3.3 ar2 and br2 are normalized X^T·W·X and X^T·W·y respectively
    ar2, br2 = (
        ar1 / (sigmax.unsqueeze(-1) * sigmax.unsqueeze(1) * nfree),
        br1 / (sigmax * sigmay.unsqueeze(1) * nfree),
    )

    # ------------------------------------------------------------------
    # 7.4 QR Decomposition and solution: solve A (ar2·A = br2)
    # ------------------------------------------------------------------
    # 7.4.1 Solve using QR decomposition
    q, r = torch.linalg.qr(ar2)
    a = torch.linalg.solve_triangular(
        r, torch.bmm(q.transpose(1, 2), br2.unsqueeze(2)), upper=True
    )
    # 7.4.2 Denormalize to calculate final coefficient A
    a = a.squeeze(-1) * (sigmay.unsqueeze(1) / sigmax)

    # ------------------------------------------------------------------
    # 7.5 Return regression coefficients
    # ------------------------------------------------------------------
    return a


# ======================================================================
# 8. Batch multiple linear regression using direct matrix inversion
# ======================================================================
def mregress_batch_inv(
    x: TensorLike, y: TensorLike, measure_errors: TensorLike | None = None
) -> TensorLike:
    r"""Perform weighted least squares regression using direct inversion.

    Parameters
    ----------
    x : torch.Tensor
        shape (group_size, npts, nterm)
        Independent variable data matrix, where group_size is the batch
        size, npts is the number of sample points, nterm is the number
        of coefficients (independent variables) to solve.
    y : torch.Tensor
        shape (group_size, npts)
        Dependent variable data matrix, must contain group_size samples,
        each with npts elements.
    measure_errors : torch.Tensor, optional
        Vector containing standard measurement errors for each point
        y[i]. In the current LASP-Adam-GPU pipeline this argument is
        typically passed as None (or 1), but the interface is kept for
        generality.

    Returns
    -------
    a : torch.Tensor
        shape (group_size, nterm)
        Regression coefficient matrix solved from equation X·A = y.

    Notes
    -----
    - Uses torch.linalg.inv to directly calculate matrix inverse,
      computation speed is slow.

    Examples
    --------
    >>> X = torch.randn(10, 100, 5, dtype=dtype, device=device)
    >>> y = torch.randn(10, 100, dtype=dtype, device=device)
    >>> a = mregress_batch_inv(X, y)
    >>> print(a.shape)
    torch.Size([10, 5])

    """

    # ------------------------------------------------------------------
    # 8.1 Input validation
    # ------------------------------------------------------------------
    # 8.1.1 Check input dimensions and get shape information
    batch_size, npts, nterm = x.shape
    if y.shape != (batch_size, npts):
        raise ValueError("X and Y have incompatible dimensions.")
    # if not x.is_contiguous():
    #     x = x.view(batch_size, npts, nterm)
    # if not y.is_contiguous():
    #     y = y.view(batch_size, npts)

    # 8.1.2 Calculate degrees of freedom
    nfree = npts - 1.0

    # ------------------------------------------------------------------
    # 8.2 Weight calculation and normalization
    # ------------------------------------------------------------------
    if measure_errors is None:
        weights = torch.ones_like(y, dtype=dtype, device=device)
    else:
        if measure_errors.shape != (batch_size, npts):
            raise ValueError("Measure errors have incompatible dimensions.")
        else:
            weights = 1 / (measure_errors**2)
    sw = weights.sum(dim=1, keepdim=True).div_(npts)
    weights = weights / sw

    # ------------------------------------------------------------------
    # 8.3 Weighted processing: calculate X^T·W·X and X^T·W·y
    # ------------------------------------------------------------------
    wx = x * weights.unsqueeze(-1)
    # 8.3.1 ar1 and br1 are X^T·W·X and X^T·W·y respectively
    ar1, br1 = (
        torch.bmm(wx.transpose(1, 2), x),
        torch.bmm(wx.transpose(1, 2), y.unsqueeze(2)).squeeze(-1),
    )
    # 8.3.2 To improve inversion stability
    sigmax, sigmay = (
        torch.sqrt((x * wx).sum(dim=1) / nfree),
        torch.sqrt((weights * y.pow(2)).sum(dim=1) / nfree),
    )
    # 8.3.3 ar2 and br2 are normalized X^T·W·X and X^T·W·y respectively
    ar2, br2 = (
        ar1 / (sigmax.unsqueeze(-1) * sigmax.unsqueeze(1) * nfree),
        br1 / (sigmax * sigmay.unsqueeze(1) * nfree),
    )

    # ------------------------------------------------------------------
    # 8.4 Direct matrix inversion and solution: (X^T·W·X)^(-1)·X^T·W·y
    # ------------------------------------------------------------------
    # 8.4.1 Calculate (X^T·W·X)^(-1)
    ar = torch.linalg.inv(ar2)
    # 8.4.2 Calculate normalized A = (X^T·W·X)^(-1)·X^T·W·y
    a = torch.bmm(ar, br2.unsqueeze(2)).squeeze(-1)
    # 8.4.3 Denormalize to calculate final coefficient A
    a.mul_(sigmay.unsqueeze(1) / sigmax)

    # ------------------------------------------------------------------
    # 8.5 Return regression coefficients
    # ------------------------------------------------------------------
    return a
