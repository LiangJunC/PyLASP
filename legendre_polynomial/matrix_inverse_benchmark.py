# -*- coding: utf-8 -*-
# @Time    : 2025/1/20 17:20
# @Author  : ljc
# @FileName: matrix_inverse_benchmark.py
# @Software: PyCharm
# Update:  2025/11/26 20:19:30


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Performance comparison of matrix inversion methods for regression.

1.1 Purpose
-----------
Compare computational efficiency of different matrix inversion methods
in multiple linear regression to provide reference for selecting
appropriate solution methods in practical applications.

1.2 Functions
-------------
1) benchmark_methods_batched: Batch test performance of different
   matrix decomposition methods, including runtime and result
   consistency verification.

1.3 Explanation
---------------
This module evaluates performance of five matrix inversion methods:
1.3.1 Method Overview
1) Cholesky decomposition (mregress_batch_cholesky): Fast computation,
   good numerical stability, recommended for positive definite matrices.
2) LU decomposition (mregress_batch_lu): Supports batch processing,
   suitable for large-scale datasets, controllable memory usage.
3) SVD decomposition (mregress_batch_svd): Highest numerical stability,
   can handle non-full-rank matrices, but slower computation.
4) QR decomposition (mregress_batch_qr): Avoids explicit inversion,
   provides good balance between efficiency and stability.
5) Direct inversion (mregress_batch_inv): Uses torch.linalg.inv to
   directly calculate matrix inverse, slower computation.

1.3.2 Performance Testing Process
Steps:
    1) Process input data in chunks to simulate batch computation
       scenarios in practical applications.
    2) Execute multiple runs for each method, calculate average runtime
       to reduce impact of random fluctuations.
    3) Use Cholesky decomposition results as baseline to verify
       consistency of results from different methods.
    4) Output detailed performance report including runtime per run,
       average time, and result differences.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
import time
import torch
from config.config import default_set, set_all_seeds

# 2.1 Set random seed
set_all_seeds()
# 2.2 Call GPU and specify data type
dtype, device = default_set()
# 2.3 Set default data type
torch.set_default_dtype(dtype)
from mregress_pytorch import (
    mregress_batch_cholesky,
    mregress_batch_lu,
    mregress_batch_svd,
    mregress_batch_qr,
    mregress_batch_inv,
)


# ======================================================================
# 3. Type definitions for better code readability
# ======================================================================
TensorLike = torch.Tensor


# ======================================================================
# 4. Batch performance testing function
# ======================================================================
def benchmark_methods_batched(
    x: TensorLike,
    y: TensorLike,
    group_size: int = 10,
    num_runs: int = 5,
) -> tuple[dict, dict]:
    r"""Perform batch performance testing on matrix inversion methods.

    Parameters
    ----------
    x : torch.Tensor
        shape (total_size, npts, nterm)
        Independent variable data matrix, where total_size is total
        number of samples, npts is number of sample points, nterm is
        number of coefficients to solve.
    y : torch.Tensor
        shape (total_size, npts)
        Dependent variable data matrix, must match first two dimensions
        of x.
    group_size : int, default=10
        Number of samples processed per batch, used to control memory
        usage. Smaller batch size reduces peak memory requirements.
    num_runs : int, default=5
        Number of repeated runs for each method to calculate average
        performance. More runs provide more stable statistical results.

    Returns
    -------
    times : dict[str, float]
        Average runtime (seconds) for each method, keys are method
        names, values are corresponding average times.
    results : dict[str, torch.Tensor]
        Regression coefficient matrices calculated by each method, keys
        are method names, values are tensors of shape (total_size, nterm).

    Notes
    -----
    - Execute multiple runs to obtain stable average performance metrics
      and compare computation results of different methods.
    - Results from all methods are compared against Cholesky
      decomposition as baseline to verify consistency.

    """

    # ------------------------------------------------------------------
    # 4.1 Method configuration
    # ------------------------------------------------------------------
    # 4.1.1 Define matrix inversion methods to be tested
    methods = {
        "Cholesky": mregress_batch_cholesky,
        "LU": mregress_batch_lu,
        "SVD": mregress_batch_svd,
        "QR": mregress_batch_qr,
        "inv": mregress_batch_inv,
    }

    # ------------------------------------------------------------------
    # 4.2 Data chunking
    # ------------------------------------------------------------------
    x_chunks, y_chunks = (
        torch.chunk(x, x.size(0) // group_size),
        torch.chunk(y, y.size(0) // group_size),
    )

    # ------------------------------------------------------------------
    # 4.3 Initialize result storage
    # ------------------------------------------------------------------
    results, times = {}, {}

    # ------------------------------------------------------------------
    # 4.4 Main performance testing loop
    # ------------------------------------------------------------------
    # 4.4.1 Iterate through each matrix inversion method
    for name, method in methods.items():
        print(f"\nTesting {name} method:")
        # --------------------------------------------------------------
        # 4.4.1.1 Multiple uuns for stable average performance
        # --------------------------------------------------------------
        total_time = 0
        for run in range(num_runs):
            start = time.perf_counter()
            # ----------------------------------------------------------
            # 4.4.1.1.1 Batch process all data chunks
            # ----------------------------------------------------------
            results_chunks = [
                method(x_chunk, y_chunk) for x_chunk, y_chunk in zip(x_chunks, y_chunks)
            ]
            result = torch.cat(results_chunks)

            # ----------------------------------------------------------
            # 4.4.1.1.2 End timing for single run
            # ----------------------------------------------------------
            end = time.perf_counter()
            run_time = end - start
            total_time += run_time
            print(f"Run {run + 1}: {run_time * 1000:.2f} ms")

        # --------------------------------------------------------------
        # 4.4.1.2 Calculate and store average time
        # --------------------------------------------------------------
        avg_time = total_time / num_runs
        times[name] = avg_time
        results[name] = result

    # ------------------------------------------------------------------
    # 4.5 Performance summary output
    # ------------------------------------------------------------------
    print("\nTotal Performance comparison:")
    for name, t in times.items():
        print(f"{name:8s}: {t * 1000:.2f} ms average per full dataset")

    # ------------------------------------------------------------------
    # 4.6 Result consistency verification
    # ------------------------------------------------------------------
    base_result = results["Cholesky"]
    print("\nResult differences from Cholesky:")
    for name, result in results.items():
        if name != "Cholesky":
            diff = torch.abs(base_result - result).max().item()
            print(f"{name:8s}: {diff:.2e}")

    # ------------------------------------------------------------------
    # 4.7 Return performance statistics and computation results
    # ------------------------------------------------------------------
    return times, results


# ======================================================================
# 5. Usage examples
# ======================================================================
# x = torch.randn(2500, 1325, 51, device=device, dtype=dtype)
# y = torch.randn(2500, 1325, device=device, dtype=dtype)
# times, results = benchmark_methods_batched(x, y, group_size=30)
