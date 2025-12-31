# -*- coding: utf-8 -*-
# @Time    : 2025/2/2 16:20
# @Author  : ljc
# @FileName: config.py
# @Software: PyCharm
# Update:  2025/11/26 16:44:32


# ======================================================================
# 1. Introduction
# ======================================================================
r"""PyTorch environment configuration for LASP-Adam-GPU and LASP-Adam-CPU.

1.1 Purpose
-----------
Configure PyTorch runtime environment, set computation precision and
random seeds for reducing randomness.

1.2 Functions
-------------
1) default_set: Set default floating-point precision and computation
   device (CPU/GPU).
2) set_all_seeds: Set seeds for all random number generators to reduce
   randomness.

1.3 Explanation
---------------
1.3.1 default_set
This function configures PyTorch's default floating-point data type
and computation device. It supports three precision levels:
    - 16-bit (float16): Lower precision, faster computation, less
      memory
    - 32-bit (float32): Standard precision, balanced performance
    - 64-bit (float64): Higher precision, slower computation, more
      memory
When using CUDA with float32, TF32 acceleration is automatically
enabled for better performance.

1.3.2 set_all_seeds Function
This function sets seeds for all random number generators including
Python's random module, NumPy, and PyTorch (both CPU and GPU). This
helps reduce randomness.

1.4 Notes
---------
- We have observed that LASP-Adam-GPU may produce slightly different
  results on each run, even with all random seeds set. This is due
  to:
    1) Non-deterministic operations in CUDA (e.g., atomicAdd in
       parallel reduction)
    2) Floating-point arithmetic differences in parallel execution
    3) Optimization strategies in cuDNN and cuBLAS libraries
    4) Hardware-level variations in GPU computation order
- LASP-Adam-CPU produces consistent results across runs because CPU
  operations are inherently sequential and deterministic.
- The differences in GPU results are typically minor (on the order
  of numerical precision) and do not affect scientific conclusions,
  but users should be aware of this behavior when reproducing exact
  numerical values.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
import torch
import numpy as np
import random
import os

# 2.1 Limit to single GPU by default
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ======================================================================
# 3. Set default floating-point type and computation device
# =====================================================================
def default_set(type=32) -> tuple:
    r"""Set default floating-point precision and computation device.

    Parameters
    ----------
    type : int, default=32
        Floating-point precision in bits, can be 16, 32, or 64.

    Returns
    -------
    dtype : torch.dtype
        PyTorch data type corresponding to the specified precision.
    device : torch.device
        Computation device (cuda or cpu).

    Raises
    ------
    ValueError
        If the provided type parameter is not a supported value.

    Notes
    -----
    - GPU (CUDA) is used by default if available, otherwise falls back
      to CPU.
    - When using float32 on CUDA, TF32 acceleration is automatically
      enabled for matrix multiplication operations.
    - float16 provides faster computation but lower precision.
    - float64 provides higher precision but slower computation.

    """

    # ------------------------------------------------------------------
    # 3.1 Determine computation device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 3.2 Set data type based on precision parameter
    # ------------------------------------------------------------------
    if type == 16:
        dtype = torch.float16
    elif type == 32:
        dtype = torch.float32
    elif type == 64:
        dtype = torch.float64
    else:
        raise ValueError(
            f"Not supported floating point precision: {type}, please use 16, 32 or 64"
        )

    # ------------------------------------------------------------------
    # 3.3 Set global default data type
    # ------------------------------------------------------------------
    torch.set_default_dtype(dtype)

    # ------------------------------------------------------------------
    # 3.4 Enable TF32 acceleration for float32 on CUDA
    # ------------------------------------------------------------------
    if type == 32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # 3.5 Return data type and computation device
    # ------------------------------------------------------------------
    return dtype, device


# ======================================================================
# 4. Set random seeds
# ======================================================================
def set_all_seeds(seed=666) -> None:
    r"""Set seeds for all random number generators to reduce randomness.

    Parameters
    ----------
    seed : int, default=666
        Random seed value to use for all random number generators.

    Returns
    -------
    None

    Notes
    -----
    - This function sets seeds for Python's random module, NumPy, and
      PyTorch (both CPU and GPU).
    - LASP-Adam-GPU: We have observed that even with all random seeds
      set, LASP-Adam-GPU may produce slightly different results on
      each run. This non-determinism is caused by:
      1) CUDA's parallel reduction operations (e.g., atomicAdd) that
         execute in non-deterministic order
      2) Floating-point arithmetic variations in parallel GPU threads
      3) cuDNN and cuBLAS library optimizations that prioritize speed
         over determinism
      4) Hardware-level variations in GPU computation order
    - LASP-Adam-CPU: In contrast, LASP-Adam-CPU produces exactly the
      same results across multiple runs with the same random seed.
      This is because CPU operations are inherently sequential and
      deterministic, without the parallelism-induced non-determinism
      present in GPU computations.

    """

    # ------------------------------------------------------------------
    # 4.1 Set Python's random seed
    # ------------------------------------------------------------------
    random.seed(seed)

    # ------------------------------------------------------------------
    # 4.2 Set NumPy's random seed
    # ------------------------------------------------------------------
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 4.3 Set PyTorch's random seed for CPU
    # ------------------------------------------------------------------
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # 4.4 Configure GPU-related random seeds and settings
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------
    # 4.5 Set Python hash seed for dictionary consistency
    # ------------------------------------------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ------------------------------------------------------------------
    # 4.6 Additional deterministic settings (optional)
    # ------------------------------------------------------------------
    # if hasattr(torch, 'use_deterministic_algorithms'):
    #     torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
