# -*- coding: utf-8 -*-
# @Time    : 2025/1/28 11:26
# @Author  : ljc
# @FileName: file_paths.py
# @Software: PyCharm
# Update:  2025/11/26 22:26:27


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Path management for PyLASP project file structure.

1.1 Purpose
-----------
Provide centralized path management for PyLASP project, enabling easy
access to model files, test data, and root directory across all modules.

1.2 Functions
-------------
1) TGM_MODEL_FILE: Return the path to the stellar spectral model file
   (elodie32_flux_tgm.fits).
2) TEST_DATA_DIR: Return the path to the test_data directory
   containing sample spectral data.
3) LASP_ROOT: Return the PyLASP root directory path.

1.3 Explanation
---------------
1.3.1 Path Resolution Strategy
All paths are constructed relative to the location of this file
(file_paths.py), which should be placed in the PyLASP root directory.

1.3.2 File Structure
The expected PyLASP directory structure:
    PyLASP/
    ├── file_paths.py (this file)
    ├── tgm_model/
    │   └── elodie32_flux_tgm.fits
    ├── test_data/
    │   └── LAMOST_spec_fits/
    │       └── *.fits
    └── ... (other modules)

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
import os


# ======================================================================
# 3. Define PyLASP root directory path
# ======================================================================
LASP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# 4. Model path
# ======================================================================
def TGM_MODEL_FILE() -> str:
    file = os.path.join(LASP_ROOT_PATH, "tgm_model/elodie32_flux_tgm.fits").replace(
        "\\", "/"
    )
    return file


# ======================================================================
# 5. Test samples path
# ======================================================================
def TEST_DATA_DIR() -> str:
    file = os.path.join(LASP_ROOT_PATH, "test_data/").replace("\\", "/")
    return file


# ======================================================================
# 6. PyLASP path
# ======================================================================
def LASP_ROOT() -> str:
    file = os.path.join(LASP_ROOT_PATH, "").replace("\\", "/")

    return file
