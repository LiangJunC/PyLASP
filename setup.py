# -*- coding: utf-8 -*-
# @Time    : 2025/1/28 11:26
# @Author  : ljc
# @FileName: setup.py
# @Software: PyCharm
# Update:  2025/11/26 22:30:00


# ======================================================================
# 1. Introduction
# ======================================================================
r"""PyLASP package installation configuration.

Purpose
-------
Configure and install the PyLASP package with all required dependencies
for stellar parameter inference from spectroscopic data.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
from setuptools import setup, find_packages
import sys


# ======================================================================
# 3. Check Python version compatibility
# ======================================================================
if sys.version_info < (3, 9):
    sys.exit("Python 3.9 or later is required.")


# ======================================================================
# 4. Define package dependencies
# ======================================================================
install_requires = [
    "numpy==1.26.4",
    "pandas==2.2.3",
    "astropy==6.0.1",
    "scipy==1.13.1",
    "matplotlib==3.9.4",
    "tqdm==4.67.1",
    "joblib==1.4.2",
    "filelock==3.18.0",
    # 'torch==2.5.0',
    # PyTorch version. Note: default installation uses CPU version
    # (LASP-CurveFit-CPU). For GPU version (LASP-Adam-GPU), please
    # refer to PyTorch official website
    # GPU installation command example: pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
]


# ======================================================================
# 5. Configure and install PyLASP package
# ======================================================================
setup(
    name="PyLASP",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=install_requires,
    python_requires=">=3.9",
)
