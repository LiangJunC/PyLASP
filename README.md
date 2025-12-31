<div align="center">

# **PyLASP** - The Python Version of the LASP

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![SciPy](https://img.shields.io/badge/SciPy-Required-%236680aa.svg)](https://scipy.org/)
[![Astropy](https://img.shields.io/badge/Astropy-Required-%236680aa.svg)](https://www.astropy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Required-%236680aa.svg?)](https://pandas.pydata.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Optional-bluegrey.svg)](https://pytorch.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Optional-bluegrey.svg?)](https://matplotlib.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

</div>

## 📋 Table of Contents
<div>
    <ul>
        <li><a href="#overview"> Overview</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#project-structure">Project Structure</a></li>
        <ul>
        <li><a href="#the-core-modules-in-lasp-curvefit">The Core Modules in LASP-CurveFit</a></li>
        <li><a href="#the-core-modules-in-lasp-adam-gpu">The Core Modules in LASP-Adam-GPU</a></li>
        </ul>
        <li><a href="#workflow">Workflow</a></li>
        <ul>
        <li><a href="#lasp-curvefit-inference-process">LASP-CurveFit Inference Process</a></li>
        <li><a href="#lasp-adam-gpu-inference-process">LASP-Adam-GPU Inference Process</a></li>
        </ul>
        <li><a href="#parameter-inference-example">Parameter Inference Example</a></li>
        <ul>
        <li><a href="#lasp-curvefit-inference-example">LASP-CurveFit Inference Example</a></li>
        <li><a href="#lasp-adam-gpu-inference-example">LASP-Adam-GPU Inference Example</a></li>
        </ul>
        <li><a href="#limitations-and-future-work">Limitations and Future Work</a></li>
        <li><a href="#citation">Citation</a></li>
        <li><a href="#license">License</a></li>
    </ul>
</div>

---

<h2 id="overview">🔭 Overview</h2>

**PyLASP** (The Python Version of the LAMOST Stellar Parameter Pipeline) is a modern, modular reimplementation of the original LASP (**LASP-MPFit**), which was developed in Interactive Data Language (IDL) and employed the [`ULySS`](http://ulyss.univ-lyon1.fr/) software package to infer radial velocity, effective temperature, surface gravity, and metallicity from observed spectra. 

**PyLASP** refactors the **LASP-MPFit** with two complementary modules:
- **LASP-CurveFit** — a new implementation of the **LASP-MPFit** fitting procedure that runs on CPU, preserving legacy logic while improving data I/O and multithreaded execution efficiency.
- **LASP-Adam-GPU** — a GPU accelerated method that introduces grouped optimization by constructing a joint residual function over multiple observed and model spectra, enabling high-throughput parameter inference across tens of millions of spectra.

**PyLASP** provides both **No Clean** and **Clean** strategies:
- **No Clean strategy** — a computationally efficient strategy that fits spectra without iterative pixel rejection. It is faster but may yield lower accuracy for spectra containing significant artifacts.
- **Clean strategy** — an iterative strategy that identifies and rejects anomalous flux points during the fitting process, specifically those whose model–data discrepancies cannot be reasonably explained by the spectral emulator. This approach improves robustness for spectra with defects or irregularities, but is computationally slower than the **No Clean strategy**.

---

<h2 id="installation">🔧 Installation</h2>

Follow the steps below to set up **PyLASP** in a clean conda environment.
1. Create an independent conda environment:

```bash
conda create -n PyLASP-env python=3.10
```
2. Activate the environment:

```bash
conda activate PyLASP-env
```
3. Navigate to the PyLASP project folder:

```bash
cd /path/to/PyLASP

```

4. Install the package and its dependencies:
```bash

pip install -e .
```

⚠️ **Note:** The above steps install only the dependencies for **LASP-CurveFit**. To enable **LASP-Adam-GPU**, install the appropriate PyTorch version in the same environment. See the [`official PyTorch installation guide`](https://pytorch.org/get-started/locally/) for details.

---

<h2 id="project-structure">🖥️ Project Structure</h2>
<h3 id="the-core-modules-in-lasp-curvefit"> The Core Modules in LASP-CurveFit</h3>

```
LASP-CurveFit/
│
├── tgm_model/                            # Spectral emulator
│   └── elodie32_flux_tgm.fits            # ELODIE polynomial coefficients
│
├── test_data/                            # Test data examples
│   ├── LAMOST_spec_fits/                 # LAMOST spectrum FITS files
│   │   └── *.fits
│   └── PyLASP_inferred_results/          # Parameter results inferred by PyLASP
│       └── *.csv
│
├── file_paths.py                         # Obtain the PyLASP file path
│
├── uly_read_lms/                         # Spectrum data reading
│   ├── uly_spect_alloc.py                # Initialize spectrum dictionary
│   ├── uly_spect_get.py                  # Extract fields from the spectrum dictionary
│   ├── uly_spect_extract.py              # Update dictionary entries based on wavelength range
│   └── uly_spect_read_lms.py             # Construct the LAMOST spectrum dictionary (example implementation)
│
├── uly_tgm/                              # Model spectrum structure
│   ├── uly_tgm.py                        # Define model spectrum dictionary
│   └── uly_tgm_init.py                   # Initialize model spectrum dictionary
│
├── uly_tgm_eval/                         # Model spectrum generation
│   └── uly_tgm_eval.py                   # Generate model spectra from parameters
│
├── WRS/                                  # Wavelength resampling
│   ├── xrebin.py                         # Interpolation methods
│   └── uly_spect_logrebin.py             # Spectrum resampling implementation
│
├── resolution_reduction/                 # Resolution matching
│   └── convol.py                         # Spectral resolution reduction
│
├── legendre_polynomial/                  # Shape correction
│   └── mregress.py                       # Legendre polynomial coefficient calculation
│
├── clean_outliers/                       # Outlier rejection
│   └── clean.py                          # Clean strategy
│ 
└── uly_fit/                              # Parameter fitting core
    ├── robust_sigma.py                   # Robust standard deviation calculation
    ├── uly_fit_init.py                   # Initialize a model spectrum dictionary
    ├── uly_makeparinfo.py                # Configure parameters to be optimized
    ├── uly_fit_conv_weight_poly.py       # Model preprocessing: convolution + weighting + shape correction
    ├── uly_fit_a_cmp.py                  # Compute best-fit parameters
    └── ulyss.py                          # Wrapper integrating uly_fit_a_cmp for parameter inference
```

---

<h3 id="the-core-modules-in-lasp-adam-gpu">The Core Modules in LASP-Adam-GPU</h3>

```
LASP-Adam-GPU/
│
├── config/                               # Configuration files for LASP-Adam-GPU
│   └── config.py                         # Data type and device configuration
│
├── tgm_model/                            # Spectral emulator
│   └── elodie32_flux_tgm.fits            # ELODIE polynomial coefficients
│
├── test_data/                            # Test data examples
│   ├── LAMOST_spec_fits/                 # LAMOST spectrum FITS files
│   │   └── *.fits
│   ├── LAMOST_spec_pt/                   # LAMOST spectrum files in .pt format
│   │   └── *.pt
│   └── PyLASP_inferred_results/          # Parameter results inferred by PyLASP
│       └── *.csv
│
├── file_paths.py                         # Obtain the PyLASP file path
│
├── data_to_pt/                           # Convert FITS spectra to .pt format
│   └── data_to_pt.py                     # Step 1: Convert observed spectra to .pt format
│
├── uly_tgm_eval/                         # Model spectrum generation
│   └── uly_tgm_eval_pytorch.py           # Step 2: Generate N model spectra
│
├── WRS/                                  # Wavelength resampling
│   └── xrebin_pytorch.py                 # Step 3: Resample N model spectra to observed wavelengths
│
├── resolution_reduction/                 # Resolution matching
│   └── convol_pytorch.py                 # Step 4: Reduce the resolution of N model spectra to observed spectra
│
├── legendre_polynomial/                  # Shape correction
│   ├── matrix_inverse_benchmark.py       # Efficiency comparison of matrix inversion methods
│   └── mregress_pytorch.py               # Step 5: Correct the shape of N model spectra to match observed spectra
│
│── clean_outliers/                       # Outlier rejection
│   └── clean_pytorch.py                  # Step 6: Clean strategy
│
│── model_err/                            # Parameter uncertainty estimation
│   ├── loss_reduced.py                   # Compute the flux residuals of N spectra
│   └── model_err.py                      # Step 7: Compute the parameter errors of N spectra
│ 
└── uly_fit/                              # Parameter fitting core
    ├── ulyss_pytorch.py                  # Initialize spectrum info for .pt storage (depends on LASP-CurveFit)
    └── uly_fit_conv_poly_pytorch.py      # Run steps 1–7 to compute best-fit parameters and save results to CSV
```

---

<h2 id="workflow">🚀 Workflow</h2>

<h3 id="lasp-curvefit-inference-process">LASP-CurveFit Inference Process</h3>

**Step 1:** Read a target spectrum and store it in a dictionary: [`uly_spect_read_lms.py`](uly_read_lms/uly_spect_read_lms.py)

**Step 2:** Set the initial values for the parameters to be inferred, the Legendre polynomial degree, the model location, and whether to enable the Clean strategy, etc.: [`ulyss.py`](uly_fit/ulyss.py)

**Step 3:** [`ulyss.py`](uly_fit/ulyss.py) further updates the model dictionary and the observed-spectrum dictionary, and passes them to: [`uly_fit_a_cmp.py`](uly_fit/uly_fit_a_cmp.py)

**Step 4:** [`uly_fit_a_cmp.py`](uly_fit/uly_fit_a_cmp.py) constructs the objective function and iteratively calls: [`uly_fit_conv_weight_poly.py`](uly_fit/uly_fit_conv_weight_poly.py), which performs: 
- Generate the model spectra: [`uly_tgm_eval.py`](uly_tgm_eval/uly_tgm_eval.py)
- Resample the model spectra to the observed wavelength grid: [`uly_spect_logrebin.py`](WRS/uly_spect_logrebin.py)
- Match the resolution of the model spectra to the observed spectra: [`convol.py`](resolution_reduction/convol.py)
- Correct the shape of the model spectra to match observed spectra: [`mregress.py`](legendre_polynomial/mregress.py)<br>

and finally saves the inferred results as CSV file.

<h3 id="lasp-adam-gpu-inference-process">LASP-Adam-GPU Inference Process</h3>

**Step 1:** Convert spectrum to .pt format: [`data_to_pt.py`](data_to_pt/data_to_pt.py)

**Step 2:** Configure parameters in [`uly_fit_conv_poly_pytorch.py`](uly_fit/uly_fit_conv_poly_pytorch.py) — including whether to enable the Clean strategy — then construct the objective function and iteratively call:
- Generate N model spectra: [`uly_tgm_eval_pytorch.py`](uly_tgm_eval/uly_tgm_eval_pytorch.py)
- Resample N model spectra to the observed wavelength grid: [`xrebin_pytorch.py`](WRS/xrebin_pytorch.py)
- Match the resolution of N model spectra to the observed spectra: [`convol_pytorch.py`](resolution_reduction/convol_pytorch.py) 
- Correct the shape of N model spectra to match the observed spectra: [`mregress_pytorch.py`](legendre_polynomial/mregress_pytorch.py)
- Apply the Clean strategy (optional): [`clean_pytorch.py`](clean_outliers/clean_pytorch.py)

**Step 3:** Once the objective function converges, [`uly_fit_conv_poly_pytorch.py`](uly_fit/uly_fit_conv_poly_pytorch.py) calls [`model_err.py`](model_err/model_err.py) to compute the parameter errors of N spectra and saves the final results as a CSV file.

---

<h2 id="parameter-inference-example">⚙️ Parameter Inference Example</h2>

<h3 id="lasp-curvefit-inference-example">LASP-CurveFit Inference Example</h3>

- **LASP-CurveFit** is used to infer stellar parameters: see `case 2` in [`tutorial.ipynb`](tutorial.ipynb)
- Individual spectrum parameter inference using `curve_fit`
- Uses `joblib` to provide multiprocessing support for large spectroscopic datasets
- Preserves original IDL logic

<h3 id="lasp-adam-gpu-inference-example">LASP-Adam-GPU Inference Example</h3>

- **LASP-Adam-GPU** is used to infer stellar parameters for N spectra simultaneously: see `case 3` in [`tutorial.ipynb`](tutorial.ipynb)
- Performs multi-spectrum parameter inference using the `Adam` optimizer
- Provides significantly higher throughput for large-scale datasets
- Easily extensible to multi-element or joint-parameter inference

---

<h2 id="limitations-and-future-work">🔄 Limitations and Future Work</h2>

| Feature                   | Current Status                                                       | Planned Improvement                                              | Implementation Plan                                                               |
|---------------------------|----------------------------------------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Initial Parameter Guess   | Single initial guess per spectrum                                    | Support for multiple initializations to improve robustness       | Grid search over parameter space; select solution with lowest χ²                  |
| Wavelength Coverage       | Single continuous range (e.g., 4200-5700 Å)                          | Add support for disjoint wavelength segments (e.g., 4200-4500 Å and 5200-5700 Å) | Apply wavelength mask array; set mask=0 for excluded regions (e.g., 4500-5200 Å)  |
| Two-Stage Fitting         | First-stage implemented                                              | Full two-stage pipeline integration                              | Remove pseudo-continuum from observed spectra manually, then run PyLASP inference |
| Multi-Abundance Inference | Only RV, $T_{\rm eff}$, log $g$, [Fe/H]                              | Joint inference of multiple elemental abundances                 | Multi-objective optimization with extended spectral model                         |
| Legendre Polynomial       | Multiplicative correction supported; additive mode not yet successful| Enable both multiplicative and additive polynomial corrections | Iterative testing and implementation refinement                                     |
| Wavelength Sampling       | Tested with log-uniform grids (ln λ)                                 | Support for linear-uniform and non-uniform wavelength grids    | Progressive testing across different sampling schemes                               |

---

<h2 id="citation">📄 Citation</h2>

When using this code, please cite the following works:

1. [`ULySS: a full spectrum fitting package`](https://ui.adsabs.harvard.edu/abs/2009A&A...501.1269K)
2. [`Coudé-feed stellar spectral library – atmospheric parameters`](https://ui.adsabs.harvard.edu/abs/2011A&A...525A..71W)
3. [`The first data release (DR1) of the LAMOST regular survey`](https://ui.adsabs.harvard.edu/abs/2015RAA....15.1095L)
4. [`Scalable Stellar Parameter Inference Using Python-Based LASP: From CPU Optimization to GPU Acceleration`](https://doi.org/10.3847/1538-4357/ae1446)

---

<h2 id="license">⚖ License</h2>

This project is released under the [`GNU General Public License v3.0`](https://www.gnu.org/licenses/gpl-3.0.html). See the [`LICENSE`](./LICENSE) file for details.

---
