# -*- coding: utf-8 -*-
# @Time    : 2025/11/20 14:50
# @Author  : ljc
# @FileName: uly_makeparinfo.py
# @Software: Pycharm
# Update:  2025/11/26 21:24:13


# ======================================================================
# 1. Introduction
# ======================================================================
r"""Utilities for building parameter-information structures for LASP-CurveFit.

1.1 Purpose
-----------
Provide helper functions to initialize LOSVD parameters and stellar
parameter information, converted from IDL uly_makeparinfo related
routines.

1.2 Functions
-------------
1) uly_init_losvd_params: Initialize LOSVD parameter guesses, limits,
   and parameter-information dictionaries.
2) uly_init_stellar_params: Extract and flatten parameter-information
   from a component (cmp) or multi-component (cmps) structure.
3) uly_get_infer_params_set: Merge LOSVD and stellar parameter
   information into a single list suitable for numerical optimizers.

1.3 Explanation
---------------
These helpers decouple the construction of parameter-information
(parinfo) from the core fitting routine. LOSVD parameters (cz, sigma,
higher-order Gauss-Hermite moments) are initialized with physically
sensible defaults and transformed from velocity space (km/s) to
logarithmic wavelength space (ln lambda) in pixel units. Stellar
parameters are taken directly from the component structure
cmp["para"] and converted into a unified parinfo list, which is
then passed to the fitting engine (e.g. uly_fit_a_cmp).

1.4 Notes
---------
- This is a Python-specific rewrite and optimization, not a complete
  port of all IDL features, applicable to LASP-CurveFit.
- For PyLASP, only a single component (n_cmp = 1) is used at present.
  Support for multiple components can be added gradually if needed.

"""


# ======================================================================
# 2. Import libraries
# ======================================================================
import numpy as np
import copy


# ======================================================================
# 3. Type definitions and constants
# ======================================================================
ArrayLike = np.ndarray
C_light: float = 299792.458  # Speed of light


# ======================================================================
# 4. Initialize LOSVD parameter information
# ======================================================================
def uly_init_losvd_params(
    velscale: float,
    obs_step: float,
    npix: int,
    voff: float,
    kmoment: int = 2,
    cz_guess: float | None = None,
    sigma_guess: float | None = None,
    kfix: ArrayLike | None = None,
    klimits: ArrayLike | None = None,
) -> list[dict] | None:
    r"""Initialize LOSVD parameters.

    Parameters
    ----------
    velscale : float
        Velocity per step in ln wavelength space (km/s).
        Note: velscale = ln_step * c = log10_step * c * ln(10).
    obs_step : float
        Logarithmic wavelength step in ln(lambda) per pixel of
        observation.
    npix : int
        Number of pixels in the fitting region.
    voff : float
        Velocity offset in km/s that may be applied to the model grid.
    kmoment : int
        Order of Gauss-Hermite moments. Set to 2 for [cz, sigma],
        4 for [cz, sigma, h3, h4], 6 for [cz, sigma, h3, h4, h5, h6].
        LASP sets to 2.
    cz_guess : float or None, optional
        Initial guess for LOSVD first parameter.
    sigma_guess : float or None, optional
        Initial guess for LOSVD second parameter.
    kfix : np.ndarray or None, optional
        Whether to fix LOSVD parameters. 1 fixes corresponding LOSVD
        parameter during minimization.
    klimits : np.ndarray or None, optional
        Upper and lower bounds for parameters to test.

    Returns
    -------
    parinfok : list of dict or None
        If kmoment == 0, returns None. Otherwise, returns a list
        of kmoment parameter-information dictionaries.

    Raises
    ------
    ValueError
        - If kmoment is not one of [0, 2, 4, 6].
        - If a kinematic guess lies outside the corresponding limits.
        - If the wavelength range is too small compared to the
          combination of voff, cz_guess and sigma_guess.

    Notes
    -----
    - LOSVD parameter limits for cz and sigma are converted from km/s
      to pixel units via log(1 + v / c) / obs_step.
    - Higher-order Gauss-Hermite moments (h3, h4, ...) are kept in
      linear units with default limits [-0.3, 0.3].

    Examples
    --------
    >>> parinfok = uly_init_losvd_params(
    ...     cz_guess=0.0,
    ...     sigma_guess=100.0,
    ...     velscale=50.0,
    ...     kmoment=2,
    ...     kfix=None,
    ...     klimits=None,
    ...     obs_step=1e-4,
    ...     npix=3000,
    ...     voff=0.0,
    ... )
    >>> len(parinfok)
    2
    >>> parinfok[0]["fixed"]
    0

    """

    # ------------------------------------------------------------------
    # 4.1 Initialize LOSVD guesses
    # ------------------------------------------------------------------
    if cz_guess is None:
        cz_guess = 0
    if sigma_guess is None:
        sigma_guess = velscale
    kguess = [cz_guess, sigma_guess]
    if kmoment not in [0, 2, 4, 6]:
        raise ValueError("Kmoment should be 0, 2, 4 or 6.")
    if kmoment > 2:
        kguess = kguess + [0] * (kmoment - 2)

    # ------------------------------------------------------------------
    # 4.2 Handle fixed/free flags (kfix) for each LOSVD parameter
    # ------------------------------------------------------------------
    nst = len(kguess)
    if kfix is None:
        kfix = np.zeros(nst, dtype=int)
    elif len(kfix) < nst:
        kfix = np.concatenate([kfix, np.zeros(nst - len(kfix), dtype=int)])
    elif len(kfix) > nst:
        kfix = kfix[:nst]

    # ------------------------------------------------------------------
    # 4.3 Parameter ranges for LOSVD moments in velocity space
    # ------------------------------------------------------------------
    if kmoment > 0:
        if klimits is None:
            klimits = np.zeros((2, kmoment))
        klimits[0:2, 0] = kguess[0] + np.array([-2e3, 2e3])
        if kmoment >= 2:
            klimits[0:2, 1] = [0.3 * obs_step * C_light, 1e3]
            for k in range(2, kmoment):
                klimits[0:2, k] = [-0.3, 0.3]
        for k in range(kmoment):
            if (kfix[k] == 0) and (kguess[k] < klimits[0, k]):
                raise ValueError(
                    f"Guess on kinematic moment {k}: {kguess[k]} "
                    f"is lower than the limit: {klimits[0, k]}"
                )
            elif (kfix[k] == 0) and (kguess[k] > klimits[1, k]):
                raise ValueError(
                    f"Guess on kinematic moment {k}: {kguess[k]} "
                    f"is higher than the limit: {klimits[1, k]}"
                )
        klimits[0:2, 0] = np.log(1 + klimits[0:2, 0] / C_light) / obs_step
        if kmoment >= 2:
            klimits[0:2, 1] = np.log(1 + klimits[0:2, 1] / C_light) / obs_step

    # ------------------------------------------------------------------
    # 4.4 Store LOSVD information as a list of parameter dictionaries
    # ------------------------------------------------------------------
    if kmoment == 0:
        parinfok = None
    if kmoment > 0:
        parinfok = [
            {
                "value": 0.0,
                "step": 1e-2,
                "limits": [0.0, 0.0],
                "limited": [1, 1],
                "fixed": 0,
            }
            for _ in range(kmoment)
        ]
        parinfok[0]["value"] = np.log(1 + kguess[0] / C_light) / obs_step
        parinfok[0]["limits"] = klimits[:, 0]
        if kfix[0] == 1:
            parinfok[0]["fixed"] = 1
            parinfok[0]["step"] = 0
            parinfok[0]["limits"] = [parinfok[0]["value"], parinfok[0]["value"]]
    if kmoment > 1:
        parinfok[1]["value"] = np.log(1 + kguess[1] / C_light) / obs_step
        parinfok[1]["limits"] = klimits[:, 1]
        if kfix[1] == 1:
            parinfok[1]["fixed"] = 1
            parinfok[1]["step"] = 0
            parinfok[1]["limits"] = [parinfok[1]["value"], parinfok[1]["value"]]
        if npix <= ((abs(voff) + abs(kguess[0]) + 5 * kguess[1]) / velscale):
            raise ValueError(
                "Wavelength range is too small, or velocity shift too big."
            )
    if kmoment > 2:
        for k in range(2, kmoment):
            parinfok[k]["value"] = kguess[k]
            parinfok[k]["limits"] = [-0.3, 0.3]
            parinfok[k]["step"] = 1e-3
        fixh = np.where(kfix == 1)[0]
        if len(fixh) > 0:
            for idx in fixh:
                parinfok[idx]["fixed"] = 1
                parinfok[idx]["step"] = 0
                parinfok[idx]["limits"] = [kguess[idx], kguess[idx]]

    # ------------------------------------------------------------------
    # 4.5 Return initial LOSVD parameter-information list
    # ------------------------------------------------------------------
    return parinfok


# ======================================================================
# 5. Extract initial stellar-parameter information from cmp structure
# ======================================================================
def uly_init_stellar_params(
    cmp: dict | list[dict], deep_copy: bool = False
) -> list[dict]:
    """Construct parameter-information from component structure.

    Parameters
    ----------
    cmp : dict or list of dict
        Component structure(s). For single-component fits, this is a
        single component dictionary. For multi-component fits, this
        should be a list of component dictionaries.
    deep_copy : bool, optional
        Controls whether cmp and poly are deep-copied during the
        iterative optimization. If False (default), uly_fitfunc
        operates in-place on the external poly["poly"] and
        cmp["weight"], reproducing the behavior of the original IDL
        implementation. If True, uly_fitfunc works on internal deep
        copies created at each call, so that the external poly["poly"]
        and cmp["weight"] remain unchanged throughout the fit.

    Returns
    -------
    list
        pinf : list of dict
            A flattened list of parameter-information dictionaries.

    Raises
    ------
    ValueError
        - If a component missing parameter information.

    Notes
    -----
    - For PyLASP, cmp is a single component for now (n_cmp,
      number_cmp = 1); support for multiple components (n_cmp,
      number_cmp != 1) will be added gradually.

    Examples
    --------
    Single-component example
    ~~~~~~~~~~~~~~~~~~~~~~~~
    >>> cmp = {
    ...     "para": [
    ...         {
    ...             "guess": [5800.0],
    ...             "step": 50.0,
    ...             "limits": [4500.0, 7000.0],
    ...             "limited": [1, 1],
    ...             "fixed": 0,
    ...         },
    ...         {
    ...             "guess": [4.3],
    ...             "step": 0.1,
    ...             "limits": [0.0, 5.0],
    ...             "limited": [1, 1],
    ...             "fixed": 0,
    ...         },
    ...     ]
    ... }
    >>> pinf = uly_init_stellar_params(cmp)
    >>> len(pinf)
    2
    >>> pinf[0]["value"]
    [5800.0]

    Multi-component example
    ~~~~~~~~~~~~~~~~~~~~~~~
    >>> cmp_multi = [
    ...     {
    ...         "para": [
    ...             {
    ...                 "guess": [5800.0],
    ...                 "step": 50.0,
    ...                 "limits": [4500.0, 7000.0],
    ...                 "limited": [1, 1],
    ...                 "fixed": 0,
    ...             }
    ...         ]
    ...     },
    ...     {
    ...         "para": [
    ...             {
    ...                 "guess": [5000.0],
    ...                 "step": 50.0,
    ...                 "limits": [3500.0, 6500.0],
    ...                 "limited": [1, 1],
    ...                 "fixed": 0,
    ...             }
    ...         ]
    ...     },
    ... ]
    >>> pinf_multi = uly_init_stellar_params(cmp_multi)
    >>> len(pinf_multi)
    2
    >>> values = [p["value"] for p in pinf_multi]
    >>> print(values)
    [[5800.0], [5000.0]]

    """

    # ------------------------------------------------------------------
    # 5.1 Normalize to a component list: support possible
    #     multi-component cases
    # ------------------------------------------------------------------
    if deep_copy:
        cmp = copy.deepcopy(cmp)
    cmp = [cmp] if isinstance(cmp, dict) else list(cmp)

    # ------------------------------------------------------------------
    # 5.2 Collect parameter information from all components
    # ------------------------------------------------------------------
    all_para = []
    for comp in cmp:
        for par in comp.get("para", None):
            if par is None:
                raise ValueError("Lack of parameter information in component.")
            else:
                all_para.append(par)

    # ------------------------------------------------------------------
    # 5.3 Extract key information for all free parameters
    # ------------------------------------------------------------------
    pinf = [
        {
            "value": par["guess"],
            "step": par.get("step", 1e-2),
            "limits": par.get("limits", [0.0, 0.0]),
            "limited": par.get("limited", [1, 1]),
            "fixed": par.get("fixed", 0),
        }
        for par in all_para
    ]

    # ------------------------------------------------------------------
    # 5.4 Return initial parameter information
    # ------------------------------------------------------------------
    return pinf


# ======================================================================
# 6. Merge LOSVD and stellar parameter-information lists
# ======================================================================
def uly_get_infer_params_set(
    cmp: dict | list[dict],
    velscale: float,
    obs_step: float,
    npix: int,
    voff: float,
    kmoment: int = 2,
    cz_guess: float | None = None,
    sigma_guess: float | None = None,
    kfix: ArrayLike | None = None,
    klimits: ArrayLike | None = None,
    deep_copy: bool = False,
) -> list[dict]:
    r"""Merge LOSVD and stellar parameter-information lists.

    Parameters
    ----------
    cmp : dict or list of dict
        Component structure(s). For single-component fits, this is a
        single component dictionary. For multi-component fits, this
        should be a list of component dictionaries.
    velscale : float or None, optional
        Velocity per step in ln wavelength space (km/s).
        Note: velscale = ln_step * c = log10_step * c * ln(10).
    obs_step : float
        Logarithmic wavelength step in ln(lambda) per pixel of
        observation.
    npix : int
        Number of pixels in the fitting region.
    voff : float
        Velocity offset in km/s that may be applied to the model grid.
    kmoment : int, optional
        Order of Gauss-Hermite moments. Set to 2 for [cz, sigma],
        4 for [cz, sigma, h3, h4], 6 for [cz, sigma, h3, h4, h5, h6].
        LASP sets to 2.
    cz_guess : float or None, optional
        Initial guess for LOSVD first parameter.
    sigma_guess : float or None, optional
        Initial guess for LOSVD second parameter.
    kfix : np.ndarray or None, optional
        Whether to fix LOSVD parameters. 1 fixes corresponding LOSVD
        parameter during minimization.
    klimits : np.ndarray or None, optional
        Upper and lower bounds for parameters to test.
    deep_copy : bool, optional
        Controls whether cmp and poly are deep-copied during the
        iterative optimization. If False (default), uly_fitfunc
        operates in-place on the external poly["poly"] and
        cmp["weight"], reproducing the behavior of the original IDL
        implementation. If True, uly_fitfunc works on internal deep
        copies created at each call, so that the external poly["poly"]
        and cmp["weight"] remain unchanged throughout the fit.

    Returns
    -------
    parinfo : list of dict
        Full parameter-information list used by the fitting routine.

    Notes
    -----
    - This helper acts as a thin wrapper around uly_init_losvd_params
      and uly_init_stellar_params, ensuring a consistent ordering
      of parameters.

    Examples
    --------
    >>> cmp = {
    ...     "para": [
    ...         {"guess": [5800.0], "step": 50.0, "limits": [4500.0, 7000.0]},
    ...         {"guess": [4.3], "step": 0.1, "limits": [0.0, 5.0]},
    ...     ]
    ... }
    >>> parinfo = uly_get_infer_params_set(
    ...     cmp=cmp,
    ...     cz_guess=0.0,
    ...     sigma_guess=100.0,
    ...     velscale=50.0,
    ...     kmoment=2,
    ...     kfix=None,
    ...     klimits=None,
    ...     obs_step=1e-4,
    ...     npix=3000,
    ...     voff=0.0,
    ... )
    >>> len(parinfo) >= 2
    True
    >>> isinstance(parinfo[0], dict)
    True

    """

    # ------------------------------------------------------------------
    # 6.1 Initialize stellar-parameter list
    # ------------------------------------------------------------------
    parinfo = uly_init_stellar_params(cmp=cmp, deep_copy=deep_copy)

    # ------------------------------------------------------------------
    # 6.2 Initialize LOSVD parameter-information list
    # ------------------------------------------------------------------
    parinfok = uly_init_losvd_params(
        cz_guess=cz_guess,
        sigma_guess=sigma_guess,
        velscale=velscale,
        kmoment=kmoment,
        kfix=kfix,
        klimits=klimits,
        obs_step=obs_step,
        npix=npix,
        voff=voff,
    )

    # ------------------------------------------------------------------
    # 6.3 Merge LOSVD parameters and stellar parameters
    # ------------------------------------------------------------------
    if parinfok is not None:
        parinfo = parinfok + parinfo

    # ------------------------------------------------------------------
    # 6.4 Return full parameter-information set
    # ------------------------------------------------------------------
    return parinfo
