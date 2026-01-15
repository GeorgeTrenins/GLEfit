#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bath.py
@Time    :   2026/01/15 10:28:05
@Author  :   George Trenins
@Desc    :   Utilities for describing dissipation
'''

from __future__ import print_function, division, absolute_import
from collections.abc import Callable
from scipy.optimize import root_scalar, RootResults
import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_interp_spline, BSpline


def WAM_discretization(
    omega: npt.NDArray[np.floating], 
    Lambda: npt.NDArray[np.floating], 
    nbath: int,
    **kwargs
) -> tuple[float, npt.NDArray[np.floating]]:
    """Discretize the bath spectrum Λ(ω) = J(ω)/ω according to the scheme
    by P. L. Walters, T. C. Allen, and N. Makri in J. Comp. Chem., 38, pp. 110-115 (2017) [DOI: 10.1002/jcc.24527]

    Args:
        Lambda (Callable[[float], float]): cosine transform of the memory kernel
        nbath (int): number of discrete bath modes

    Returns:
        weight: float
        freqs: np.ndarray
    """
    
    spline: BSpline = make_interp_spline(omega, Lambda, **kwargs)
    antiderivative: BSpline = spline.antiderivative(nu=1)
    wmax: float = omega[-1]
    cumulative_spectrum = lambda x: np.nan_to_num(
        antiderivative(x, extrapolate=False), nan=0.0)
    exact_reorganisation = 4/np.pi * cumulative_spectrum(wmax)  # Eq. (2.5)
    weight = np.pi * exact_reorganisation / (4 * nbath)
    freqs = []
    prev = 0.0 
    for j in range(nbath):
        fun = lambda x: ( cumulative_spectrum(x) - (j+1/2) * weight)  # Eq. (2.6)
        ans: RootResults = root_scalar(fun, method="bisect", bracket=[prev, wmax])
        if not ans.converged:
            raise RuntimeError(f"Failed to find discrete frequency number {j+1}")
        freqs.append(ans.root)
        prev = ans.root
    freqs = np.asarray(freqs)
    return weight, freqs