#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bath.py
@Time    :   2026/01/15 10:28:05
@Author  :   George Trenins
@Desc    :   Utilities for describing dissipation
'''

from __future__ import print_function, division, absolute_import
import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar


def WAM_discretization(
    omega: npt.NDArray[np.floating],
    Lambda: npt.NDArray[np.floating],
    nbath: int,
    wmax: float = np.inf,
) -> tuple[float, npt.NDArray[np.floating]]:
    """Discretize a bath spectrum Λ(ω) into nbath harmonic modes of equal
    weight, by the scheme of P. L. Walters, T. C. Allen and N. Makri,
    J. Comput. Chem. 38, 110 (2017).

    Let C(ω) be the integral of Λ from omega[0] to ω, and ω_max the smaller
    of wmax and omega[-1]. Each mode has the weight ε = C(ω_max)/nbath, and
    the frequency ωⱼ that solves C(ωⱼ) = (j − ½)ε, for j = 1, ..., nbath.

    Args:
        omega (ndarray): increasing frequencies, from 0 to discretize all
            of Λ
        Lambda (ndarray): Λ at omega, non-negative, interpolated by a cubic
            spline
        nbath (int): number of modes
        wmax (float, optional): upper limit of the integral. Defaults to
            omega[-1].

    Returns:
        tuple: the weight ε (float) and the nbath frequencies (ndarray)

    Raises:
        RuntimeError: if the bisection for a frequency does not converge

    Notes:
        Neither the start of the grid nor the sign of Λ is checked. A grid
        from ω₀ > 0 discretizes Λ on [ω₀, ω_max] only, and where Λ < 0, C
        is not monotonic and the frequencies are meaningless.
    """
    antiderivative = make_interp_spline(omega, Lambda).antiderivative(nu=1)
    wmax = min(wmax, omega[-1])

    def cumulative_spectrum(x):
        # zero outside the grid
        return np.nan_to_num(antiderivative(x, extrapolate=False), nan=0.0)

    weight = cumulative_spectrum(wmax) / nbath
    freqs = []
    prev = 0.0
    for j in range(nbath):
        ans = root_scalar(
            lambda x: cumulative_spectrum(x) - (j + 1/2) * weight,
            method="bisect", bracket=[prev, wmax])
        # root_scalar returns rather than raises if the bisection fails
        if not ans.converged:
            raise RuntimeError(
                f"Failed to find discrete frequency number {j+1}")
        freqs.append(ans.root)
        prev = ans.root
    return weight, np.asarray(freqs)
