#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2025/09/22 08:38:08
@Author  :   George Trenins
@Desc    :   Base class for calculating model spectral densities and corresponding memory-friction kernels
'''


from __future__ import print_function, division, absolute_import
from abc import ABC, abstractmethod
from functools import wraps
import numbers
import numpy as np
from scipy.integrate import quad


def check_parameters(eta: float, omega_cut: float) -> None:
    """Raise ValueError unless eta ≥ 0 and omega_cut > 0, so also for NaN."""
    if not eta >= 0:
        raise ValueError(f"eta must be non-negative, got {eta}")
    if not omega_cut > 0:
        raise ValueError(f"omega_cut must be positive, got {omega_cut}")


def check_nm_omega(func):
    """Make func raise TypeError unless the normal-mode frequency is a real
    number, such as np.float32 or np.int64, and ValueError unless it is
    non-negative, so also for NaN. func receives it as a Python float, so
    that its arithmetic is in double precision."""
    @wraps(func)
    def wrapper(self, nm_omega: float, *args, **kwargs):
        if not isinstance(nm_omega, numbers.Real):
            raise TypeError(
                f"The normal-mode frequency must be a real number, "
                f"got `{type(nm_omega).__name__}`")
        if not nm_omega >= 0:
            raise ValueError(
                f"The normal-mode frequency must be non-negative, "
                f"got {nm_omega}")
        return func(self, float(nm_omega), *args, **kwargs)
    return wrapper


class BaseSpectralDensity(ABC):
    """Base class for models of a memory kernel K(t) and its cosine
    transform, the spectrum

        Λ(ω) = ∫₀^∞ K(t) cos(ωt) dt.

    Subclasses implement Lambda and K in closed form, both even. The nm
    methods transform them for a normal mode of frequency ωₙ ≥ 0.
    """

    @abstractmethod
    def Lambda(self, omega):
        raise NotImplementedError

    @abstractmethod
    def K(self, t):
        raise NotImplementedError

    @check_nm_omega
    def nmLambda(self, nm_omega, omega):
        """Return the spectrum for a normal mode of frequency ωₙ,

            Λₙ(ω) = θ(x) √x Λ(ω√x),  where x = 1 − ωₙ²/ω²

        and θ is the Heaviside step function, so Λₙ = 0 for |ω| ≤ ωₙ.

        Args:
            nm_omega (float): ωₙ ≥ 0
            omega (float or ndarray): frequencies

        Returns:
            Λ(omega) for ωₙ = 0, else a float array of the shape of omega
        """
        if nm_omega == 0:
            return self.Lambda(omega)
        w = np.abs(omega)
        diff = w - nm_omega
        mask = diff > 0
        # u = |ω|√x, as a product that stays accurate near ωₙ
        u = np.sqrt(diff[mask] * (w[mask] + nm_omega))
        ans = np.zeros_like(omega, dtype=float)
        ans[mask] = u / w[mask] * self.Lambda(u)
        return ans

    @check_nm_omega
    def nmK(self, nm_omega, t, **kwargs):
        """Return the memory kernel for a normal mode of frequency ωₙ,

            Kₙ(t) = (2/π) ∫ Λₙ(ω) cos(ωt) dω, from ωₙ to ∞,

        with scipy.integrate.quad at each t, using the Fourier weight for
        t ≠ 0.

        Args:
            nm_omega (float): ωₙ ≥ 0
            t (float or ndarray): times
            **kwargs: passed to quad, such as epsabs

        Returns:
            float for a Python scalar t, else a float array of the shape
                of np.atleast_1d(t)
        """
        def integrand(w):
            # (2/π) Λₙ(w) for the floats w ≥ ωₙ that quad passes
            diff = w - nm_omega
            if diff <= 0.0:
                return 0.0
            x = np.sqrt(diff * (w + nm_omega))
            return (2.0/np.pi) * self.Lambda(x) * x/w

        t_arr = np.atleast_1d(t).astype(float)
        out = np.empty_like(t_arr)
        for i, ti in enumerate(t_arr):
            if ti == 0.0:
                # the integrand does not oscillate
                out[i] = quad(integrand, nm_omega, np.inf, **kwargs)[0]
            else:
                out[i] = quad(integrand, nm_omega, np.inf,
                              weight='cos', wvar=ti, **kwargs)[0]
        return out[0] if np.isscalar(t) else out
            
