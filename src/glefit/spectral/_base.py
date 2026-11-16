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


def check_parameters(eta: float, omega_cut: float) -> None:
    """Raise ValueError unless eta ≥ 0 and omega_cut > 0, so also for NaN."""
    if not eta >= 0:
        raise ValueError(f"eta must be non-negative, got {eta}")
    if not omega_cut > 0:
        raise ValueError(f"omega_cut must be positive, got {omega_cut}")


def check_nm_omega(func):
    """Make func raise TypeError unless the normal-mode frequency is a real
    number, such as np.float32 or np.int64, and ValueError unless it is
    non-negative, so also for NaN."""
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
        return func(self, nm_omega, *args, **kwargs)
    return wrapper


class BaseSpectralDensity(ABC):
    """Base class for calculating model spectral densities. Defines the 
    following methods:

        * J(omega): spectral density
        * Lambda(omega): spectrum, Lambda = J/omega
        * K(t): memory-friction kernel
        
    """

    @abstractmethod
    def J(self, omega):
        raise NotImplementedError
    
    @abstractmethod
    def Lambda(self, omega):
        raise NotImplementedError
    
    @abstractmethod
    def K(self, t):
        raise NotImplementedError
        
    @check_nm_omega
    def nmJ(self, nm_omega, omega):
        if nm_omega == 0:
            return self.J(omega)
        else:
            w = np.abs(omega)
            diff = w - nm_omega
            diff2 = diff * (w + nm_omega) # ω² - ωn²
            mask = diff > 0
            ans = np.zeros_like(omega, dtype=float)
            ans[mask] = self.J(np.sqrt(diff2[mask]))
            return ans
        
    @check_nm_omega
    def nmLambda(self, nm_omega, omega):
        if nm_omega == 0:
            return self.Lambda(omega)
        else:
            w = np.abs(omega)
            diff = w - nm_omega
            diff2 = diff * (w + nm_omega) # ω² - ωn²
            mask = diff > 0
            ans = np.zeros_like(omega, dtype=float)
            ans[mask] = self.J(np.sqrt(diff2[mask])) / w[mask]
            return ans
        
    @check_nm_omega
    def nmK(self, nm_omega, t, **kwargs):
        from scipy.integrate import quad
        def f(w):
            w = np.abs(w)
            diff = w - nm_omega
            if isinstance(w, (int, float)):
                if diff <= 0.0:
                    ans = 0.0
                else:
                    x = np.sqrt(diff * (w + nm_omega))
                    ans = (2.0/np.pi) * self.Lambda(x) * x/w
            else:
                ans = np.zeros_like(w)
                mask = diff > 0
                x = np.sqrt(diff[mask] * (w[mask] + nm_omega))
                ans[mask] = (2.0/np.pi) * self.Lambda(x) * x/w[mask]
            return ans

        def K_scalar(ti):
            # Special case t = 0: use non-oscillatory integral 
            if ti == 0.0:
                val, err = quad(f, nm_omega, np.inf, **kwargs)
            else:
                val, err = quad(f, nm_omega, np.inf,
                                weight='cos', wvar=ti,
                                **kwargs)
            return val

        t_arr = np.atleast_1d(t).astype(float)
        out = np.empty_like(t_arr)

        for i, ti in enumerate(t_arr):
            out[i] = K_scalar(ti)

        return out[0] if np.isscalar(t) else out
            
