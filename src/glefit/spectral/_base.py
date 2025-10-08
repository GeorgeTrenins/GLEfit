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
import numpy as np

def check_nm_omega(func):
    @wraps(func)
    def wrapper(self, nm_omega: float, *args, **kwargs):
        if not isinstance(nm_omega, (int, float)):
            raise TypeError(
                f"Expecting the normal-mode frequency to be a scalar, "
                f"instead got `{type(nm_omega).__name__}`"
            )
        if nm_omega < 0:
            raise ValueError(
                f"Expecting the normal-mode frequency to be positive, instead got {nm_omega}"
            )
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
            ans = np.zeros_like(omega)
            ans[mask] = self.J(np.sqrt(diff2[mask]))
        
    @check_nm_omega
    def nmLambda(self, nm_omega, omega):
        if nm_omega == 0:
            return self.Lambda(omega)
        else:
            w = np.abs(omega)
            diff = w - nm_omega
            diff2 = diff * (w + nm_omega) # ω² - ωn²
            mask = diff > 0
            ans = np.zeros_like(omega)
            ans[mask] = self.J(np.sqrt(diff2[mask])) / w[mask]
        
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
            