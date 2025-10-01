#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   expohmic.py
@Time    :   2024/10/02 11:21:35
@Author  :   George Trenins
@Desc    :   Exponentially damped Ohmic spectral density
'''


from __future__ import print_function, division, absolute_import
from ._base import BaseSpectralDensity
import numpy as np
from typing import Union


class ExpOhmic(BaseSpectralDensity):

    def __init__(self, 
                 eta: float, 
                 omega_cut: float, 
                 *args, **kwargs) -> None:
        """Exponentially damped Ohmic spectral density

        J(ω) = η * ω * exp(-ω/ωc)
        K(t) = 2 * η * ωc / π * (1 + ωc²t²) 

        Args:
            eta (float): static friction
            omega_cut (float): cut-off frequency
        """
        self.eta = eta
        self.omega_cut = omega_cut

    def J(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Spectral density at frequency omega.
        """
        return self.eta * np.abs(omega) * np.exp(-np.abs(omega)/self.omega_cut)
    
    def Lambda(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Spectral density at frequency omega, divided by omega.
        """
        return self.eta * np.exp(-np.abs(omega)/self.omega_cut)
    
    def K(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Memory-friction kernel at time t. This tends to
        2η * δ(t) as ω_c -> Inf.
        """
        wc = self.omega_cut
        return 2 * self.eta * wc / (1 + (wc*t)**2) / np.pi
    