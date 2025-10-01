#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   debye.py
@Time    :   2025/09/22 08:42:27
@Author  :   George Trenins
@Desc    :   Debye spectral density
'''


from __future__ import print_function, division, absolute_import
from ._base import BaseSpectralDensity
import numpy as np
from typing import Union


class Debye(BaseSpectralDensity):
    
    def __init__(self, 
                 eta: float, 
                 omega_cut: float, 
                 *args, **kwargs) -> None:
        """Initialize a Debye spectral density

        J(ω) = η * ω * ωc^2 / (ωc^2 + ω^2)
        K(t) = η * ωc * exp(-ωc*t)

        Args:
            eta (float): static friction
            omega_cut (float): cut-off frequency
        """
        self.eta = eta
        self.omega_cut = omega_cut

    def J(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Spectral density at frequency omega.
        """
        return self.eta * self.omega_cut**2 * omega / (
            omega**2 + self.omega_cut**2
        )
    
    def Lambda(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Spectral density at frequency omega, divided by omega.
        """
        return self.eta * self.omega_cut**2 / (
            omega**2 + self.omega_cut**2
        )
    
    def K(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Memory-friction kernel at time t. This tends to
        2η * δ(t) as ω_c -> Inf.
        """
        return self.eta * self.omega_cut * np.exp(-self.omega_cut*np.abs(t))
    