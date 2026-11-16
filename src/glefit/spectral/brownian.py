#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   brownian.py
@Time    :   2026/11/16 16:07:48
@Author  :   George Trenins
@Desc    :   Brownian oscillator spectral density
'''


from __future__ import print_function, division, absolute_import
from ._base import BaseSpectralDensity, check_parameters
import numpy as np
from typing import Union


class BrownianOscillator(BaseSpectralDensity):

    def __init__(self, eta: float, omega_0: float, gamma: float) -> None:
        """Brownian oscillator spectral density, with

        Λ(ω) = η ω₀⁴ / ((ω² - ω₀²)² + γ²ω²)
        K(t) = (η ω₀²/γ) exp(-γ|t|/2) [cos(ω̄t) + (γ/2ω̄) sin(ω̄|t|)]

        where ω̄ = √(ω₀² - γ²/4). For γ > 2ω₀, ω̄ is imaginary, and cos and
        sin become cosh and sinh.

        Args:
            eta (float): static friction
            omega_0 (float): undamped frequency of the oscillator
            gamma (float): damping rate of the oscillator

        Raises:
            ValueError: unless eta ≥ 0, omega_0 > 0 and gamma > 0
        """
        check_parameters(eta, omega_0=omega_0, gamma=gamma)
        self.eta = eta
        self.omega_0 = omega_0
        self.gamma = gamma

    def Lambda(
            self, omega: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Spectrum, the cosine transform of K, at frequency omega.
        """
        w0 = self.omega_0
        return self.eta * w0**4 / (
            (omega**2 - w0**2)**2 + (self.gamma*omega)**2)

    def K(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Memory-friction kernel at time t.

        Notes:
            With s = |t|, ω̄² = ω₀² - γ²/4 and sinc(x) = sin(x)/x, 
                K / (η ω₀²/γ) = exp(-γs/2) [cos(ω̄s) + (γs/2) sinc(ω̄s)] 
            for ω̄² ≥ 0. For ω̄² < 0, with ω̃ = √(-ω̄²) and the rates λ± = γ/2 ± ω̃, 
                K / (η ω₀²/γ) = exp(-λ₋s) [1 + λ₋ (1 - exp(-2ω̃s))/(2ω̃)],
            λ₋ is computed as ω₀²/λ₊, since γ/2 - ω̃ cancels for γ ≫ ω₀, 
            and 1 - exp(-2ω̃s) with expm1, since it is small just above critical damping.
        """
        w0, g = self.omega_0, self.gamma
        s = np.abs(t)
        scale = self.eta * w0**2 / g
        # ω̄² as a product, which is accurate near critical damping
        wb2 = (w0 - g/2) * (w0 + g/2)
        if wb2 >= 0:
            wb = np.sqrt(wb2)
            # np.sinc(x) is sin(πx)/(πx)
            return scale * np.exp(-g*s/2) * (
                np.cos(wb*s) + g/2 * s * np.sinc(wb*s/np.pi))
        wt = np.sqrt(-wb2)
        lm = w0**2 / (g/2 + wt)
        return scale * np.exp(-lm*s) * (
            1 - lm * np.expm1(-2*wt*s) / (2*wt))
