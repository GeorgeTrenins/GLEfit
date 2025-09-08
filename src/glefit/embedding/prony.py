#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   prony.py
@Time    :   2025/09/08 11:00:54
@Author  :   George Trenins
@Desc    :   Prony series embedding 
'''


from __future__ import print_function, division, absolute_import
from ._base import BaseEmbedder, ScalarArr
from typing import Optional
import numpy as np
import numpy.typing as npt


class PronyEmbedder(BaseEmbedder):
    _naux = 1

    def __len__(self) -> int:
        return self._naux

    def __init__(self, theta: float, gamma: float, *args, **kwargs) -> None:
        """
        Initialize the Prony Markovian embedding, K(t) = θ^2 exp(-γt).

        Parameters
        ----------
        θ : float
            Coupling strength (units of reciprocal time). Must be positive.
        γ : float
            Relaxation rate (units of reciprocal time). Must be positive.

        Raises
        ------
        ValueError
            If θ or γ are not positive float values.
        TypeError
            If θ or γ are not float values.
        """
        # Check types
        if not isinstance(theta, (int, float)):
            raise TypeError(f"θ must be a float, got {type(theta).__name__}")
        if not isinstance(gamma, (int, float)):
            raise TypeError(f"γ must be a float, got {type(gamma).__name__}")
        
        # Convert to float and check positivity
        gamma_float = float(gamma)
        theta_float = float(theta)
        
        if theta_float <= 0:
            raise ValueError(f"θ must be positive, got {theta_float}")
        if gamma_float <= 0:
            raise ValueError(f"γ must be positive, got {gamma_float}")

        super().__init__(*args, **kwargs)
        self._gamma = gamma_float
        self._theta = theta_float
    
    @property
    def drift_matrix(self) -> npt.NDArray[np.floating]:
        """The drift matrix of the extended Markovian system.
        
        Returns
        -------
        Drift matrix for the Prony embedding,
           [ 0   w ]
           [ w   γ ]
            
        Notes
        -----
        This matrix is also accessible through the alias 'A'.
        """
        self._A[0,0] = 0.0
        self._A[0,1] = self._theta
        self._A[1,0] = self._theta
        self._A[1,1] = self._gamma
        return np.copy(self._A)
    
    def kernel(self, time: ScalarArr, nu: Optional[int]=0) -> npt.NDArray[np.floating]:
        """
        Memory kernel function for the Prony embedding, K(t) = θ^2 exp(-γt)

        Parameters
        ----------
        time : scalar or array-like
            The input time values for which to compute the kernel.
        nu : int, optional
            Order of the derivative with respect to embedding parameters.
            Only accepts 0 or 1.

        Returns
        -------
        np.ndarray
            For nu=0, returns the kernel value(s) as a float or array of the same shape as `time`
            For nu=1, returns a 2 x N array, where N is the number of time points:
            - ans[0] contains the derivative with respect to θ,
            - ans[1] contains the derivative with respect to γ.

        Raises
        ------
        ValueError
            If `nu` is not 0 or 1, or if `time` is not scalar or 1D array for nu=1.
        """

        super().kernel(time, nu=nu)
        if nu == 0:
            return self._theta**2 * np.exp(-self._gamma * np.abs(time))
        elif nu == 1:
            time = np.abs(np.atleast_1d(time))
            if time.ndim != 1:
                raise ValueError(f"Expecting `time` to be scalar or a 1D array, "
                                 f"instead got {time.ndim = }.")
            exp_gamma_t = np.exp(-self._gamma*time)
            ans = np.empty((2, len(time)))
            # First derivative term: d/dθ [θ^2 exp(-γt)]
            ans[0] = 2 * self._theta * exp_gamma_t
            # Second derivative term: d/dγ [θ^2 exp(-γt)]
            ans[1] = -(self._theta**2) * time * exp_gamma_t
            return ans
        else:
            raise ValueError(f"Invalid value for nu = {nu}. Valid values are 0 and 1.")
        

    def spectrum(self, frequency: ScalarArr, nu: Optional[int]=0) -> npt.NDArray[np.floating]:
        """
        Spectrum for the Prony embedding, γ * θ^2 / (γ^2 + ω^2)

        Parameters
        ----------
        frequency : scalar or array-like
            Array of frequency values at which to evaluate the spectrum.

        nu : int (optional)
            Order of the derivative with respect to embedding parameters 
            (default is 0)

        Returns
        -------
        np.ndarray
            For nu=0, returns the spectrum value(s) as a float or array of the same shape as `frequency`
            For nu=1, returns a 2 x N array, where N is the number of frequency points:
            - ans[0] contains the derivative with respect to θ,
            - ans[1] contains the derivative with respect to γ.

        Raises
        ------
        ValueError
            If `nu` is not 0 or 1, or if `time` is not scalar or 1D array for nu=1.
        """
        super().spectrum(frequency, nu=nu)
        frequency = np.asarray(frequency)
        if nu == 0:
            return self._gamma * self._theta**2 / (
                   self._gamma**2 + frequency**2)
        elif nu == 1:
            if frequency.ndim != 1:
                raise ValueError(f"Expecting `frequency` to be scalar or a 1D array, "
                                 f"instead got {frequency.ndim = }.")
            ans = np.empty((2, len(frequency)))
            # First derivative term: d/dθ [γ θ^2 / (γ^2 + ω^2)] = 2 θ γ / (γ^2 + ω^2)
            ans[0] = 2 * self._theta * self._gamma / (
                     self._gamma**2 + frequency**2)
            # Second derivative term: 
            # d/dγ [γ * θ^2 / (γ^2 + ω^2)] = θ^2 * (ω^2 - γ^2) / (γ^2 + ω^2)^2
            ans[1] = self._theta**2 * (frequency**2 - self._gamma**2) / (
                     self._gamma**2 + frequency**2)**2
            return ans
        else:
            raise ValueError(f"Invalid value for nu = {nu}. Valid values are 0 and 1.")

