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
from glefit.mappers import LowerBoundMapper
import numpy as np
import numpy.typing as npt


class PronyEmbedder(BaseEmbedder):
    _naux = 1

    def __len__(self) -> int:
        return self._naux
    
    def _get_nparam(self) -> int:
        """Number of independent parameters used to define the drift matrix.
        """
        return 2

    def __init__(self, theta: float, gamma: float, *args, **kwargs) -> None:
        """
        Initialize the Prony Markovian embedding, K(t) = θ^2 exp(-γt). 

        Parameters
        ----------
        θ : float
            Coupling strength (units of reciprocal time). Must be positive.
        γ : float
            Relaxation rate (units of reciprocal time). Must be positive.

        **kwargs
        --------
        mappers : Tuple[BaseMapper, BaseMapper]
            constraint mappers, by default imposes θ,γ > 0

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
        kwargs.setdefault("mappers", [LowerBoundMapper(), LowerBoundMapper()])
        super().__init__(*args, **kwargs)
        self.params = np.asarray([theta, gamma], dtype=float)

    def compute_drift_matrix(
            self, 
            params: npt.NDArray
        ) -> npt.NDArray[np.floating]:
        """The drift matrix of the extended Markovian system.
        
        Returns
        -------
        Drift matrix for the Prony embedding,
           [ 0   θ ]
           [-θ   γ ]
        """
        theta, gamma = np.asarray(params)
        A = np.zeros((2,2))
        A[0,0] =  0.0
        A[0,1] =  theta
        A[1,0] = -theta
        A[1,1] =  gamma
        return A
    
    def drift_matrix_param_grad(
            self, 
            params: npt.NDArray
        ) -> npt.NDArray[np.floating]:
        """The gradient of the drift matrix of the extended Markovian system.
        
        Returns
        -------
        Derivative of the drift matrix
           [ 0   θ ]
           [-θ   γ ]
        with respect to θ (grad[0]):
           [ 0   1 ]
           [-1   0 ]
        and γ (grad[1]):
           [ 0   0 ]
           [ 0   1 ]
        """
        grad_A = np.zeros((2,2,2))
        # Derivative with respect to θ
        grad_A[0,0,1] = 1.0
        grad_A[0,1,0] =-1.0
        # Derivative with respect to γ
        grad_A[1,1,1] = 1.0
        return grad_A
    
    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Memory kernel function for the Prony embedding, K(t) = θ^2 exp(-γt)
        """
        theta, gamma = self.params
        return theta**2 * np.exp(-gamma * time)
    
    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Gradient of the memory kernel function for the Prony embedding, K(t) = θ^2 exp(-γt),
        w.r.t. [θ, γ]
        """
        theta, gamma = self.params
        exp_gamma_t = np.exp(-gamma*time)
        ans = np.empty((2, len(time)))
        # d/dθ
        ans[0] = 2 * theta * exp_gamma_t
        # d/dγ
        ans[1] = -(theta**2) * time * exp_gamma_t
        return ans
    
    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Hessian of the memory kernel function for the Prony embedding, K(t) = θ^2 exp(-γt),
        w.r.t. [θ, γ]
        """
        theta, gamma = self.params
        exp_gamma_t = np.exp(-gamma*time)
        ans = np.empty((2, 2, len(time)))
        # d²/dθ² 
        ans[0,0] = 2*exp_gamma_t
        # d²/dθdγ
        ans[0,1] = ans[1,0] = -2 * theta * time * exp_gamma_t
        # d²/dγ² 
        ans[1,1] = theta**2 * time**2 * exp_gamma_t
        return ans
    
    def spectrum_func(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """
        Spectrum for the Prony embedding, γ * θ^2 / (γ^2 + ω^2)
        """
        theta, gamma = self.params
        return gamma * theta**2 / (gamma**2 + frequency**2)
        
    def spectrum_grad(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """
        Gradient of the spectrum for the Prony embedding, γ * θ^2 / (γ^2 + ω^2),
        w.r.t. [θ, γ]
        """
        theta, gamma = self.params
        ans = np.empty((2, len(frequency)))
        # d/dθ [γ θ^2 / (γ^2 + ω^2)] = 2 θ γ / (γ^2 + ω^2)
        ans[0] = 2 * theta * gamma / (gamma**2 + frequency**2)
        # d/dγ [γ * θ^2 / (γ^2 + ω^2)] = θ^2 * (ω^2 - γ^2) / (γ^2 + ω^2)^2
        ans[1] = theta**2 * (frequency**2 - gamma**2) / (gamma**2 + frequency**2)**2
        return ans
    
    def spectrum_hess(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """
        Hessian of the spectrum for the Prony embedding, γ * θ^2 / (γ^2 + ω^2),
        w.r.t. [θ, γ]
        """
        theta, gamma = self.params
        g2 = gamma**2
        w2 = frequency**2
        gw2 = g2 + w2
        ans = np.empty((2, 2, len(frequency)))
        # d²/dθ² 
        ans[0,0] = 2*gamma / gw2
        # d²/dθdγ 
        ans[0,1] = ans[1,0] = 2*theta * (w2 - g2) / gw2**2
        # d²/dγ²
        ans[1,1] = 2*gamma*theta**2 * (g2 - 3*w2) / gw2**3
        return ans