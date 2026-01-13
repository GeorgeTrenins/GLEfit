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

    def __len__(self) -> int:
        return 1
    
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

    @classmethod
    def from_dict(
        cls, 
        parameters: dict
    ) -> "PronyEmbedder":
        theta = parameters.pop("theta")
        gamma = parameters.pop("gamma")
        return cls(theta, gamma, **parameters)

    def compute_drift_matrix(
            self, 
            params: npt.NDArray
        ) -> npt.NDArray[np.floating]:
        theta, gamma = np.asarray(params)
        A = np.zeros((2,2))
        A[0,1] =  theta
        A[1,0] = -theta
        A[1,1] =  gamma
        return A
    
    def drift_matrix_param_grad(
            self, 
            params: npt.NDArray
        ) -> npt.NDArray[np.floating]:
        """Gradient of drift matrix w.r.t. [θ, γ].
        
        Returns array of shape (2, 2, 2) where:
        - grad[0] = dA/dθ
        - grad[1] = dA/dγ
        """
        grad_A = np.zeros((2, 2, 2))
        # dA/dθ
        grad_A[0, 0, 1] = 1.0
        grad_A[0, 1, 0] = -1.0
        # dA/dγ
        grad_A[1, 1, 1] = 1.0
        return grad_A
    
    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """K(t)=θ² exp(-γ|t|)."""
        t = np.abs(np.atleast_1d(time))
        theta, gamma = self.primitive_params
        return theta**2 * np.exp(-gamma * t)
    
    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        t = np.abs(np.atleast_1d(time))
        theta, gamma = self.primitive_params
        exp_gamma_t = np.exp(-gamma*t)
        ans = np.empty((2, len(t)))
        ans[0] = 2 * theta * exp_gamma_t
        ans[1] = -(theta**2) * t * exp_gamma_t
        return ans
    
    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        t = np.abs(np.atleast_1d(time))
        theta, gamma = self.primitive_params
        exp_gamma_t = np.exp(-gamma*t)
        ans = np.empty((2, 2, len(t)))
        ans[0,0] = 2*exp_gamma_t
        ans[0,1] = ans[1,0] = -2 * theta * t * exp_gamma_t
        ans[1,1] = theta**2 * t**2 * exp_gamma_t
        return ans
    
    def spectrum_func(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        freq = np.abs(np.atleast_1d(frequency))
        theta, gamma = self.primitive_params
        return gamma * theta**2 / (gamma**2 + freq**2)
        
    def spectrum_grad(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        freq = np.abs(np.atleast_1d(frequency))
        theta, gamma = self.primitive_params
        ans = np.empty((2, len(freq)))
        ans[0] = 2 * theta * gamma / (gamma**2 + freq**2)
        ans[1] = theta**2 * (freq**2 - gamma**2) / (gamma**2 + freq**2)**2
        return ans
    
    def spectrum_hess(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        freq = np.abs(np.atleast_1d(frequency))
        theta, gamma = self.primitive_params
        g2 = gamma**2
        w2 = freq**2
        gw2 = g2 + w2
        ans = np.empty((2, 2, len(freq)))
        ans[0,0] = 2*gamma / gw2
        ans[0,1] = ans[1,0] = 2*theta * (w2 - g2) / gw2**2
        ans[1,1] = 2*gamma*theta**2 * (g2 - 3*w2) / gw2**3
        return ans