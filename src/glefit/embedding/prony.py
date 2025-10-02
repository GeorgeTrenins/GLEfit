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
        Initialize the Prony Markovian embedding, K(t) = θ^2 exp(-γt). Under the hood,
        the embedder imposes positivity constraints by mapping 
        θ -> exp(x1) and γ -> exp(x2)


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
        self.params = np.asarray([theta_float, gamma_float])
        
    def _forward_map(self, params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform optimizable params so that inequality constraints are automatically imposed"""
        return np.log(params)
    
    def _inverse_map(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Undo the inequality constraint-imposing mapping.
        """
        return np.exp(x)
    
    def _jac_px(self, x):
        return np.exp(x)
    
    def _hess_px(self, x):
        return np.exp(x)

    def _compute_drift_matrix(
            self, 
            params: npt.NDArray
        ) -> npt.NDArray[np.floating]:
        """The drift matrix of the extended Markovian system.
        
        Returns
        -------
        Drift matrix for the Prony embedding,
           [ 0   θ ]
           [ θ   γ ]
        """
        theta, gamma = np.asarray(params)
        self._A[0,0] = 0.0
        self._A[0,1] = theta
        self._A[1,0] = theta
        self._A[1,1] = gamma
        return np.copy(self._A)
    
    def _drift_matrix_param_grad(
            self, 
            params: npt.NDArray
        ) -> npt.NDArray[np.floating]:
        """The gradient of the drift matrix of the extended Markovian system.
        
        Returns
        -------
        Derivative of the drift matrix
           [ 0   θ ]
           [ θ   γ ]
        with respect to θ (grad[0]):
           [ 0   1 ]
           [ 1   0 ]
        and γ (grad[1]):
           [ 0   0 ]
           [ 0   1 ]
        """
        self._grad_A[:] = 0.0
        # Derivative with respect to θ
        self._grad_A[0,0,1] = 1.0
        self._grad_A[0,1,0] = 1.0
        # Derivative with respect to γ
        self._grad_A[1,1,1] = 1.0
        return np.copy(self._grad_A)
    
    def _kernel(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Memory kernel function for the Prony embedding, K(t) = θ^2 exp(-γt)
        """
        theta, gamma = self.params
        return theta**2 * np.exp(-gamma * time)
    
    def _kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Gradient of the memory kernel function for the Prony embedding, K(t) = θ^2 exp(-γt),
        w.r.t. [θ, γ]
        """
        theta, gamma = self.params
        exp_gamma_t = np.exp(-gamma*time)
        ans = np.empty((2, len(time)))
        # First derivative term: d/dθ [θ^2 exp(-γt)]
        ans[0] = 2 * theta * exp_gamma_t
        # Second derivative term: d/dγ [θ^2 exp(-γt)]
        ans[1] = -(theta**2) * time * exp_gamma_t
        return ans
    
    def _kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Hessian of the memory kernel function for the Prony embedding, K(t) = θ^2 exp(-γt),
        w.r.t. [θ, γ]
        """
        theta, gamma = self.params
        exp_gamma_t = np.exp(-gamma*time)
        ans = np.empty((2, 2, len(time)))
        # d²/dθ² [θ^2 exp(-γt)]
        ans[0,0] = 2*exp_gamma_t
        # d²/dθdγ [θ^2 exp(-γt)]
        ans[0,1] = ans[1,0] = -2 * theta * time * exp_gamma_t
        # d²/dγ² [θ^2 exp(-γt)]
        ans[1,1] = theta**2 * time**2 * exp_gamma_t
        return ans
    
    def _spectrum(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """
        Spectrum for the Prony embedding, γ * θ^2 / (γ^2 + ω^2)
        """
        theta, gamma = self.params
        return gamma * theta**2 / (gamma**2 + frequency**2)
        
    def _spectrum_grad(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """
        Gradient of the spectrum for the Prony embedding, γ * θ^2 / (γ^2 + ω^2),
        w.r.t. [θ, γ]
        """
        theta, gamma = self.params
        ans = np.empty((2, len(frequency)))
        # First derivative term: d/dθ [γ θ^2 / (γ^2 + ω^2)] = 2 θ γ / (γ^2 + ω^2)
        ans[0] = 2 * theta * gamma / (gamma**2 + frequency**2)
        # Second derivative term: 
        # d/dγ [γ * θ^2 / (γ^2 + ω^2)] = θ^2 * (ω^2 - γ^2) / (γ^2 + ω^2)^2
        ans[1] = theta**2 * (frequency**2 - gamma**2) / (gamma**2 + frequency**2)**2
        return ans
    
    def _spectrum_hess(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """
        Hessian of the spectrum for the Prony embedding, γ * θ^2 / (γ^2 + ω^2),
        w.r.t. [θ, γ]
        """
        theta, gamma = self.params
        g2 = gamma**2
        w2 = frequency**2
        gw2 = g2 + w2
        ans = np.empty((2, 2, len(frequency)))
        # d²/dθ² [θ^2 exp(-γt)]
        ans[0,0] = 2*gamma / gw2
        # d²/dθdγ [θ^2 exp(-γt)]
        ans[0,1] = ans[1,0] = 2*theta * (w2 - g2) / gw2**2
        # d²/dγ² [θ^2 exp(-γt)]
        ans[1,1] = 2*gamma*theta**2 * (g2 - 3*w2) / gw2**3
        return ans