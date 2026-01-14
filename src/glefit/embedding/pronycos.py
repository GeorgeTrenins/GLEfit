#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   pronycos.py
@Time    :   2025/10/08 10:17:53
@Author  :   George Trenins
@Desc    :   "Oscillatory" Prony embedding, e.g. 10.1103/PhysRevB.89.134303
'''


from __future__ import print_function, division, absolute_import
from ._base import BaseEmbedder, ScalarArr
from glefit.mappers import LowerBoundMapper
import numpy as np
import numpy.typing as npt


class PronyCosineEmbedder(BaseEmbedder):

    def __len__(self) -> int:
        return 2
    
    def _get_nparam(self) -> int:
        """Number of independent parameters used to define the drift matrix.
        """
        return 3

    def __init__(self, theta: float, gamma: float, omega: float, *args, **kwargs) -> None:
        """
        Initialize the oscillatory Prony Markovian embedding, 
            K(t) = θ^2 exp(-γt) cos(ωt). 

        Parameters
        ----------
        θ : float
            Coupling strength (units of reciprocal time). Must be positive.
        γ : float
            Relaxation rate (units of reciprocal time). Must be positive.
        ω : float
            Harmonic oscillation rate (units of reciprocal time). Must be positive.

        **kwargs
        --------
        mappers : Tuple[BaseMapper, BaseMapper]
            constraint mappers, by default imposes θ,γ,ω > 0

        Raises
        ------
        ValueError
            If input arguments are not positive float values.
        TypeError
            If input are not float values.
        """

        # Check types
        for variable, symbol in zip([theta, gamma, omega], "θγω"):
            if not isinstance(variable, (int, float)):
                raise TypeError(f"{symbol} must be a float, got {type(variable).__name__}")
        kwargs.setdefault(
            "mappers", 3*[LowerBoundMapper(),]
        )
        super().__init__(*args, **kwargs)
        self.conventional_params = np.asarray([theta, gamma, omega], dtype=float)

    @classmethod
    def from_dict(
        cls, 
        kwargs: dict
    ) -> "PronyCosineEmbedder":
        theta = kwargs.pop("theta")
        gamma = kwargs.pop("gamma")
        omega = kwargs.pop("omega")
        return cls(theta, gamma, omega, **kwargs)

    def compute_drift_matrix(
        self, 
        primitive_params: npt.NDArray
    ) -> npt.NDArray[np.floating]:
        theta, gamma, omega = np.asarray(primitive_params)
        A = np.zeros((3,3))
        A[0,1] = theta
        A[1,0] = -theta
        A[1,1] = A[2,2] = gamma
        A[1,2] = omega
        A[2,1] = -omega
        return A
    
    def drift_matrix_param_grad(
        self, 
        primitive_params: npt.NDArray
    ) -> npt.NDArray[np.floating]:
        """The gradient of the drift matrix of the extended Markovian system.
        
        Returns
        -------
        Derivative of the drift matrix
           [ 0   θ   0]
           [-θ   γ   ω]
           [ 0  -ω   γ]
        with respect to θ, γ, and ω (grad[0], grad[1], grad[2])
        """
        grad_A = np.zeros((3,3,3))
        # d/dθ 
        grad_A[0,0,1] = 1.0
        grad_A[0,1,0] = -1.0
        # d/dγ 
        grad_A[1,1,1] = grad_A[1,2,2] = 1.0
        # d/dω 
        grad_A[2,1,2] = 1.0
        grad_A[2,2,1] = -1.0
        return grad_A
    
    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        t = np.abs(np.atleast_1d(time))
        theta, gamma, omega = self.primitive_params
        return theta**2 * np.exp(-gamma * t) * np.cos(omega*t)
    
    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        t = np.abs(np.atleast_1d(time))
        theta, gamma, omega = self.primitive_params
        exp_gamma_t = np.exp(-gamma*t)
        cos_omega_t = np.cos(omega*t)
        ans = np.empty((3, len(t)))
        ans[0] = 2 * theta * exp_gamma_t * cos_omega_t
        ans[1] = -(theta**2) * t * exp_gamma_t * cos_omega_t
        ans[2] = -(theta**2) * t * exp_gamma_t * np.sin(omega*t)
        return ans
    
    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        t = np.abs(np.atleast_1d(time))
        theta, gamma, omega = self.primitive_params
        exp_gamma_t = np.exp(-gamma*t)
        cos_omega_t = np.cos(omega*t)
        sin_omega_t = np.sin(omega*t)
        ans = np.empty((3, 3, len(t)))
        ans[0,0] = 2 * exp_gamma_t * cos_omega_t
        ans[0,1] = ans[1,0] = -2 * theta * t * exp_gamma_t * cos_omega_t
        ans[0,2] = ans[2,0] = -2 * theta * t * exp_gamma_t * sin_omega_t
        ans[1,1] = theta**2 * t**2 * exp_gamma_t * cos_omega_t
        ans[1,2] = ans[2,1] = theta**2 * t**2 * exp_gamma_t * sin_omega_t
        ans[2,2] = -(theta**2) * t**2 * exp_gamma_t * cos_omega_t
        return ans
    
    def spectrum_func(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        freq = np.abs(np.atleast_1d(frequency))
        theta, gamma, omega = self.primitive_params
        g2 = gamma**2
        gth2 = gamma * theta**2
        ans = gth2 / (g2 + (freq - omega)**2)
        ans += gth2 / (g2 + (freq + omega)**2)
        ans /= 2
        return ans
        
    def spectrum_grad(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        freq = np.abs(np.atleast_1d(frequency))
        theta, gamma, omega = self.primitive_params
        g2 = gamma**2
        fdiff = freq - omega
        fdiff2 = fdiff**2
        fsum = freq + omega
        fsum2 = fsum**2
        ans = np.empty((3, len(freq)))
        ans[0] = gamma * theta * ( 1/(g2 + fdiff2) + 1/(g2 + fsum2) )
        ans[1] = -(theta**2 / 2) * (
            (g2 - fdiff2) / (g2 + fdiff2)**2 + 
            (g2 - fsum2) / (g2 + fsum2)**2 
        )
        ans[2] = gamma * theta**2 * (
            fdiff/(g2 + fdiff2)**2 - 
            fsum/(g2 + fsum2)**2
        )
        return ans
    
    def spectrum_hess(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        freq = np.abs(np.atleast_1d(frequency))
        theta, gamma, omega = self.primitive_params
        g2 = gamma**2
        fdiff = freq - omega
        fdiff2 = fdiff**2
        fsum = freq + omega
        fsum2 = fsum**2
        ans = np.empty((3,3,len(freq)))
        ans[0,0] = gamma * ( 1/(g2 + fdiff2) + 1/(g2 + fsum2) )
        ans[0,1] = ans[1,0] = -theta * (
            (g2 - fdiff2) / (g2 + fdiff2)**2 + 
            (g2 - fsum2) / (g2 + fsum2)**2 
        )
        ans[0,2] = ans[2,0] = 2 * gamma * theta * (
            fdiff/(g2 + fdiff2)**2 - 
            fsum/(g2 + fsum2)**2
        )
        ans[1,1] = gamma * theta**2 * (
            (g2 - 3*fdiff2) / (g2 + fdiff2)**3 + 
            (g2 - 3*fsum2) / (g2 + fsum2)**3 
        )
        ans[1,2] = ans[2,1] = theta**2 * (
            fdiff * (-3*g2 + fdiff2) / (g2 + fdiff2)**3 - 
            fsum * (-3*g2 + fsum2) / (g2 + fsum2)**3 
        )
        ans[2,2] = gamma * theta**2 * (
            (-g2 + 3*fdiff2) / (g2 + fdiff2)**3 + 
            (-g2 + 3*fsum2) / (g2 + fsum2)**3 
        )
        return ans