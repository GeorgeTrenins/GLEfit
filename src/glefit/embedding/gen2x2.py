#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   gen2x2.py
@Time    :   2025/10/15 10:27:11
@Author  :   George Trenins
@Desc    :   "General" two-variable embedder
'''


from __future__ import print_function, division, absolute_import
from ._base import BaseEmbedder, ScalarArr
from glefit.mappers import LowerBoundMapper, IdentityMapper
from glefit.utils.special import coscosh, sincsinhc
from collections import namedtuple
import numpy as np
import numpy.typing as npt

_cparams = namedtuple('CParams', ['theta1', 'theta2', 'gamma', 'delta', 'Omega'])
_params = namedtuple('Params', ['r', 'zeta', 'lamda', 'Gamma'])


class TwoAuxEmbedder(BaseEmbedder):
    _naux = 2

    def __len__(self) -> int:
        return self._naux
    
    def _get_nparam(self) -> int:
        """Number of independent parameters used to define the drift matrix.
        """
        return 4
    
    def __init__(self, theta: npt.NDArray[np.floating], gamma: float, delta: float, Omega: float, *args, **kwargs) -> None:
        """Initialize the two-variable Markovian embedder with the drift matrix

            (  0    θ[0] θ[1])
        A = (-θ[0]  γ+δ    Ω )
            (-θ[1]  -Ω   γ-δ )

        with the corresponding kernel
            K(t) = exp(-γ·t) [
                (θ[0]²+θ[1]²) cosh(t·sqrt(δ²-Ω²)) - 
                (θ[0]²-θ[1]²) sinh(t·sqrt(δ²-Ω²)) · δ/sqrt(δ²-Ω²)
            ]
        which, under the hood, is converted to the four-parameter function imposing the Lyapunov stability condition,
            K(t) = r² exp(-λt) exp(-|max(Γ,|ζ|)|·t) [ C(Γ,t) - ζ|t|·S(Γ,t) ]
        where
            C(Γ,t) = cosh(Γt) if Γ > 0 else cos(Γt)
        and
            S(Γ,t) = sinch(Γt) if Γ > 0 else sinc(Γt)
        with
            r, λ > 0 and Γ, ζ ∈ R.

        Parameters
        ----------
            theta : np.ndarray, shape(2,)
            gamma  : float
            delta : float
            Omega : float

        Raises
        ------
        TypeError
            If input are not float values.
        ValueError
            If input arguments fall outside their respective domains.
        """

        # Check types
        for variable, symbol in zip([gamma, Omega, delta], "γδΩ"):
            if not isinstance(variable, (int, float)):
                raise TypeError(f"{symbol} must be a float, got {type(variable).__name__}")
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
            raise ValueError(f"θ must be a length-2 vector, instead got {theta.shape = }")
        
        self._cparams = np.concatenate([theta, [gamma, delta, Omega]])
        self._named_cparams = _cparams(*self._cparams)
        params = self.from_conventional(self._cparams)
        kwargs.setdefault(
            "mappers", [
                LowerBoundMapper(), # r > 0
                IdentityMapper(),   # ζ ∈ R
                LowerBoundMapper(), # λ > 0
                IdentityMapper()    # Γ ∈ R
        ])
        super().__init__(*args, **kwargs)
        self.params = params
        self._named_params = _params(*self.params)

    @staticmethod
    def from_conventional(
        cparams: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Convert the 'conventional' embedding parameters to the reduced parameter set that removes the embedding degenracy

        Args:
            cparams : np.ndarray, shape(5,)
                conventional parameters in the order [ θ[0], θ[1], γ, δ, Ω ]

        Returns:
            params : np.ndarray, shape(4,)
                non-degenerate parameters [ r, ζ, λ, Γ ]
        """
        cparams = np.asarray(cparams, dtype=float)
        if cparams.shape != (5,):
           raise ValueError(f"The conventional parameters must be supplied as a length-5 vector, instead got {cparams.shape = }")
        th2 = cparams[:2]**2
        gamma, delta, Omega = cparams[2:]
        r = np.sqrt(np.sum(th2))
        d2O2 = delta**2 - Omega**2
        Gamma = np.sign(d2O2) * np.sqrt(np.abs(d2O2))
        zeta = delta * (th2[0] - th2[1]) / (th2[0] + th2[1])
        lamda = gamma - max(Gamma, abs(zeta))
        params = np.array([r, zeta, lamda, Gamma], dtype=float)
        return params
    
    @staticmethod
    def to_conventional(
        params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Convert non-degenerate embedding parameters to conventional embedding variables
        Args:
            cparams : np.ndarray, shape(4,)
                non-degenerate parameters in the order [ r, ζ, λ, Γ ]

        Returns:
            params : np.ndarray, shape(5,)
                conventional parameters [ θ[0], θ[1], γ, δ, Ω ]
        """
        params = np.asarray(params, dtype=float)
        if params.shape != (4,):
           raise ValueError(f"The non-degenerate parameters must be supplied as a length-4 vector, instead got {params.shape = }")
        r, zeta, lamda, Gamma = params
        if Gamma > 0:
            if abs(zeta) < Gamma:
                alpha = np.abs(zeta) / Gamma
                delta = np.sign(zeta)*Gamma
                Omega = 0.0
                theta1 = r * np.sqrt((1 + alpha)/2)
                theta2 = r * np.sqrt((1 - alpha)/2)
            else:
                delta = zeta
                Omega = np.sqrt(zeta**2 - Gamma**2)
                theta1 = r
                theta2 = 0.0
        else:
            delta = zeta
            Omega = np.sqrt(zeta**2 + Gamma**2)
            theta1 = r
            theta2 = 0.0
        gamma = lamda + max(Gamma, abs(zeta))
        return np.asarray([theta1, theta2, gamma, delta, Omega])
    
    @staticmethod
    def rotate_matrix(
        theta: npt.NDArray[np.floating],
        A: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Rotate the drift matrix A and coupling vector theta into a basis where
        the symmetric component of A is diagonal
        """
        Asym = (A + A.T)/2
        eigs = np.linalg.eigh(Asym)
        if np.any(eigs.eigenvalues < 0):
            raise ValueError(f"The auxiliary variable drift matrix must be positive-definite. Instead, the symmetric component {Asym = } was found to have the eigenvalues {eigs.eigenvalues}.")
        new_theta = np.dot(eigs.eigenvectors.T, theta)
        new_A = np.linalg.multi_dot([eigs.eigenvectors.T, A, eigs.eigenvectors])
        return new_theta, new_A

    @classmethod
    def from_matrix(
        cls, 
        theta: npt.NDArray[np.floating],
        A: npt.NDArray[np.floating]
    ) -> "TwoAuxEmbedder":
        """Construct the two-variable embedder from a 2x2 drift matrix A and a coupling vector θ.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
           raise ValueError(f"θ must be a length-2 vector, instead got {theta.shape = }")
        A = np.asarray(A, dtype=float)
        if A.shape != (2,2):
           raise ValueError(f"A must be a 2-by-2 array, instead got {A.shape = }") 
        theta, A = cls.rotate_matrix(theta, A)
        gamma = (A[0,0] + A[1,1]) / 2
        delta = (A[0,0] - A[1,1]) / 2
        Omega = A[0,1]
        return cls(theta, gamma, delta, Omega)

    def compute_drift_matrix(
            self, 
            params: npt.NDArray
        ) -> npt.NDArray[np.floating]:
        """The drift matrix of the extended Markovian system.
        
        Returns
        -------
        Drift matrix for the two-variable embedding
            (  0    θ[0] θ[1])
        A = (-θ[0]  γ+δ    Ω )
            (-θ[1]  -Ω   γ-δ )
        """
        theta1, theta2, gamma, delta, Omega = self.to_conventional(params)
        A = np.zeros((3,3))
        A[0,1] = theta1
        A[0,2] = theta2
        A[1,0] = -theta1
        A[2,0] = -theta2
        A[1,1] = gamma + delta
        A[2,2] = gamma - delta
        A[1,2] = Omega
        A[2,1] = -Omega
        return A
    
    def drift_matrix_param_grad(
            self, 
            params: npt.NDArray
        ) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Memory kernel function for the two-variable embedding, 
            K(t) = r² exp(-λt) exp(-|max(Γ,|ζ|)|·t) [ C(Γ,t) - ζ|t|·S(Γ,t) ]
        """
        t = np.abs(time)
        r, zeta, lamda, Gamma = self.params
        gamma = lamda + max(Gamma, abs(zeta))
        Gt = Gamma*t
        ans = coscosh(Gt) - zeta*t*sincsinhc(Gt)
        ans *= np.exp(-gamma*t)
        return r**2 * ans
    
    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        raise NotImplementedError
    
    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        raise NotImplementedError
    
    def spectrum_func(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        raise NotImplementedError
    
    def spectrum_grad(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        raise NotImplementedError
    
    def spectrum_hess(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        raise NotImplementedError