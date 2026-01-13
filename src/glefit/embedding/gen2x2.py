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
from glefit.utils.special import expcoscosh, expsincsinhc
from collections import namedtuple
import numpy as np
import numpy.typing as npt

_params = namedtuple('Params', ['r', 'alpha', 'lamda', 'Gamma'])
_DEFAULTS = dict(sigma=5.0, threshold=20.0)

#TODO:
# * update params setter to also update cparams


class TwoAuxEmbedder(BaseEmbedder):

    def __len__(self) -> int:
        return 2
    
    def _get_nparam(self) -> int:
        """Number of conventional parameters used to define the drift matrix.
        """
        return 5
    
    def _get_ndof(self) -> int:
        """Number of primitive (optimizable) parameters.
        """
        return 4

    @property
    def ndof(self) -> int:
        """Number of primitive (optimizable) parameters.
        """
        return self._get_ndof()
    
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
            K(t) = r² exp(-λt) exp(-max(Γ,|α|)·t) [ C(Γ,t) - αt·S(Γ,t) ]
        where
            C(Γ,t) = cosh(Γt) if Γ > 0 else cos(Γt)
        and
            S(Γ,t) = sinhc(Γt) if Γ > 0 else sinc(Γt)
        with
            r, λ > 0 and Γ, α ∈ R.

        Parameters
        ----------
            theta : np.ndarray, shape(2,)
            gamma : float
            delta : float
            Omega : float

        Raises
        ------
        TypeError
            If input are not float values.
        ValueError
            If input arguments fall outside their respective domains.
        """

        from glefit.utils.special import Softmax, Softabs
        self.softmax = Softmax(
            sigma=kwargs.get("sigma", _DEFAULTS["sigma"]),
            threshold=kwargs.get("threshold", _DEFAULTS["threshold"]),
        )
        self.softabs = Softabs(
            sigma=kwargs.get("sigma", _DEFAULTS["sigma"]),
            threshold=kwargs.get("threshold", _DEFAULTS["threshold"]),
        )
        for variable, symbol in zip([gamma, Omega, delta], "γδΩ"):
            if not isinstance(variable, (int, float)):
                raise TypeError(f"{symbol} must be a float, got {type(variable).__name__}")
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
            raise ValueError(f"θ must be a length-2 vector, instead got {theta.shape = }")
        kwargs.setdefault(
            "mappers", [
                LowerBoundMapper(), # r > 0
                IdentityMapper(),   # α ∈ R
                LowerBoundMapper(), # λ > 0
                IdentityMapper()    # Γ ∈ R
        ])
        super().__init__(*args, **kwargs)
        # set primitive params directly
        cparams = np.concatenate([theta, [gamma, delta, Omega]])
        prim_params = self.to_primitive(cparams)
        self._params[:] = prim_params
        self._x[:] = self._forward_map(prim_params)
        self._named_params = _params(*self._params)

    @classmethod
    def from_dict(
        cls, 
        parameters: dict
    ) -> "TwoAuxEmbedder":
        theta = parameters.pop("theta")
        gamma = parameters.pop("gamma")
        delta = parameters.pop("delta")
        Omega = parameters.pop("Omega")
        return cls(theta, gamma, delta, Omega, **parameters)

    def _gamma_lower_bound(self, Gamma, alpha, nu=0):
        if nu == 0:
            return self.softmax( Gamma, self.softabs(alpha, nu=0), nu=0)
        elif nu == 1:
            ans = self.softmax( Gamma, self.softabs(alpha, nu=0), nu=1)
            ans[1] *= self.softabs(alpha, nu=1)
            return ans
        elif nu == 2:
            # Second derivatives of γ = Softmax(Γ, Softabs(α))
            abs_alpha = self.softabs(alpha, nu=0)
            d_abs_alpha = self.softabs(alpha, nu=1)
            d2_abs_alpha = self.softabs(alpha, nu=2)
            # First derivatives of Softmax
            d_softmax = self.softmax(Gamma, abs_alpha, nu=1)  # shape (2,)
            # Second derivatives of Softmax
            d2_softmax = self.softmax(Gamma, abs_alpha, nu=2)  # shape (2,2)
            # Hessian: [d²γ/dΓ², d²γ/dΓdα; d²γ/dαdΓ, d²γ/dα²]
            ans = np.empty((2, 2))
            # d²γ/dΓ²
            ans[0, 0] = d2_softmax[0, 0]
            # d²γ/dΓdα = d²γ/dαdΓ (chain rule)
            ans[0, 1] = ans[1, 0] = d2_softmax[0, 1] * d_abs_alpha
            # d²γ/dα² (full chain rule)
            ans[1, 1] = d2_softmax[1, 1] * d_abs_alpha**2 + d_softmax[1] * d2_abs_alpha
            return ans
        else:
            raise ValueError(f"Expecting nu = 0, 1, or 2, instead got nu = {nu}")

    def to_primitive(
        self,
        cparams: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Convert the 'conventional' embedding parameters to the primitive parameter set that removes the embedding degenracy

        Args:
            cparams : np.ndarray, shape(5,)
                conventional parameters in the order [ θ[0], θ[1], γ, δ, Ω ]

        Returns:
            params : np.ndarray, shape(4,)
                non-degenerate parameters [ r, α, λ, Γ ]
        """
        cparams = np.asarray(cparams, dtype=float)
        if cparams.shape != (5,):
           raise ValueError(f"The conventional parameters must be supplied as a length-5 vector, instead got {cparams.shape = }")
        th2 = cparams[:2]**2
        gamma, delta, Omega = cparams[2:]
        r = np.sqrt(np.sum(th2))
        d2O2 = delta**2 - Omega**2
        Gamma = np.sign(d2O2) * np.sqrt(np.abs(d2O2))
        alpha = delta * (th2[0] - th2[1]) / (th2[0] + th2[1])
        lamda = gamma - self._gamma_lower_bound(Gamma, alpha, nu=0)
        params = np.array([r, alpha, lamda, Gamma], dtype=float)
        return params
    
    def to_conventional(
        self,
        params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Convert non-degenerate embedding parameters to conventional embedding variables
        Args:
            params : np.ndarray, shape(4,)
                non-degenerate parameters in the order [ r, α, λ, Γ ]

        Returns:
            cparams : np.ndarray, shape(5,)
                conventional parameters [ θ[0], θ[1], γ, δ, Ω ]
        """
        params = np.asarray(params, dtype=float)
        if params.shape != (4,):
           raise ValueError(f"The non-degenerate parameters must be supplied as a length-4 vector, instead got {params.shape = }")
        r, alpha, lamda, Gamma = params
        if abs(alpha) < Gamma:
            delta = Gamma
            Omega = 0.0
            theta1 = r * np.sqrt((1 + alpha/Gamma)/2)
            theta2 = r * np.sqrt((1 - alpha/Gamma)/2)
        else:
            delta = abs(alpha)
            Omega = np.sqrt(alpha**2 - np.sign(Gamma)*Gamma**2)
            theta1 = r if alpha > 0 else 0.0 
            theta2 = 0.0 if alpha > 0 else r
        gamma = lamda + self._gamma_lower_bound(Gamma, alpha)
        return np.asarray([theta1, theta2, gamma, delta, Omega])
    
    def jac_conventional(
        self,
        params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Convert non-degenerate embedding parameters to conventional embedding variables and compute the jacobian of the conventional vars w.r.t. params

        Args:
            params : np.ndarray, shape(4,)
                non-degenerate parameters in the order [ r, α, λ, Γ ]

        Returns:
            cparams : np.ndarray, shape(5,)
                conventional parameters [ θ[0], θ[1], γ, δ, Ω ]
            jac : np.ndarray, shape(5, 4)
                jac[i,j] = D[ cparams[i], params[j] ]
        """
        params = np.asarray(params, dtype=float)
        if params.shape != (4,):
           raise ValueError(f"The non-degenerate parameters must be supplied as a length-4 vector, instead got {params.shape = }")
        r, alpha, lamda, Gamma = params
        gamma = lamda + self._gamma_lower_bound(Gamma, alpha)
        jac = np.zeros((5,4))
        if abs(alpha) < Gamma:
            delta = Gamma
            Omega = 0.0
            sqrt_p = np.sqrt((1 + alpha/Gamma)/2)
            sqrt_m = np.sqrt((1 - alpha/Gamma)/2)
            theta1 = r * sqrt_p
            theta2 = r * sqrt_m
            # θ[0] w.r.t. [ r, α, λ, Γ ]
            jac[0,0] = sqrt_p
            jac[0,1] = r / (4 * Gamma * sqrt_p)
            jac[0,3] = -r*alpha / (4*Gamma**2) / sqrt_p
            # θ[1] w.r.t. [ r, α, λ, Γ ]
            jac[1,0] = sqrt_m
            jac[1,1] = -r / (4 * Gamma * sqrt_m)
            jac[1,3] = r*alpha / (4*Gamma**2) / sqrt_m
            # δ w.r.t [ r, α, λ, Γ ]
            jac[3,3] = 1.0
            # Ω w.r.t [ r, α, λ, Γ ], but Ω = 0
        else:
            delta = abs(alpha)
            abs_Gamma = abs(Gamma)
            Omega = np.sqrt(alpha**2 - abs_Gamma*Gamma)
            if alpha > 0:
                theta1 = r
                # θ[0] w.r.t. [ r, α, λ, Γ ]
                jac[0,0] = 1.0
                theta2 = 0.0
            else:
                theta1 = 0.0
                theta2 = r
                # θ[1] w.r.t. [ r, α, λ, Γ ]
                jac[1,0] = 1.0
            # δ w.r.t [ r, α, λ, Γ ]
            jac[3,1] = np.sign(alpha)
            # Ω w.r.t [ r, α, λ, Γ ]
            jac[4,1] = alpha/Omega
            jac[4,3] = -abs_Gamma/Omega
        # γ w.r.t [ r, α, λ, Γ ]
        tmp = self._gamma_lower_bound(Gamma, alpha, nu=1)
        jac[2,1] = tmp[1]
        jac[2,2] = 1.0
        jac[2,3] = tmp[0]
        cparams = np.asarray([theta1, theta2, gamma, delta, Omega])
        return cparams, jac
    
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
        A: npt.NDArray[np.floating],
        **kwargs
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
        return cls(theta, gamma, delta, Omega, **kwargs)

    def compute_drift_matrix(
            self, 
            params: npt.NDArray
        ) -> npt.NDArray[np.floating]:
        """Calculate the drift matrix for a given set of conventional parameters, 
        packed as a 5-tuple in the order ( θ[0], θ[1], γ, δ, Ω )

        Returns
        -------
        numpy.ndarray
            A 2x2 array.
        """
        theta1, theta2, gamma, delta, Omega = params
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
        """The gradient of the drift matrix with respect to the primitive parameters.

        In terms of *conventional* parameters 
        cparams = [θ[0], θ[1], γ, δ, Ω]
            
                (  0    θ[0] θ[1])
            A = (-θ[0]  γ+δ    Ω )
                (-θ[1]  -Ω   γ-δ )
        """
    
        grad_A_conventional = np.zeros((5, 3, 3), dtype=float)
        # dA/d(theta1)
        grad_A_conventional[0, 0, 1] = 1.0
        grad_A_conventional[0, 1, 0] = -1.0
        # dA/d(theta2)
        grad_A_conventional[1, 0, 2] = 1.0
        grad_A_conventional[1, 2, 0] = -1.0
        # dA/d(gamma)
        grad_A_conventional[2, 1, 1] = 1.0
        grad_A_conventional[2, 2, 2] = 1.0
        # dA/d(delta)
        grad_A_conventional[3, 1, 1] = 1.0
        grad_A_conventional[3, 2, 2] = -1.0
        # dA/d(Omega)
        grad_A_conventional[4, 1, 2] = 1.0
        grad_A_conventional[4, 2, 1] = -1.0
        # chain rule
        _, jac = self.jac_conventional(params)
        grad_A = np.einsum(
            'lij,lk->kij', 
            grad_A_conventional,
            jac)
        return grad_A
    
    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Memory kernel function for the two-variable embedding, 
            K(t) = r² exp(-γt) [ C(Γt) - α·t·S(Γt) ]
            where γ = λ + Softmax(Γ, Softabs(α))
        """
        t = np.abs(np.atleast_1d(time))
        r, alpha, lamda, Gamma = self.primitive_params
        gamma = lamda + self._gamma_lower_bound(Gamma, alpha)
        ans = expcoscosh(Gamma, t, gamma)
        ans -= alpha*t*expsincsinhc(Gamma, t, gamma)
        return r**2 * ans
    
    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Gradient of the memory kernel function for the two-variable embedding w.r.t. primitive parameters [r, α, λ, Γ].
        
            K(t) = r² exp(-γt) [ C(Γt) - α·t·S(Γt) ]
            where γ = λ + Softmax(Γ, Softabs(α))
        """
        t = np.abs(np.atleast_1d(time))
        r, alpha, lamda, Gamma = self.primitive_params
        gamma = lamda + self._gamma_lower_bound(Gamma, alpha)
        C = expcoscosh(Gamma, t, gamma)
        S = expsincsinhc(Gamma, t, gamma)
        K_base = C - alpha*t*S
        ans = np.empty((4, len(t)))
        d_gamma_Gamma, d_gamma_alpha  = self._gamma_lower_bound(Gamma, alpha, nu=1)
        ans[0] = 2*r * K_base
        ans[1] = r**2 * (-t * S - d_gamma_alpha * t * K_base)
        ans[2] = -r**2 * t * K_base
        dC = expcoscosh(Gamma, t, gamma, nu=1, wrt="Gamma")
        dS = expsincsinhc(Gamma, t, gamma, nu=1, wrt="Gamma")
        ans[3] = r**2 * (dC - alpha * t * dS - d_gamma_Gamma * t * K_base)
        return ans

    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """
        Hessian of the memory kernel function for the two-variable embedding w.r.t. primitive parameters [r, α, λ, Γ].
        Shape: (4, 4, len(time))
        """
        raise NotImplementedError
    
    def spectrum_func(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        freq = np.abs(np.atleast_1d(frequency))
        r, alpha, lamda, Gamma = self.primitive_params
        gamma = lamda + self._gamma_lower_bound(Gamma, alpha)
        w2 = freq**2
        g2 = gamma**2
        y = Gamma * abs(Gamma)
        ans = (alpha + gamma)*w2 - (alpha - gamma)*(g2 - y)
        ans /= g2**2 + 2*g2*(w2-y) + (w2+y)**2
        ans *= r**2
        return ans

    def spectrum_grad(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        freq = np.abs(np.atleast_1d(frequency))
        r, alpha, lamda, Gamma = self.primitive_params
        gamma = lamda + self._gamma_lower_bound(Gamma, alpha)
        w2 = freq**2
        g2 = gamma**2
        y = Gamma * abs(Gamma)
        numerator = (alpha + gamma)*w2 - (alpha - gamma)*(g2 - y)
        denominator = g2**2 + 2*g2*(w2 - y) + (w2 + y)**2
        S_base = numerator / denominator
        ans = np.empty((4, len(freq)))
        ans[0] = 2 * r * S_base
        d_num_gamma = w2 - y - 2*alpha*gamma + 3*g2
        d_denom_gamma = 4 * gamma * (g2 + w2 - y)
        dS_dgamma = r**2 * (d_num_gamma / denominator - numerator / denominator**2 * d_denom_gamma) 
        d_gamma_Gamma, d_gamma_alpha = self._gamma_lower_bound(Gamma, alpha, nu=1)
        d_num_alpha = w2 + y - g2
        ans[1] = r**2 * d_num_alpha / denominator + dS_dgamma * d_gamma_alpha
        ans[2] = dS_dgamma
        dy_dGamma = 2 * abs(Gamma)
        d_num_y = alpha - gamma
        d_denom_y = 2*(y + w2 - g2)
        dS_dy = r**2 * (d_num_y / denominator - numerator * d_denom_y / denominator**2)
        ans[3] = dS_dy * dy_dGamma + dS_dgamma * d_gamma_Gamma
        return ans
    
    def spectrum_hess(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """
        Hessian of the spectrum for the two-variable embedding w.r.t. primitive parameters [r, α, λ, Γ].
        
        S(ω) = r² · [(α+γ)ω² - (α-γ)(γ²-y)] / [γ⁴ + 2γ²(ω²-y) + (ω²+y)²]
        where y = Γ·|Γ| and γ = λ + Softmax(Γ, Softabs(α))
        
        Shape: (4, 4, len(freq))
        """
        freq = np.abs(np.atleast_1d(frequency))
        r, alpha, lamda, Gamma = self.primitive_params
        gamma = lamda + self._gamma_lower_bound(Gamma, alpha)
        
        w2 = freq**2
        g2 = gamma**2
        y = Gamma * abs(Gamma)
        numerator = (alpha + gamma)*w2 - (alpha - gamma)*(g2 - y)
        denominator = g2**2 + 2*g2*(w2 - y) + (w2 + y)**2
        S_base = numerator / denominator
        
        # First derivatives of numerator and denominator
        d_num_gamma = w2 - y - 2*alpha*gamma + 3*g2
        d_denom_gamma = 4 * gamma * (g2 + w2 - y)
        d_num_alpha = w2 - g2 + y
        d_num_y = alpha - gamma
        d_denom_y = 2*(w2 + y - g2)
        
        # dS/dγ, dS/dy
        dS_dgamma = r**2 * (d_num_gamma / denominator - numerator / denominator**2 * d_denom_gamma)
        dS_dy = r**2 * (d_num_y / denominator - numerator * d_denom_y / denominator**2)
        
        # Get first and second derivatives of γ
        d_gamma_Gamma, d_gamma_alpha = self._gamma_lower_bound(Gamma, alpha, nu=1)
        
        d2_gamma_mat = self._gamma_lower_bound(Gamma, alpha, nu=2)
        d2_gamma_Gamma2 = d2_gamma_mat[0, 0]
        d2_gamma_GammaAlpha = d2_gamma_mat[0, 1]
        d2_gamma_alpha2 = d2_gamma_mat[1, 1]
        
        # dy/dΓ and d²y/dΓ²
        dy_dGamma = 2 * abs(Gamma)
        d2y_dGamma2 = 2 * np.sign(Gamma) if Gamma != 0 else 0
        
        # Second derivatives of numerator and denominator w.r.t. γ
        d2_num_gamma2 = -2*alpha + 6*gamma
        d2_denom_gamma2 = 12*g2 + 4*(w2 - y)
        
        # Second derivatives w.r.t. y
        d2_num_y2 = 0
        d2_denom_y2 = 2
        
        # Cross derivatives
        d2_num_gamma_alpha = -2*gamma
        d2_num_gamma_y = -1
        d2_denom_gamma_y = -4*gamma
        d2_num_alpha_y = 1
        
        # Second derivative d²S/dγ²
        d2S_dgamma2 = r**2 * (
            d2_num_gamma2 / denominator 
            - 2 * d_num_gamma * d_denom_gamma / denominator**2
            - numerator * d2_denom_gamma2 / denominator**2
            + 2 * numerator * d_denom_gamma**2 / denominator**3
        )
        
        # d²S/dγdα_direct
        d2S_dgamma_dalpha = r**2 * (
            d2_num_gamma_alpha / denominator
            - d_num_alpha * d_denom_gamma / denominator**2
        )
        
        # d²S/dγdy
        d2S_dgamma_dy = r**2 * (
            d2_num_gamma_y / denominator
            - d_num_y * d_denom_gamma / denominator**2
            - d_num_gamma * d_denom_y / denominator**2
            - numerator * d2_denom_gamma_y / denominator**2
            + 2 * numerator * d_denom_gamma * d_denom_y / denominator**3
        )
        
        # d²S/dα²_direct
        d2S_dalpha2_direct = 0
        
        # d²S/dαdy_direct
        d2S_dalpha_dy = r**2 * (
            d2_num_alpha_y / denominator
            - d_num_alpha * d_denom_y / denominator**2
        )
        
        # d²S/dy²
        d2S_dy2 = r**2 * (
            d2_num_y2 / denominator
            - 2 * d_num_y * d_denom_y / denominator**2
            - numerator * d2_denom_y2 / denominator**2
            + 2 * numerator * d_denom_y**2 / denominator**3
        )
        
        # Build the Hessian matrix
        hess = np.zeros((4, 4, len(freq)))
        
        # H[0,0] = d²S/dr²
        hess[0, 0] = 2 * S_base
        
        # H[0,1] = H[1,0] = d²S/drda
        hess[0, 1] = hess[1, 0] = 2 * r * (d_num_alpha / denominator + dS_dgamma * d_gamma_alpha / r**2)
        
        # H[0,2] = H[2,0] = d²S/drdλ
        hess[0, 2] = hess[2, 0] = 2 * r * dS_dgamma / r**2
        
        # H[0,3] = H[3,0] = d²S/drdΓ
        hess[0, 3] = hess[3, 0] = 2 * r * (dS_dy * dy_dGamma / r**2 + dS_dgamma * d_gamma_Gamma / r**2)
        
        # H[1,1] = d²S/dα²
        hess[1, 1] = (
            d2S_dalpha2_direct
            + 2 * d2S_dgamma_dalpha * d_gamma_alpha
            + d2S_dgamma2 * d_gamma_alpha**2
            + dS_dgamma * d2_gamma_alpha2
        )
        
        # H[1,2] = H[2,1] = d²S/dαdλ
        hess[1, 2] = hess[2, 1] = d2S_dgamma_dalpha + d2S_dgamma2 * d_gamma_alpha
        
        # H[1,3] = H[3,1] = d²S/dαdΓ
        hess[1, 3] = hess[3, 1] = (
            d2S_dalpha_dy * dy_dGamma
            + d2S_dgamma_dalpha * d_gamma_Gamma
            + d2S_dgamma_dy * d_gamma_alpha * dy_dGamma
            + d2S_dgamma2 * d_gamma_alpha * d_gamma_Gamma
            + dS_dgamma * d2_gamma_GammaAlpha
        )
        
        # H[2,2] = d²S/dλ²
        hess[2, 2] = d2S_dgamma2
        
        # H[2,3] = H[3,2] = d²S/dλdΓ
        hess[2, 3] = hess[3, 2] = (
            d2S_dgamma_dy * dy_dGamma
            + d2S_dgamma2 * d_gamma_Gamma
        )
        
        # H[3,3] = d²S/dΓ²
        hess[3, 3] = (
            dS_dy * d2y_dGamma2
            + 2 * d2S_dgamma_dy * dy_dGamma * d_gamma_Gamma
            + d2S_dy2 * dy_dGamma**2
            + d2S_dgamma2 * d_gamma_Gamma**2
            + dS_dgamma * d2_gamma_Gamma2
        )
        
        return hess