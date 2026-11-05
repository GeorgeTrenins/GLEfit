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
from glefit.utils.special import sinc
import numpy as np
import numpy.typing as npt
import copy

# In the overdamped regime, the code needs sinhc(x) - sinhc''(x), 
# which loses accuracy for large x. Set a switch above which evaluation
# is in terms of decaying exponentials.
_SINHC_SWITCH = 1.0

# Keys of earlier versions, and what replaced them
_REMOVED_KEYS = {
    "sigma": "γ now follows from y, without Softmax or Softabs",
    "threshold": "γ now follows from y, without Softmax or Softabs",
    "Omega": "the antisymmetric coupling between s₁ and s₂ is now 'zeta'",
}

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
    
    def __init__(self, theta: npt.NDArray[np.floating], gamma: float, delta: float, zeta: float, *args, **kwargs) -> None:
        """Initialise the embedder from conventional parameters.

        Args:
            theta (array-like): the couplings [θ₁, θ₂]
            gamma (float): the mean decay rate γ
            delta (float): half the difference δ between the decay rates of s₁ and s₂
            zeta (float): the antisymmetric coupling ζ between s₁ and s₂
            *args, **kwargs: passed to BaseEmbedder; `mappers` replaces the default mappers

        Raises:
            TypeError: if gamma, delta or zeta is not a float or int, or if `sigma`,
                `threshold` or `Omega` is passed, which belonged to earlier versions
            ValueError: if theta does not have two entries, or if the kernel lies outside
                the admissible region γ > |α|, γ² > g

        Notes:
            The drift matrix is

                    (  0    θ₁    θ₂ )
                A = ( -θ₁  γ+δ    ζ  )
                    ( -θ₂  -ζ    γ-δ )

            and the kernel is 
                     K(t) = r² exp(-γ|t|)[cosh(t√g) - α|t| sinhc(t√g)], 
            with
                  r² = θ₁² + θ₂²,     α = δ(θ₁² - θ₂²)/r²,         g = δ² - ζ² 
            The embedder fits the primitive parameters [r, α, g, y], where
                  γ² = (g + α²)/2 + sqrt(((g - α²)/2)² + exp(2y)), 
            and builds the drift matrix in the maximal gauge δ = γ.
        """

        self._reject_removed_keys(kwargs)
        for variable, symbol in zip([gamma, delta, zeta], "γδζ"):
            if not isinstance(variable, (int, float)):
                raise TypeError(f"{symbol} must be a float, got {type(variable).__name__}")
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
            raise ValueError(f"θ must be a length-2 vector, instead got {theta.shape = }")
        kwargs.setdefault(
            "mappers", [
                LowerBoundMapper(), # r > 0
                IdentityMapper(),   # α ∈ R
                IdentityMapper(),   # g ∈ R
                IdentityMapper()    # y ∈ R
        ])
        super().__init__(*args, **kwargs)
        # set primitive params directly
        conventional_parameters = np.concatenate([theta, [gamma, delta, zeta]])
        primitive_parameters = self.to_primitive(conventional_parameters)
        self._primitive_params[:] = primitive_parameters
        self._x[:] = self._forward_map(primitive_parameters)

    @classmethod
    def from_dict(
        cls, 
        parameters: dict
    ) -> "TwoAuxEmbedder":
        """Return an embedder built from a config dictionary.

        Args:
            parameters (dict): `theta`, `gamma`, `delta` and `zeta`, plus keyword
                arguments for the constructor. It is not modified.

        Returns:
            TwoAuxEmbedder: the new embedder

        Raises:
            TypeError: if a key of an earlier version, such as `Omega`, is present
        """
        param_copy = copy.deepcopy(parameters)
        # before the pops, so that `Omega` in place of `zeta` is not a bare KeyError
        cls._reject_removed_keys(param_copy)
        theta = param_copy.pop("theta")
        gamma = param_copy.pop("gamma")
        delta = param_copy.pop("delta")
        zeta = param_copy.pop("zeta")
        return cls(theta, gamma, delta, zeta, **param_copy)

    @staticmethod
    def _reject_removed_keys(parameters: dict) -> None:
        """Raise TypeError for a key that belonged to an earlier version of the embedder.

        Args:
            parameters (dict): keyword arguments or config parameters

        Raises:
            TypeError: if a key of _REMOVED_KEYS is present, naming what replaced it
        """
        for key, reason in _REMOVED_KEYS.items():
            if key in parameters:
                raise TypeError(f"TwoAuxEmbedder no longer takes '{key}': {reason}")

    @staticmethod
    def from_harmonic(
        weight: float, omega: float, gamma: float 
    ) -> dict[str, float]:
        """Return the conventional parameters of a damped harmonic mode, for from_dict.

        Args:
            weight (float): the weight of the mode, as from WAM_discretization
            omega (float): the frequency of the mode
            gamma (float): its damping rate

        Returns:
            dict: `theta` = [sqrt(2·weight/π), 0], `gamma`, `delta` = 0 and `zeta` = omega

        Notes:
            The kernel is (2·weight/π)·exp(-γ|t|)·cos(ωt).
        """
        conventional_parameters = {
            "theta" : [float(np.sqrt(2*weight/np.pi)), 0.0],
            "gamma" : float(gamma),
            "delta" : 0.0,
            "zeta" : float(omega)
        }
        return conventional_parameters

    @classmethod
    def from_matrix(
        cls, 
        theta: npt.NDArray[np.floating],
        A: npt.NDArray[np.floating],
        **kwargs
    ) -> "TwoAuxEmbedder":
        """Return an embedder with the given auxiliary drift matrix and couplings.

        Args:
            theta (array-like): the couplings [θ₁, θ₂]
            A (array-like): the 2×2 drift matrix of the auxiliary variables
            **kwargs: passed to the constructor

        Returns:
            TwoAuxEmbedder: an embedder with the same kernel, in the maximal gauge

        Raises:
            ValueError: if theta or A has the wrong shape, if the symmetric part of A has
                a negative eigenvalue, or if the kernel is not admissible

        Notes:
            θ and A are first rotated so that the symmetric part of A is diagonal, which
            gives the form [[γ+δ, ζ], [-ζ, γ-δ]].
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
        zeta = A[0,1]
        return cls(theta, gamma, delta, zeta, **kwargs)

    @staticmethod
    def _gaps(alpha, g, y):
        """Return the gaps γ² - α² and γ² - g, and D = sqrt(h² + exp(2y)) with h = (g - α²)/2.

        Args:
            alpha (float): the asymmetry α
            g (float): δ² - ζ²
            y (float): the primitive parameter that sets γ

        Returns:
            tuple of float: γ² - α², γ² - g and D

        Notes:
            γ² = (g + α²)/2 + D, so the gaps are D + h and D - h, and their product
            is exp(2y). The smaller gap is computed as exp(2y) divided by the larger,
            which avoids cancellation near the boundary of the admissible region.
        """
        h = (g - alpha**2) / 2
        e = np.exp(y)
        D = np.hypot(h, e)
        if h >= 0:
            gap_alpha = D + h
            gap_g = e * (e / gap_alpha)
        else:
            gap_g = D - h
            gap_alpha = e * (e / gap_g)
        return gap_alpha, gap_g, D

    @staticmethod
    def _gamma_pm(gamma, alpha, gap_alpha):
        """Return γ + α and γ - α.

        Args:
            gamma (float): the mean decay rate γ
            alpha (float): the asymmetry α
            gap_alpha (float): γ² - α², from _gaps

        Returns:
            tuple of float: γ + α and γ - α

        Notes:
            The larger factor, γ + |α|, is computed directly, and the smaller one as
            (γ² - α²) divided by the larger, which avoids cancellation when γ ≈ |α|.
        """
        larger = gamma + abs(alpha)
        smaller = gap_alpha / larger
        return (larger, smaller) if alpha >= 0 else (smaller, larger)

    @classmethod
    def _gamma(cls, alpha, g, y, nu=0):
        """Return γ, or its first or second derivatives with respect to (α, g, y).

        Args:
            alpha (float): the asymmetry α
            g (float): δ² - ζ²
            y (float): the primitive parameter that sets γ
            nu (int): 0 for γ, 1 for the gradient, 2 for the Hessian

        Returns:
            float or ndarray: γ, or an array of shape (3,) or (3, 3)

        Raises:
            ValueError: if nu is not 0, 1 or 2
        """
        gap_alpha, gap_g, D = cls._gaps(alpha, g, y)
        gamma = np.sqrt(alpha**2 + gap_alpha)
        if nu == 0:
            return gamma
        # gradient of s = γ² with respect to (α, g, y)
        ds = np.array([alpha * gap_g / D, gap_alpha / (2*D), gap_alpha * gap_g / D])
        if nu == 1:
            return ds / (2*gamma)
        elif nu == 2:
            d2s = cls._d2_gamma_squared(alpha, gap_alpha, gap_g, D)
            return d2s / (2*gamma) - np.outer(ds, ds) / (4*gamma**3)
        else:
            raise ValueError(f"Expecting nu = 0, 1, or 2, instead got nu = {nu}")

    @classmethod
    def _gap_g(cls, alpha, g, y, nu=0):
        """Return G = γ² - g, or its first or second derivatives with respect to (α, g, y).

        Args:
            alpha (float): the asymmetry α
            g (float): δ² - ζ²
            y (float): the primitive parameter that sets γ
            nu (int): 0 for G, 1 for the gradient, 2 for the Hessian

        Returns:
            float or ndarray: G, or an array of shape (3,) or (3, 3)

        Raises:
            ValueError: if nu is not 0, 1 or 2

        Notes:
            The gradient is G·[α/D, -1/(2D), (γ² - α²)/D], which stays accurate as G → 0.
        """
        gap_alpha, gap_g, D = cls._gaps(alpha, g, y)
        if nu == 0:
            return gap_g
        elif nu == 1:
            return gap_g * np.array([alpha/D, -1/(2*D), gap_alpha/D])
        elif nu == 2:
            # g enters G linearly
            return cls._d2_gamma_squared(alpha, gap_alpha, gap_g, D)
        else:
            raise ValueError(f"Expecting nu = 0, 1, or 2, instead got nu = {nu}")

    @staticmethod
    def _d2_gamma_squared(alpha, gap_alpha, gap_g, D):
        """Return the Hessian of γ² with respect to (α, g, y).

        Args:
            alpha (float): the asymmetry α
            gap_alpha (float): γ² - α², from _gaps
            gap_g (float): γ² - g, from _gaps
            D (float): D, from _gaps

        Returns:
            ndarray: shape (3, 3)
        """
        e2 = gap_alpha * gap_g  # exp(2y)
        h = (gap_alpha - gap_g) / 2
        c = e2 / D**3
        return np.array([
            [gap_g/D + alpha**2 * c, -alpha * c/2,  alpha * h * c],
            [-alpha * c/2,            c/4,           -h * c/2],
            [alpha * h * c,           -h * c/2,      c * (2*h**2 + e2)]])

    def to_primitive(
        self,
        conventional_params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Return the primitive parameters [r, α, g, y] of conventional parameters.

        Args:
            conventional_params (array-like): [θ₁, θ₂, γ, δ, ζ]

        Returns:
            ndarray: [r, α, g, y], with y = ln[(γ² - α²)(γ² - g)]/2

        Raises:
            ValueError: if conventional_params does not have five entries, or if the
                kernel is not inside the admissible region γ > |α|, γ² > g

        Notes:
            γ ∓ α and γ² - g are computed as sums of terms that are non-negative when
            γ ≥ |δ|, which keeps y accurate near the boundary.
        """
        conventional_params = np.asarray(conventional_params, dtype=float)
        if conventional_params.shape != (5,):
           raise ValueError(f"The conventional parameters must be supplied "
                            f"as a length-5 vector, instead got {conventional_params.shape = }")
        th2 = conventional_params[:2]**2
        gamma, delta, zeta = conventional_params[2:]
        r2 = th2[0] + th2[1]
        alpha = delta * (th2[0] - th2[1]) / r2
        g = delta**2 - zeta**2
        # γ ∓ α and γ² - g as sums of terms that are non-negative when γ ≥ |δ|
        gamma_minus_alpha = ((gamma - delta)*th2[0] + (gamma + delta)*th2[1]) / r2
        gamma_plus_alpha = ((gamma + delta)*th2[0] + (gamma - delta)*th2[1]) / r2
        gap_g = (gamma - delta)*(gamma + delta) + zeta**2
        if not (gamma_minus_alpha > 0 and gamma_plus_alpha > 0 and gap_g > 0):
            raise ValueError(
                f"The memory kernel must lie inside the admissible region "
                f"γ > |α|, γ² > g, instead got γ = {gamma}, α = {alpha}, g = {g}")
        y = (np.log(gamma_minus_alpha * gamma_plus_alpha) + np.log(gap_g)) / 2
        primitive_params = np.array([np.sqrt(r2), alpha, g, y], dtype=float)
        return primitive_params

    def to_conventional(
        self,
        primitive_params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Return the conventional parameters of primitive parameters, in the maximal gauge δ = γ.

        Args:
            primitive_params (array-like): [r, α, g, y]

        Returns:
            ndarray: [θ₁, θ₂, γ, δ, ζ], with θ₁,₂ = r·sqrt((1 ± α/γ)/2), δ = γ and
            ζ = sqrt(γ² - g)

        Raises:
            ValueError: if primitive_params does not have four entries
        """
        primitive_params = np.asarray(primitive_params, dtype=float)
        if primitive_params.shape != (4,):
           raise ValueError(
               f"The non-degenerate parameters must be supplied "
               f"as a length-4 vector, instead got {primitive_params.shape = }")
        r, alpha, g, y = primitive_params
        gap_alpha, gap_g, _ = self._gaps(alpha, g, y)
        gamma = np.sqrt(alpha**2 + gap_alpha)
        gamma_plus_alpha, gamma_minus_alpha = self._gamma_pm(gamma, alpha, gap_alpha)
        theta1 = r * np.sqrt(gamma_plus_alpha / (2*gamma))
        theta2 = r * np.sqrt(gamma_minus_alpha / (2*gamma))
        return np.asarray([theta1, theta2, gamma, gamma, np.sqrt(gap_g)])

    def jac_conventional(
        self,
        primitive_params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Return the conventional parameters and their Jacobian with respect to the primitive ones.

        Args:
            primitive_params (array-like): [r, α, g, y]

        Returns:
            tuple of ndarray: [θ₁, θ₂, γ, δ, ζ], as from to_conventional, and the Jacobian
            jac[i, j] = ∂conventional[i]/∂primitive[j], shape (5, 4)

        Raises:
            ValueError: if primitive_params does not have four entries

        Notes:
            The rows are built from logarithmic derivatives of the gaps (see _gaps),
            which stay accurate near the boundary of the admissible region.
        """
        primitive_params = np.asarray(primitive_params, dtype=float)
        if primitive_params.shape != (4,):
           raise ValueError(f"The primitive parameters must be supplied as a length-4 vector, instead got {primitive_params.shape = }")
        r, alpha, g, y = primitive_params
        conventional_params = self.to_conventional(primitive_params)
        theta1, theta2, gamma, _, zeta = conventional_params
        gap_alpha, gap_g, D = self._gaps(alpha, g, y)
        gamma_plus_alpha, gamma_minus_alpha = self._gamma_pm(gamma, alpha, gap_alpha)
        # Logarithmic derivatives with respect to [ r, α, g, y ]. Near the boundary of the
        # admissible region, the gaps and the smaller of γ ± α are small; differentiating
        # them directly avoids the cancellation in, e.g., ∂(γ² - g)/∂g = 2γ·∂γ/∂g - 1.
        d_alpha = np.array([0.0, 1.0, 0.0, 0.0])
        dlog_r = np.array([1/r, 0.0, 0.0, 0.0])
        d_gamma = np.concatenate([[0.0], self._gamma(alpha, g, y, nu=1)])
        dlog_gap_alpha = np.array([0.0, -alpha/D, 1/(2*D), gap_g/D])
        dlog_gap_g = np.array([0.0, alpha/D, -1/(2*D), gap_alpha/D])
        if alpha >= 0:
            dlog_plus = (d_gamma + d_alpha) / gamma_plus_alpha
            dlog_minus = dlog_gap_alpha - dlog_plus
        else:
            dlog_minus = (d_gamma - d_alpha) / gamma_minus_alpha
            dlog_plus = dlog_gap_alpha - dlog_minus
        jac = np.empty((5,4))
        # θ[0], θ[1] = r·sqrt((γ ± α)/(2γ))
        jac[0] = theta1 * (dlog_r + (dlog_plus - d_gamma/gamma)/2)
        jac[1] = theta2 * (dlog_r + (dlog_minus - d_gamma/gamma)/2)
        # γ and δ = γ
        jac[2] = jac[3] = d_gamma
        # ζ = sqrt(γ² - g)
        jac[4] = zeta * dlog_gap_g/2
        return conventional_params, jac
    
    @staticmethod
    def rotate_matrix(
        theta: npt.NDArray[np.floating],
        A: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Return θ and A in the basis where the symmetric part of A is diagonal.

        Args:
            theta (ndarray): the couplings, shape (2,)
            A (ndarray): the auxiliary drift matrix, shape (2, 2)

        Returns:
            tuple of ndarray: the rotated θ and A

        Raises:
            ValueError: if the symmetric part of A has a negative eigenvalue
        """
        Asym = (A + A.T)/2
        eigs = np.linalg.eigh(Asym)
        if np.any(eigs.eigenvalues < 0):
            raise ValueError(
                f"The auxiliary variable drift matrix must be positive-definite. "
                f"Instead, the symmetric component {Asym = } was found to have "
                f"the eigenvalues {eigs.eigenvalues}.")
        new_theta = np.dot(eigs.eigenvectors.T, theta)
        new_A = np.linalg.multi_dot([eigs.eigenvectors.T, A, eigs.eigenvectors])
        return new_theta, new_A

    def compute_drift_matrix(
        self, 
        primitive_params: npt.NDArray
    ) -> npt.NDArray[np.floating]:
        """Return the drift matrix Ap in the maximal gauge.

        Args:
            primitive_params (array-like): [r, α, g, y]

        Returns:
            ndarray: shape (3, 3)
        """
        theta1, theta2, gamma, delta, zeta = self.to_conventional(primitive_params)
        A = np.zeros((3,3))
        A[0,1] = theta1
        A[0,2] = theta2
        A[1,0] = -theta1
        A[2,0] = -theta2
        A[1,1] = gamma + delta
        A[2,2] = gamma - delta
        A[1,2] = zeta
        A[2,1] = -zeta
        return A
    
    def drift_matrix_param_grad(
        self, 
        primitive_params: npt.NDArray
    ) -> npt.NDArray[np.floating]:
        """Return the derivatives of the drift matrix with respect to the primitive parameters.

        Args:
            primitive_params (array-like): [r, α, g, y]

        Returns:
            ndarray: ∂Ap/∂p, shape (4, 3, 3)

        Notes:
            The derivatives with respect to the conventional parameters [θ₁, θ₂, γ, δ, ζ]
            are constant, and jac_conventional supplies the chain rule.
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
        # dA/d(zeta)
        grad_A_conventional[4, 1, 2] = 1.0
        grad_A_conventional[4, 2, 1] = -1.0
        # chain rule
        _, jac = self.jac_conventional(primitive_params)
        grad_A = np.einsum(
            'lij,lk->kij', 
            grad_A_conventional,
            jac)
        return grad_A
    
    @staticmethod
    def _abs_grid(grid: ScalarArr) -> npt.NDArray[np.floating]:
        """Return the absolute values of a time or frequency grid as a 1-d float array.

        Args:
            grid (float or array-like): times or frequencies

        Returns:
            ndarray: shape (ngrid,)

        Notes:
            The closed forms are even in t and ω. The cast to float stops integer
            input from being truncated.
        """
        return np.abs(np.atleast_1d(np.asarray(grid, dtype=float)))

    @staticmethod
    def _damped_cosh_sinhc(g, gamma, gap_g, t):
        """Return exp(-γt)·cosh(t√g) and exp(-γt)·t·sinhc(t√g).

        Args:
            g (float): δ² - ζ²; g > 0 is overdamped, g ≤ 0 underdamped or critical
            gamma (float): the mean decay rate γ
            gap_g (float): γ² - g, from _gaps
            t (ndarray): non-negative times, as floats

        Returns:
            tuple of ndarray: the two functions at each time, each with the shape of t

        Notes:
            For g < 0, cosh and t·sinhc become cos(ω̄t) and sin(ω̄t)/ω̄ with ω̄ = √(-g),
            and for g = 0 they are 1 and t. For g > 0 they are written with decaying
            exponentials, so nothing overflows. The slow rate γ - √g is computed as
            (γ² - g)/(γ + √g).
        """
        if g < 0:
            omega_bar = np.sqrt(-g)
            decay = np.exp(-gamma*t)
            return decay*np.cos(omega_bar*t), decay*np.sin(omega_bar*t)/omega_bar
        if g == 0:
            decay = np.exp(-gamma*t)
            return decay, decay*t
        kappa = np.sqrt(g)
        decay = np.exp(-gap_g/(gamma + kappa)*t)
        em1 = np.expm1(-2*kappa*t)
        return decay*(1 + em1/2), -decay*em1/(2*kappa)

    @staticmethod
    def _damped_sinhc_g_derivative(g, gamma, gap_g, t):
        """Return exp(-γt) times the derivative of sinhc(t√g) with respect to g.

        Args:
            g (float): δ² - ζ²; g > 0 is overdamped, g ≤ 0 underdamped or critical
            gamma (float): the mean decay rate γ
            gap_g (float): γ² - g, from _gaps
            t (ndarray): non-negative times, as floats

        Returns:
            ndarray: the damped derivative at each time, with the shape of t

        Notes:
            With s = sinc and z = t√(-g), the derivative is (t²/4)[s(z) + s''(z)].
            This form is used for g ≤ 0, where z is real, and for g > 0 at short
            times, √g·t ≤ _SINHC_SWITCH, where z is imaginary. At longer times it
            loses accuracy, so (cosh - sinhc)/(2g) is evaluated with decaying
            exponentials instead.
        """
        if g <= 0:
            z = t*np.sqrt(-g)
            return np.exp(-gamma*t) * t**2/4 * (sinc(z) + sinc(z, nu=2))
        kappa = np.sqrt(g)
        ans = np.empty_like(t)
        small = kappa*t <= _SINHC_SWITCH
        ts, tl = t[small], t[~small]
        z = 1j*kappa*ts
        ans[small] = np.exp(-gamma*ts) * ts**2/4 * np.real(sinc(z) + sinc(z, nu=2))
        em1 = np.expm1(-2*kappa*tl)
        decay = np.exp(-gap_g/(gamma + kappa)*tl)
        ans[~small] = decay * (1 + em1/2 + em1/(2*kappa*tl)) / (2*g)
        return ans

    def _kernel_partials(self, t, nu):
        """Return K/r² and, for nu = 1, its partial derivatives with respect to (α, g, γ).

        Args:
            t (ndarray): non-negative times, as floats
            nu (int): 0 for the value only, 1 to add the partial derivatives

        Returns:
            ndarray or tuple of ndarray: f = exp(-γt)[cosh(t√g) - α·t·sinhc(t√g)],
            shape (len(t),), and for nu = 1 also its partial derivatives, shape (3, len(t))

        Notes:
            Each partial derivative holds the other two of (α, g, γ) fixed.
            _chain_rule converts them to derivatives with respect to [r, α, g, y].
        """
        r, alpha, g, y = self.primitive_params
        gap_alpha, gap_g, _ = self._gaps(alpha, g, y)
        gamma = np.sqrt(alpha**2 + gap_alpha)
        C, tS = self._damped_cosh_sinhc(g, gamma, gap_g, t)
        f = C - alpha*tS
        if nu == 0:
            return f
        # ∂cosh(t√g)/∂g = (t²/2)·sinhc(t√g)
        dS_dg = self._damped_sinhc_g_derivative(g, gamma, gap_g, t)
        return f, np.array([-tS, t*tS/2 - alpha*t*dS_dg, -t*f])

    def _spectrum_partials(self, freq, nu):
        """Return Λ/r² and, for nu ≥ 1, its partial derivatives with respect to (α, G, γ),
        where G = γ² - g.

        Args:
            freq (ndarray): non-negative frequencies, as floats
            nu (int): 0 for the value only, 1 to add the gradient, 2 to add the Hessian

        Returns:
            ndarray or tuple of ndarray: f = N/M, shape (len(freq),), then for nu ≥ 1
            its gradient, shape (3, len(freq)), and for nu = 2 its Hessian,
            shape (3, 3, len(freq))

        Notes:
            N = (γ+α)ω² + (γ-α)G and M = (G-ω²)² + 4γ²ω² are sums of non-negative terms.
            G replaces g as a variable because the chain rule through g cancels near
            the boundary γ² = g.
        """
        r, alpha, g, y = self.primitive_params
        gap_alpha, G, _ = self._gaps(alpha, g, y)
        gamma = np.sqrt(alpha**2 + gap_alpha)
        gamma_plus_alpha, gamma_minus_alpha = self._gamma_pm(gamma, alpha, gap_alpha)
        w2 = freq**2
        N = gamma_plus_alpha*w2 + gamma_minus_alpha*G
        M = (G - w2)**2 + 4*gamma**2*w2
        f = N / M
        if nu == 0:
            return f
        one = np.ones_like(w2)
        dN = np.array([w2 - G, gamma_minus_alpha*one, w2 + G])
        dM = np.array([0*one, 2*(G - w2), 8*gamma*w2])
        # f·M = N, differentiated once and twice
        df = (dN - f*dM) / M
        if nu == 1:
            return f, df
        d2N = np.array([
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]])
        d2M = np.zeros((3, 3, len(w2)))
        d2M[1, 1] = 2.0
        d2M[2, 2] = 8*w2
        d2f = (d2N[:, :, None] - df[:, None]*dM[None, :] - dM[:, None]*df[None, :] - f*d2M) / M
        return f, df, d2f

    def _chain_rule(self, f, df, d2f=None, wrt_gap=False):
        """Return the gradient or the Hessian of r²·f with respect to [r, α, g, y].

        Args:
            f (ndarray): f on the grid, shape (ngrid,)
            df (ndarray): the partial derivatives of f with respect to v, shape (3, ngrid)
            d2f (ndarray, optional): the second partial derivatives of f, shape
                (3, 3, ngrid). If given, the Hessian is returned.
            wrt_gap (bool): v = (α, γ² - g, γ) if True, and v = (α, g, γ) otherwise

        Returns:
            ndarray: the gradient, shape (4, ngrid), or the Hessian, shape (4, 4, ngrid)
        """
        r, alpha, g, y = self.primitive_params
        # ∂v/∂(α, g, y)
        jac = np.eye(3)
        jac[2] = self._gamma(alpha, g, y, nu=1)
        if wrt_gap:
            jac[1] = self._gap_g(alpha, g, y, nu=1)
        grad = np.einsum('ki,kn->in', jac, df)
        if d2f is None:
            return np.concatenate([[2*r*f], r**2*grad])
        hess = np.einsum('ki,kln,lj->ijn', jac, d2f, jac)
        hess += self._gamma(alpha, g, y, nu=2)[:, :, None] * df[2]
        if wrt_gap:
            hess += self._gap_g(alpha, g, y, nu=2)[:, :, None] * df[1]
        ans = np.empty((4, 4, len(f)))
        ans[0, 0] = 2*f
        ans[0, 1:] = ans[1:, 0] = 2*r*grad
        ans[1:, 1:] = r**2*hess
        return ans

    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """Return the memory kernel K(t) = r² exp(-γ|t|)[cosh(t√g) - α|t| sinhc(t√g)].

        Args:
            time (float or array-like): times

        Returns:
            ndarray: shape (len(time),)
        """
        r = self.primitive_params[0]
        return r**2 * self._kernel_partials(self._abs_grid(time), nu=0)

    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """Return the gradient of the memory kernel with respect to [r, α, g, y].

        Args:
            time (float or array-like): times

        Returns:
            ndarray: shape (4, len(time))
        """
        return self._chain_rule(*self._kernel_partials(self._abs_grid(time), nu=1))

    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """Return the Hessian of the memory kernel with respect to [r, α, g, y].

        Args:
            time (float or array-like): times

        Returns:
            ndarray: shape (4, 4, len(time))

        Raises:
            NotImplementedError: always, until EMB-16 is resolved
        """
        raise NotImplementedError

    def spectrum_func(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """Return the spectrum Λ(ω) = r²[(γ+α)ω² + (γ-α)(γ²-g)] / [(γ²-g-ω²)² + 4γ²ω²].

        Args:
            frequency (float or array-like): frequencies

        Returns:
            ndarray: shape (len(frequency),)
        """
        r = self.primitive_params[0]
        return r**2 * self._spectrum_partials(self._abs_grid(frequency), nu=0)

    def spectrum_grad(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """Return the gradient of the spectrum with respect to [r, α, g, y].

        Args:
            frequency (float or array-like): frequencies

        Returns:
            ndarray: shape (4, len(frequency))
        """
        return self._chain_rule(*self._spectrum_partials(self._abs_grid(frequency), nu=1), wrt_gap=True)

    def spectrum_hess(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """Return the Hessian of the spectrum with respect to [r, α, g, y].

        Args:
            frequency (float or array-like): frequencies

        Returns:
            ndarray: shape (4, 4, len(frequency))
        """
        return self._chain_rule(*self._spectrum_partials(self._abs_grid(frequency), nu=2), wrt_gap=True)