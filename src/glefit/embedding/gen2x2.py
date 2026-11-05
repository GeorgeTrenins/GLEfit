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

# With G = γ² - Ω² > 0, the kernel is evaluated by _kernel_trig for √G·t up to
# this value, and by _kernel_exponentials above it (see _kernel_partials)
_SINHC_SWITCH = 1.0

# Keys of earlier versions, and what replaced them
_REMOVED_KEYS = {
    "sigma": "γ is now fitted directly, without Softmax or Softabs",
    "threshold": "γ is now fitted directly, without Softmax or Softabs",
    "Omega": "the antisymmetric coupling between s₁ and s₂ is now 'zeta'",
}

class TwoAuxEmbedder(BaseEmbedder):

    def __len__(self) -> int:
        return 2

    def _get_nparam(self) -> int:
        """Number of conventional parameters [θ₁, θ₂, γ, δ, ζ].
        """
        return 5

    def _get_ndof(self) -> int:
        """Number of primitive parameters [r, γ, Ω, φ].
        """
        return 4

    def __init__(
        self,
        theta: npt.NDArray[np.floating],
        gamma: float,
        delta: float,
        zeta: float,
        *args,
        **kwargs
    ) -> None:
        """Initialise the embedder from conventional parameters.

        Args:
            theta (array-like): the couplings [θ₁, θ₂]
            gamma (float): the mean decay rate γ
            delta (float): half the difference δ between the decay rates of s₁
                and s₂
            zeta (float): the antisymmetric coupling ζ between s₁ and s₂
            *args, **kwargs: passed to BaseEmbedder; `mappers` replaces the
                default mappers

        Raises:
            TypeError: if gamma, delta or zeta is not a float or int, or if
                `sigma`, `threshold` or `Omega` is passed, which belonged to
                earlier versions
            ValueError: if theta does not have two entries, or if the
                embedding is not admissible (see to_primitive)

        Notes:
            The drift matrix is

                     (  0    θ₁    θ₂ )
                Ap = ( -θ₁  γ+δ    ζ  )
                     ( -θ₂  -ζ    γ-δ )

            and the kernel is
                K(t) = r² exp(-γ|t|)[cosh(t√G) - γ|t| cos(2φ) sinhc(t√G)],
            with
                G = γ² - Ω²,    r² = θ₁² + θ₂²,    Ω² = γ² - δ² + ζ²,
                cos 2φ = δ(θ₁² - θ₂²)/(γr²).
            Different conventional parameters can give the same kernel, so the
            embedder fits the primitive parameters [r, γ, Ω, φ] and builds the
            drift matrix in the maximal gauge δ = γ, where θ = r(cos φ, sin φ)
            and ζ = Ω. K and Λ depend only on r², Ω² and cos 2φ, so any real
            r, Ω and φ with γ > 0 give a valid embedding.
        """
        self._reject_removed_keys(kwargs)
        for variable, symbol in zip([gamma, delta, zeta], "γδζ"):
            if not isinstance(variable, (int, float)):
                raise TypeError(
                    f"{symbol} must be a float, got {type(variable).__name__}")
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
            raise ValueError(
                f"θ must be a length-2 vector, instead got {theta.shape = }")
        kwargs.setdefault(
            "mappers", [
                IdentityMapper(),   # r ∈ R
                LowerBoundMapper(), # γ > 0
                IdentityMapper(),   # Ω ∈ R
                IdentityMapper()    # φ ∈ R
        ])
        super().__init__(*args, **kwargs)
        self.conventional_params = np.concatenate([theta, [gamma, delta, zeta]])

    @classmethod
    def from_dict(
        cls,
        parameters: dict
    ) -> "TwoAuxEmbedder":
        """Return an embedder built from a config dictionary.

        Args:
            parameters (dict): `theta`, `gamma`, `delta` and `zeta`, plus
                keyword arguments for the constructor. It is not modified.

        Returns:
            TwoAuxEmbedder: the new embedder

        Raises:
            TypeError: if a key of an earlier version, such as `Omega`, is
                present
        """
        param_copy = copy.deepcopy(parameters)
        # before the pops, so that `Omega` in place of `zeta` is not a bare
        # KeyError
        cls._reject_removed_keys(param_copy)
        theta = param_copy.pop("theta")
        gamma = param_copy.pop("gamma")
        delta = param_copy.pop("delta")
        zeta = param_copy.pop("zeta")
        return cls(theta, gamma, delta, zeta, **param_copy)

    @staticmethod
    def _reject_removed_keys(parameters: dict) -> None:
        """Raise TypeError for a key of an earlier version of the embedder.

        Args:
            parameters (dict): keyword arguments or config parameters

        Raises:
            TypeError: if a key of _REMOVED_KEYS is present, naming what
                replaced it
        """
        for key, reason in _REMOVED_KEYS.items():
            if key in parameters:
                raise TypeError(
                    f"TwoAuxEmbedder no longer takes '{key}': {reason}")

    @staticmethod
    def from_harmonic(
        weight: float, omega: float, gamma: float
    ) -> dict[str, float]:
        """Return the conventional parameters of a damped harmonic mode.

        Args:
            weight (float): the weight of the mode, as from WAM_discretization
            omega (float): the frequency of the mode
            gamma (float): its damping rate

        Returns:
            dict for from_dict: `theta` = [sqrt(2·weight/π), 0], `gamma`,
                `delta` = 0 and `zeta` = omega

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
        """Return an embedder with given couplings and auxiliary drift matrix.

        Args:
            theta (array-like): the couplings [θ₁, θ₂]
            A (array-like): the 2×2 drift matrix of the auxiliary variables
            **kwargs: passed to the constructor

        Returns:
            TwoAuxEmbedder: an embedder with the same kernel, in the maximal
                gauge

        Raises:
            ValueError: if theta or A has the wrong shape, if the symmetric
                part of A has a negative eigenvalue, or if the kernel is not
                admissible

        Notes:
            θ and A are first rotated so that the symmetric part of A is
            diagonal, which gives the form [[γ+δ, ζ], [-ζ, γ-δ]].
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
            raise ValueError(
                f"θ must be a length-2 vector, instead got {theta.shape = }")
        A = np.asarray(A, dtype=float)
        if A.shape != (2,2):
            raise ValueError(
                f"A must be a 2-by-2 array, instead got {A.shape = }")
        theta, A = cls.rotate_matrix(theta, A)
        gamma = (A[0,0] + A[1,1]) / 2
        delta = (A[0,0] - A[1,1]) / 2
        zeta = A[0,1]
        return cls(theta, gamma, delta, zeta, **kwargs)

    def to_primitive(
        self,
        conventional_params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Return the primitive parameters [r, γ, Ω, φ] of conventional ones.

        Args:
            conventional_params (array-like): [θ₁, θ₂, γ, δ, ζ]

        Returns:
            ndarray: [r, γ, Ω, φ], with r, Ω ≥ 0 and 0 ≤ φ ≤ π/2

        Raises:
            ValueError: if conventional_params does not have five entries, or
                if the embedding is not admissible: an entry is not finite,
                θ = 0, γ = 0 or γ < |δ|
        """
        conventional_params = np.asarray(conventional_params, dtype=float)
        if conventional_params.shape != (5,):
            raise ValueError(
                f"The conventional parameters must be supplied as a length-5 "
                f"vector, instead got {conventional_params.shape = }")
        theta1, theta2, gamma, delta, zeta = conventional_params
        if not (np.all(np.isfinite(conventional_params))
                and (theta1 != 0 or theta2 != 0)
                and gamma > 0 and gamma >= abs(delta)):
            raise ValueError(
                f"The embedding must be admissible, with finite parameters, "
                f"θ ≠ 0, γ > 0 and γ ≥ |δ|, instead got θ = [{theta1}, "
                f"{theta2}], γ = {gamma}, δ = {delta}, ζ = {zeta}")
        th2 = conventional_params[:2]**2
        return self._primitive_from_invariants(
            r2=th2[0] + th2[1],
            gamma=gamma,
            det=(gamma - delta)*(gamma + delta) + zeta**2,
            quad=(gamma + delta)*th2[0] + (gamma - delta)*th2[1],
            quad_adj=(gamma - delta)*th2[0] + (gamma + delta)*th2[1])

    @staticmethod
    def _primitive_from_invariants(r2, gamma, det, quad, quad_adj):
        """Return [r, γ, Ω, φ] from the invariants of θ and a 2×2 block A.

        Args:
            r2 (float): |θ|²
            gamma (float): tr A / 2
            det (float): det A
            quad (float): θᵀAθ
            quad_adj (float): θᵀ adj(A) θ = |θ|² tr A - θᵀAθ

        Returns:
            ndarray: [r, γ, Ω, φ], with r² = |θ|², Ω² = det A,
            cos²φ = θᵀAθ/(|θ|² tr A) and sin²φ = θᵀ adj(A) θ/(|θ|² tr A),
            so that 0 ≤ φ ≤ π/2

        Notes:
            K and Λ depend on θ and A only through these invariants, which do
            not change when s₁ and s₂ are rotated. All arguments must be
            non-negative.
        """
        return np.array([np.sqrt(r2), gamma, np.sqrt(det),
                         np.arctan2(np.sqrt(quad_adj), np.sqrt(quad))])

    def to_conventional(
        self,
        primitive_params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Return the conventional parameters in the maximal gauge δ = γ.

        Args:
            primitive_params (array-like): [r, γ, Ω, φ]

        Returns:
            ndarray: [θ₁, θ₂, γ, δ, ζ] = [r cos φ, r sin φ, γ, γ, Ω]

        Raises:
            ValueError: if primitive_params does not have four entries
        """
        primitive_params = np.asarray(primitive_params, dtype=float)
        if primitive_params.shape != (4,):
            raise ValueError(
                f"The primitive parameters must be supplied as a length-4 "
                f"vector, instead got {primitive_params.shape = }")
        r, gamma, Omega, phi = primitive_params
        return np.array([r*np.cos(phi), r*np.sin(phi), gamma, gamma, Omega])

    @staticmethod
    def rotate_matrix(
        theta: npt.NDArray[np.floating],
        A: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Return θ and A rotated so that the symmetric part of A is diagonal.

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
                f"The auxiliary variable drift matrix must be "
                f"positive-definite. "
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
            primitive_params (array-like): [r, γ, Ω, φ]

        Returns:
            ndarray: [
                [0,        r cos φ, r sin φ], 
                [-r cos φ,      2γ,       Ω],
                [-r sin φ,      -Ω,       0]]
        """
        theta1, theta2, gamma, delta, zeta = self.to_conventional(
            primitive_params)
        return np.array([
            [0.0,     theta1,        theta2       ],
            [-theta1, gamma + delta, zeta         ],
            [-theta2, -zeta,         gamma - delta]])

    def drift_matrix_param_grad(
        self,
        primitive_params: npt.NDArray
    ) -> npt.NDArray[np.floating]:
        """Return the derivatives of Ap with respect to primitive parameters.

        Args:
            primitive_params (array-like): [r, γ, Ω, φ]

        Returns:
            ndarray: ∂Ap/∂p, shape (4, 3, 3)
        """
        r, gamma, Omega, phi = np.asarray(primitive_params, dtype=float)
        grad = np.zeros((4, 3, 3))
        # θ = r(cos φ, sin φ) fills the top row, and -θ the left column
        grad[0, 0, 1:] = np.cos(phi), np.sin(phi)
        grad[3, 0, 1:] = -r*np.sin(phi), r*np.cos(phi)
        grad[:, 1:, 0] = -grad[:, 0, 1:]
        # A = [[2γ, Ω], [-Ω, 0]]
        grad[1, 1, 1] = 2.0
        grad[2, 1, 2], grad[2, 2, 1] = 1.0, -1.0
        return grad

    @staticmethod
    def _abs_grid(grid: ScalarArr) -> npt.NDArray[np.floating]:
        """Return |grid| for a time or frequency grid, as a 1-d float array.

        Args:
            grid (float or array-like): times or frequencies

        Returns:
            ndarray: shape (ngrid,)

        Notes:
            The closed forms are even in t and ω. The cast to float stops
            integer input from being truncated.
        """
        return np.abs(np.atleast_1d(np.asarray(grid, dtype=float)))

    def _kernel_partials(self, t):
        """Return f = K/r² and its derivatives with respect to (γ, Ω, φ).

        Args:
            t (ndarray): non-negative times, as floats

        Returns:
            tuple of ndarray: f, shape (len(t),), and its gradient,
                shape (3, len(t))

        Notes:
            f = exp(-γt)[cosh(t√G) - γt cos(2φ) sinhc(t√G)] with G = γ² - Ω².
            It is evaluated by _kernel_trig for G ≤ 0 and for
            √G·t ≤ _SINHC_SWITCH, and by _kernel_exponentials otherwise.
        """
        _, gamma, Omega, phi = self.primitive_params
        # as a product, which is accurate near critical damping Ω ≈ γ
        G = (gamma - abs(Omega))*(gamma + abs(Omega))
        trig = (t*np.sqrt(G) <= _SINHC_SWITCH if G > 0
                else np.full(t.shape, True))
        f, df = np.empty_like(t), np.empty((3, len(t)))
        for mask, form in (
            (trig, self._kernel_trig),
            (~trig, self._kernel_exponentials)
        ):
            if np.any(mask):
                f[mask], df[:, mask] = form(t[mask], gamma, Omega, phi, G)
        return f, df

    @staticmethod
    def _kernel_trig(t, gamma, Omega, phi, G):
        """Return f = exp(-γt)[cosh(t√G) - γt cos(2φ) sinhc(t√G)]
        and its derivatives with respect to (γ, Ω, φ), through G.

        Args:
            t (ndarray): non-negative times
            gamma, Omega, phi (float): the primitive parameters γ, Ω and φ
            G (float): γ² - Ω²

        Returns:
            tuple of ndarray: f, shape (len(t),), and its gradient,
                shape (3, len(t))

        Notes:
            With s = sinc and z = t√(-G), which is real for G ≤ 0,
            cosh(t√G) = cos z, sinhc(t√G) = s(z) and
            ∂sinhc(t√G)/∂G = (t²/4)[s(z) + s''(z)].
            For imaginary z, s + s'' loses precision through cancellation,
            hence _SINHC_SWITCH to _kernel_exponentials.
        """
        z = t*np.emath.sqrt(-G)
        decay = np.exp(-gamma*t)
        # exp(-γt) cosh(t√G)
        C = decay*np.real(np.cos(z))
        # exp(-γt) t sinhc(t√G)
        tS = decay*t*np.real(sinc(z))
        # exp(-γt) t ∂sinhc(t√G)/∂G
        tdS = decay*t**3/4 * np.real(sinc(z) + sinc(z, nu=2))
        c2 = np.cos(2*phi)
        f = C - gamma*c2*tS
        # partial derivatives of f w.r.t. (gamma, G, cos 2φ)
        df_gamma, df_G, df_c2 = -t*f - c2*tS, t*tS/2 - gamma*c2*tdS, -gamma*tS
        # chain rule to get partial derivatives w.r.t. (γ, Ω, φ)
        return f, np.array([
            df_gamma + 2*gamma*df_G,
            -2*Omega*df_G,
            -2*np.sin(2*phi)*df_c2
        ])

    @staticmethod
    def _kernel_exponentials(t, gamma, Omega, phi, G):
        """Return f = exp(-γt)[cosh(t√G) - γt cos(2φ) sinhc(t√G)]
        and its derivatives with respect to (γ, Ω, φ), for G > 0.

        Args:
            t (ndarray): non-negative times
            gamma, Omega, phi (float): the primitive parameters γ, Ω and φ
            G (float): γ² - Ω² > 0

        Returns:
            tuple of ndarray: f, shape (len(t),), and its gradient,
                shape (3, len(t))

        Notes:
            With κ = √G, λ± = γ ± κ, and a± = (1 ± γ cos(2φ)/κ)/2,
                     f = a₋ exp(-λ₋t) + a₊ exp(-λ₊t)

            For Ω ≪ γ, both γ - κ and the chain rule through G
            lose precision through cancellation, so λ₋ is computed as
            Ω²/(γ + κ), and the derivatives are taken through λ± and a±.

            For Ω = γ|sin 2φ| (the Debye limit), κ = γ|cos 2φ|, and
            a₋ (if cos 2φ > 0) or a₊ (if cos 2φ < 0) vanishes. That amplitude
            is computed as (γ² sin²2φ - Ω²)/[2κ(κ + γ|cos 2φ|)], which stays
            accurate near this limit, and the other as one minus it.
        """
        kappa = np.sqrt(G)
        c2, s2 = np.cos(2*phi), np.sin(2*phi)
        rates = np.array([Omega**2/(gamma + kappa), gamma + kappa])
        vanishing = ( # (γ² sin²2φ - Ω²)/[2κ(κ + γ|cos 2φ|)]
            (gamma*abs(s2) - abs(Omega))*(gamma*abs(s2) + abs(Omega)) /
            (2*kappa*(kappa + gamma*abs(c2)))
        )
        amplitudes = np.array([vanishing, 1 - vanishing] if c2 >= 0 else
                              [1 - vanishing, vanishing])
        exponentials = np.exp(-np.outer(rates, t))
        f = amplitudes @ exponentials
        # Derivatives of the decay rates: one row each for
        # (λ₋, λ₊), one column each for (∂/∂γ, ∂/∂Ω, ∂/∂φ).
        d_rates = np.array([[-rates[0], Omega, 0.0],
                            [rates[1], -Omega, 0.0]]) / kappa
        # Derivatives of the amplitudes: one row each for
        # (a₋, a₊), one column each for (∂/∂γ, ∂/∂Ω, ∂/∂φ).
        d_minus = np.array(
            [c2*Omega**2, -c2*gamma*Omega, 2*gamma*G*s2]) / (2*kappa**3)
        d_amplitudes = np.array([d_minus, -d_minus])
        # Product rule on f =  a₊ exp(-λ₊a₊t) + a₋ exp(-λ₋a₋t):
        #   ∂f = [∂a₊ - a₊ t ∂λ₊] exp(-λ₊t) + [∂a₋ - a₋ t ∂λ₋] exp(-λ₋t)
        df = (np.einsum('ik,in->kn', d_amplitudes, exponentials)
              - t*np.einsum('i,ik,in->kn', amplitudes, d_rates, exponentials))
        return f, df

    def _spectrum_partials(self, freq):
        """Return f = Λ/r² and its first two derivatives w.r.t. (γ, Ω, φ).

        Args:
            freq (ndarray): non-negative frequencies, as floats

        Returns:
            tuple of ndarray: f, shape (len(freq),), its gradient,
                shape (3, len(freq)), and its Hessian, shape (3, 3, len(freq))

        Notes:
            f = N/M with N = 2γ(ω² cos²φ + Ω² sin²φ) and
            M = (ω² - Ω²)² + 4γ²ω², which are sums of non-negative terms.
            At Ω = ω = 0, f is 0/0 and undefined: K then tends to r² sin²φ,
            and its integral diverges unless sin φ = 0.
        """
        r, gamma, Omega, phi = self.primitive_params
        w2 = freq**2
        one = np.ones_like(w2)
        sin2, sin_2phi, cos_2phi = np.sin(phi)**2, np.sin(2*phi), np.cos(2*phi)
        q = w2*np.cos(phi)**2 + Omega**2*sin2
        gap = (Omega - freq)*(Omega + freq)
        N = 2*gamma*q
        M = gap**2 + 4*gamma**2*w2
        f = N / M
        # f·M = N, differentiated once and twice
        dN = np.array([2*q, 4*gamma*Omega*sin2*one, 2*gamma*gap*sin_2phi])
        dM = np.array([8*gamma*w2, 4*Omega*gap, 0*one])
        df = (dN - f*dM) / M
        d2N = np.zeros((3, 3, len(w2)))
        d2N[0, 1] = d2N[1, 0] = 4*Omega*sin2
        d2N[0, 2] = d2N[2, 0] = 2*gap*sin_2phi
        d2N[1, 1] = 4*gamma*sin2
        d2N[1, 2] = d2N[2, 1] = 4*gamma*Omega*sin_2phi
        d2N[2, 2] = 4*gamma*gap*cos_2phi
        d2M = np.zeros((3, 3, len(w2)))
        d2M[0, 0] = 8*w2
        d2M[1, 1] = 12*Omega**2 - 4*w2
        d2f = (d2N - df[:, None]*dM[None, :] - dM[:, None]*df[None, :]
               - f*d2M) / M
        return f, df, d2f

    def _with_r(self, f, df, d2f=None):
        """Return the gradient or the Hessian of r²·f w.r.t. [r, γ, Ω, φ]
        where f is a function of (γ, Ω, φ) but not of r.

        Args:
            f (ndarray): f on the grid, shape (ngrid,)
            df (ndarray): its derivatives with respect to (γ, Ω, φ),
                shape (3, ngrid)
            d2f (ndarray, optional): its second derivatives,
                shape (3, 3, ngrid). If given, the Hessian is returned.

        Returns:
            ndarray: the gradient, shape (4, ngrid), or the Hessian,
                shape (4, 4, ngrid)
        """
        r = self.primitive_params[0]
        if d2f is None:
            return np.concatenate([[2*r*f], r**2*df])
        ans = np.empty((4, 4, len(f)))
        ans[0, 0] = 2*f
        ans[0, 1:] = ans[1:, 0] = 2*r*df
        ans[1:, 1:] = r**2*d2f
        return ans

    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """Return the memory kernel K(t).

        Args:
            time (float or array-like): times

        Returns:
            ndarray: shape (len(time),)

        Notes:
            K(t) = r² exp(-γ|t|)[cosh(t√G) - γ|t| cos(2φ) sinhc(t√G)],
            with G = γ² - Ω².
        """
        r = self.primitive_params[0]
        f, _ = self._kernel_partials(self._abs_grid(time))
        return r**2 * f

    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """Return the gradient of the memory kernel w.r.t. [r, γ, Ω, φ].

        Args:
            time (float or array-like): times

        Returns:
            ndarray: shape (4, len(time))
        """
        return self._with_r(*self._kernel_partials(self._abs_grid(time)))

    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """Return the Hessian of the memory kernel w.r.t. [r, γ, Ω, φ].

        Args:
            time (float or array-like): times

        Returns:
            ndarray: shape (4, 4, len(time))

        Raises:
            NotImplementedError: always, until EMB-16 is resolved
        """
        raise NotImplementedError

    def spectrum_func(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """Return the spectrum Λ(ω).

        Args:
            frequency (float or array-like): frequencies

        Returns:
            ndarray: shape (len(frequency),)

        Notes:
            Λ(ω) = 2γr²(ω² cos²φ + Ω² sin²φ) / [(ω² - Ω²)² + 4γ²ω²].
        """
        r = self.primitive_params[0]
        f, _, _ = self._spectrum_partials(self._abs_grid(frequency))
        return r**2 * f

    def spectrum_grad(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """Return the gradient of the spectrum w.r.t. [r, γ, Ω, φ].

        Args:
            frequency (float or array-like): frequencies

        Returns:
            ndarray: shape (4, len(frequency))
        """
        f, df, _ = self._spectrum_partials(self._abs_grid(frequency))
        return self._with_r(f, df)

    def spectrum_hess(self, frequency: ScalarArr)-> npt.NDArray[np.floating]:
        """Return the Hessian of the spectrum w.r.t. [r, γ, Ω, φ].

        Args:
            frequency (float or array-like): frequencies

        Returns:
            ndarray: shape (4, 4, len(frequency))
        """
        return self._with_r(*self._spectrum_partials(self._abs_grid(frequency)))
