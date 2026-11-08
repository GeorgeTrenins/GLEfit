#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   properties.py
@Time    :   2025/09/11 13:15:04
@Author  :   George Trenins
@Desc    :   Features of the GLE (e.g. memory kernel)
'''


from __future__ import print_function, division, absolute_import
from ._base import BaseProperty
from scipy.linalg import expm_frechet, expm
import numpy as np
import numpy.typing as npt
from typing import Optional


class MemoryKernel(BaseProperty):
    """Memory friction kernel K(t) = θᵀ exp(-|t|A) θ
    where θ is the coupling vector and A is the auxiliary variable drift matrix.

    θᵀ is read from the top row of Ap and θ from its negated left column,
    as in the dynamics. Every existing embedder makes the two equal.

    The gradient w.r.t. the drift matrix uses the matrix exponential Fréchet
    derivative for computational efficiency.
    
    Attributes:
        grid: Time points where kernel is evaluated
        target: Target kernel values at each time
    """

    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Compute K(t) = θᵀ exp(-|t|A) θ at all time points.
        
        Args:
            x: Transformed parameters. If None, uses current embedding.
            
        Returns:
            Kernel values at each time point, shape (n_times,)
        """
        if x is None:
            Ap = self.emb.A
        else:
            emb = self.emb
            Ap = emb.compute_drift_matrix(emb._inverse_map(x))
        row, col, A = Ap[0, 1:], -Ap[1:, 0], Ap[1:, 1:]
        return np.einsum(
            'ijk,j,k->i',
            expm(-np.abs(self.grid)[:,None,None] * A),
            row, col)

    @staticmethod
    def _grad_thetaT_expA_theta(
            A: npt.NDArray[np.floating],
            row: npt.NDArray[np.floating],
            time: npt.NDArray[np.floating],
            col: Optional[npt.NDArray[np.floating]] = None
            ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Compute ∇_A(θᵀ exp(-τA) θ) using matrix exponential Fréchet derivative.

        Uses scipy.linalg.expm_frechet for the directional derivative in the
        direction E = θθᵀ, ∂/∂A(θᵀ exp(-τA) θ) = −τ L_exp(-τA^T, θθᵀ)
        where L_exp is the Fréchet derivative of the matrix exponential.
        The first θ is row and the second is col.

        Args:
            A: Drift submatrix of shape (n, n)
            row: θᵀ, from the top row of Ap, shape (n,)
            time: 1-D float array of times, shape (n_times,)
            col: θ, from the negated left column of Ap, shape (n,).
                None means the same as row.

        Returns:
            Tuple of (exp_matrix, frechet_derivative) where:
            - exp_matrix: exp(-tA^T) with shape (n_times, n, n)
            - frechet_derivative: -t * L_exp(-tA^T, θθᵀ) with same shape
        """
        time = time[:, None, None]
        if col is None:
            col = row
        # Outer product: θ θᵀ
        E = np.outer(row, col)
        # Fréchet derivative: L_exp(-τA^T, E)
        expA, K = expm_frechet(-time * A.T, E, compute_expm=True)
        return expA, -time * K

    def grad_wrt_A(self, A: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of K(t) w.r.t. full drift matrix: ∂K/∂A.
        
        The drift matrix has the block structure:
            A = [  0   θᵀ ]
                [ -θ   A_aux ]
        
        where A_aux is the auxiliary variable drift matrix.
        
        Args:
            A: Full drift matrix of shape (n+1, n+1). If None, uses current embedding.
            
        Returns:
            Gradient array of shape (n_times, n+1, n+1)
        """
        if A is None:
            Ap = self.emb.A
        else:
            Ap = np.copy(A)
        
        time = np.abs(self.grid)
        ans = np.zeros((len(time),) + Ap.shape)
        row, col, A_aux = Ap[0, 1:], -Ap[1:, 0], Ap[1:, 1:]

        # Compute derivatives w.r.t. the auxiliary variable drift matrix
        expA, grad_expA = self._grad_thetaT_expA_theta(
            A_aux, row, time, col=col)

        # Fill in gradient components
        ans[:,0,1:] = np.einsum('j,tjk->tk', col, expA)    # exp(-tA)θ
        ans[:,1:,0] = -np.einsum('tkj,j->tk', expA, row)  # -exp(-tAᵀ)θ
        ans[:,1:,1:] = grad_expA
        
        return ans


class MemorySpectrum(BaseProperty):
    """Spectrum Λ(ω) = θᵀA(A² + ω²)⁻¹θ, the cosine transform of K(t). The
    spectral density is ω·Λ(ω).

    For real A, A(A² + ω²)⁻¹ = Re (A + iω)⁻¹, so Λ(ω) = Re θᵀ(A + iω)⁻¹θ. 
    Evaluation takes one complex linear solve per frequency and also works
    where A cannot be diagonalized, as for gen2x2 at critical damping. 
    It fails where A + iω is singular, and there Λ is not defined.

    θᵀ is read from the top row of Ap and θ from its negated left column,
    as in the dynamics. Every embedder makes the two equal.

    Attributes:
        grid: Frequencies where the spectrum is evaluated
        target: Target values of Λ at each frequency
    """

    @staticmethod
    def _solve_resolvent(
        Ap: npt.NDArray[np.floating],
        omega: npt.NDArray[np.floating],
        transpose: bool = False
    ) -> npt.NDArray[np.complexfloating]:
        """Return (A + iω)⁻¹θ, or (Aᵀ + iω)⁻¹θ, at each frequency.

        Args:
            Ap: full drift matrix [[0, θᵀ], [-θ, A]] of shape (n+1, n+1).
                θ is read from the negated left column, or with transpose
                from the top row.
            omega: frequencies of shape (n_freq,)
            transpose: if True, use Aᵀ instead of A

        Returns:
            Complex array of shape (n_freq, n)

        Raises:
            ValueError: if A + iω is singular at one of the frequencies
        """
        if transpose:
            theta, A = Ap[0, 1:], Ap[1:, 1:].T
        else:
            theta, A = -Ap[1:, 0], Ap[1:, 1:]
        M = A + 1j * omega[:, None, None] * np.eye(len(theta))
        try:
            return np.linalg.solve(M, theta)
        except np.linalg.LinAlgError:
            raise ValueError(
                "A + iω is singular at a grid frequency, "
                "and Λ(ω) is not defined there"
            ) from None

    @classmethod
    def _compute_spec_from_A(
        cls,
        Ap: npt.NDArray[np.floating],
        omega: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Return Λ(ω) = Re θᵀ(A + iω)⁻¹θ, shape (n_freq,), for the full
        drift matrix Ap of shape (n, n) and frequencies omega."""
        return np.real(cls._solve_resolvent(Ap, omega) @ Ap[0, 1:])

    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Compute Λ(ω) at all frequency points.

        Args:
            x: Transformed parameters. If None, uses current embedding.

        Returns:
            Λ at each frequency, shape (n_freq,)
        """
        if x is None:
            Ap = self.emb.A
        else:
            emb = self.emb
            Ap = emb.compute_drift_matrix(emb._inverse_map(x))
        return self._compute_spec_from_A(Ap, self.grid)
    
    def grad_wrt_A(self, A: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of Λ(ω) w.r.t. full drift matrix: ∂Λ/∂A.

        Args:
            A: Full drift matrix of shape (n+1, n+1). If None, uses current embedding.

        Returns:
            Gradient array of shape (n_freq, n+1, n+1)

        Notes:
            With R = (A + iω)⁻¹, Λ = Re(θᵀRθ) and ∂R/∂A_jk = -R E_jk R
            where E_jk is an (n+1, n+1) matrix unit with 1 (its only 
            non-zero entry) at (j, k). Hence, ∂Λ/∂θᵀ = Re(Rθ), 
            ∂Λ/∂θ = Re(Rᵀθ), and ∂Λ/∂A_jk = -Re[(Rᵀθ)_j (Rθ)_k].
        """
        if A is None:
            Ap = self.emb.A
        else:
            Ap = np.copy(A)

        omega = self.grid
        y = self._solve_resolvent(Ap, omega)                  # Rθ
        z = self._solve_resolvent(Ap, omega, transpose=True)  # Rᵀθ
        ans = np.zeros((len(omega),) + Ap.shape)
        ans[:, 0, 1:] = np.real(y)                   # ∂Λ/∂θᵀ, top row
        ans[:, 1:, 0] = -np.real(z)                  # ∂Λ/∂(-θ), left column
        ans[:, 1:, 1:] = -np.real(z[:, :, None] * y[:, None, :])  # ∂Λ/∂A
        return ans
