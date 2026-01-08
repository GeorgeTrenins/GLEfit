#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   properties.py
@Time    :   2025/09/11 13:15:04
@Author  :   George Trenins
@Desc    :   Features of the GLE (e.g. memory kernel)
'''


from __future__ import print_function, division, absolute_import
from ._base import BaseArrayProperty
from scipy.linalg import expm_frechet, expm
from typing import Union
import numpy as np
import numpy.typing as npt
from typing import Optional
from glefit.utils.linalg import mat_inv_vec


class MemoryKernel(BaseArrayProperty):
    """Memory friction kernel K(t) = θᵀ exp(-tA) θ
    where θ is the coupling vector and A is the auxiliary variable drift matrix.
    
    The gradient w.r.t. the drift matrix uses the matrix exponential Fréchet
    derivative for computational efficiency.
    
    Attributes:
        array: Time points where kernel is evaluated
        target: Target kernel values at each time
    """

    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Compute K(t) = θᵀ exp(-tA) θ at all time points.
        
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
        theta = Ap[0, 1:]
        A = Ap[1:,1:]
        return np.einsum(
            'ijk,j,k->i', 
            expm(-self.array[:,None,None] * A), 
            theta, theta)
    
    @staticmethod
    def _grad_thetaT_expA_theta(
            A: npt.NDArray[np.floating], 
            theta: npt.NDArray[np.floating],
            time: Union[float, npt.NDArray[np.floating]]
            ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Compute ∇_A(θᵀ exp(-τA) θ) using matrix exponential Fréchet derivative.
        
        Uses scipy.linalg.expm_frechet for the directional derivative in the 
        direction E = θθᵀ, ∂/∂A(θᵀ exp(-τA) θ) = −τ L_exp(-τA^T, θθᵀ)
        where L_exp is the Fréchet derivative of the matrix exponential.
        
        Args:
            A: Drift submatrix of shape (n, n)
            theta: Coupling vector of shape (n,)
            time: Scalar or 1D array of time values
            
        Returns:
            Tuple of (exp_matrix, frechet_derivative) where:
            - exp_matrix: exp(-tA^T) with shape (n, n) or (n_times, n, n)
            - frechet_derivative: -t * L_exp(-tA^T, θθᵀ) with same shape
            
        Raises:
            TypeError: If time has wrong dtype
            ValueError: If time array is not 1D
        """
        if isinstance(time, float):
            pass
        elif isinstance(time, np.ndarray):
            if not np.issubdtype(time.dtype, np.floating):
                raise TypeError(f"time must have a floating dtype; got {time.dtype!r}")
            if time.ndim != 1:
                raise ValueError(f"time must be 1-D; got shape {time.shape!r} with ndim={time.ndim}")
            time = time[:,None,None]
        else:
            raise TypeError(f"time must be float or a 1-D numpy array of floats; got {type(time).__name__}")
        
        # Outer product: θ θᵀ
        E = np.outer(theta, theta)
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
        
        time = self.array
        ans = np.zeros((len(time),) + Ap.shape)
        theta = Ap[0,1:]
        A_aux = Ap[1:,1:]
        
        # Compute derivatives w.r.t. the auxiliary variable drift matrix
        expA, grad_expA = self._grad_thetaT_expA_theta(A_aux, theta, time)
        
        # Fill in gradient components
        ans[:,0,1:] = np.einsum('j,tjk->tk', theta, expA)
        ans[:,1:,0] = -ans[:,0,1:]  # Skew-symmetric: gradient w.r.t. -θ is negative
        ans[:,1:,1:] = grad_expA
        
        return ans


class MemorySpectrum(BaseArrayProperty):
    """Spectral density Λ(ω) = θᵀA(A² + ω²I)⁻¹θ for GLE dynamics, computed via eigenvalue
    decomposition of the drift matrix.
    
    Attributes:
        array: Frequency points where spectrum is evaluated
        target: Target spectral density values
    """

    @staticmethod
    def _compute_spec_from_A(
        Ap: npt.NDArray[np.floating], 
        omega: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Compute spectral density via eigenvalue decomposition.
        
        Args:
            Ap: Full drift matrix of shape (n+1, n+1)
            omega: Frequency values of shape (n_freq,)
            
        Returns:
            Real-valued spectrum of shape (n_freq,)
        """
        theta = Ap[0, 1:]
        A = Ap[1:,1:]
        lamda, Q = np.linalg.eig(A)
        
        # Transform coupling vector to eigenbasis
        y = mat_inv_vec(Q, theta)  # y = Q⁻¹θ
        z = Q.T @ theta           
        
        # Sum over eigenvalues: Λ(ω) = Σᵢ yᵢzᵢλᵢ/(λᵢ² + ω²)
        return np.real(np.sum(
            y*z*lamda / (lamda**2 + omega[:,None]**2),
            axis=-1
        ))
        
    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Compute Λ(ω) at all frequency points.
        
        Args:
            x: Transformed parameters. If None, uses current embedding.
            
        Returns:
            Spectral density values at each frequency, shape (n_freq,)
        """
        if x is None:
            Ap = self.emb.A
        else:
            emb = self.emb
            Ap = emb.compute_drift_matrix(emb._inverse_map(x))
        return self._compute_spec_from_A(Ap, self.array)
    
    def grad_wrt_A(self, A: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of Λ(ω) w.r.t. full drift matrix: ∂Λ/∂A.
        
        
        Args:
            A: Full drift matrix of shape (n+1, n+1). If None, uses current embedding.
            
        Returns:
            Gradient array of shape (n_freq, n+1, n+1)
        """
        if A is None:
            Ap = self.emb.A
        else:
            Ap = np.copy(A)
        
        theta = Ap[0, 1:]
        A_aux = Ap[1:,1:]
        omega = self.array
        ans = np.zeros((len(omega),) + Ap.shape)
        
        # Eigenvalue decomposition
        lamda, Q = np.linalg.eig(A_aux)
        Qinv = np.linalg.inv(Q)
        a = mat_inv_vec(Q, theta)  # a = Q⁻¹θ
        b = Q.T @ theta
        M1 = 1/(lamda**2 + omega[:,None]**2)        # 1/(λᵢ² + ωⱼ²)
        y = np.einsum('ij,...j->...i', Q, M1*a)     # y = (A² + ω²I)⁻¹θ
        M2 = lamda*M1                               # λᵢ/(λᵢ² + ωⱼ²)
        v = np.einsum('ij,...j->...i', Q, M2*a)     # v = A(A² + ω²I)⁻¹θ
        u = np.einsum('...j,jk->...k', b*M2, Qinv)  # u = [θᵀA(A² + ω²I)⁻¹]ᵀ
        M3 = lamda*M2                               # λᵢ²/(λᵢ² + ωⱼ²)
        w = np.einsum('...j,jk->...k', b*M3, Qinv)  # w = [θᵀA²(A² + ω²I)⁻¹]ᵀ
        ans[:,0,1:] = np.real(v)                    # ∂Λ/∂θ
        ans[:,1:,0] = -ans[:,0,1:]                  # ∂Λ/∂(-θ) = -∂Λ/∂θ
        ans[:,1:,1:] = np.real(
            (theta - w)[:,:,None] * y[:,None,:] - u[:,:,None]*v[:,None,:]
        )                                           # ∂Λ/∂A_aux
        return ans