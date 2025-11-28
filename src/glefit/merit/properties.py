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

    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> float:
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
            ):
        """
        Returns ∇_A (theta^T exp(-tau A) theta), same shape as A.
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
        #TODO: In principle, should check that theta is 1-D, and A is 2-D
        E = np.outer(theta, theta)           # θ θ^T
        expA, K = expm_frechet(-time * A.T, E, compute_expm=True)  # L_exp(-τA^T, E)
        return expA, -time * K

    def grad_wrt_A(self, A: Optional[npt.NDArray[np.floating]]=None) -> npt.NDArray[np.floating]:
        if A is None:
            Ap = self.emb.A
        else:
            Ap = np.copy(A)
        time = self.array
        ans = np.zeros((len(time),) + Ap.shape)
        theta = Ap[0,1:]
        A = Ap[1:,1:]
        expA, grad_expA = self._grad_thetaT_expA_theta(A, theta, time)
        ans[:,0,1:] = np.einsum('j,tjk->tk', theta, expA)
        ans[:,1:,0] = -ans[:,0,1:]
        ans[:,1:,1:] = grad_expA
        return ans



class MemorySpectrum(BaseArrayProperty):

    @staticmethod
    def _compute_spec_from_A(
        Ap: npt.NDArray[np.floating], omega: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        theta = Ap[0, 1:]
        A = Ap[1:,1:]
        lamda, Q = np.linalg.eig(A)
        # y = inv(Q) θ
        y = mat_inv_vec(Q, theta)
        # z = transpose(Q) θ
        z = Q.T @ theta
        return np.sum(
            y*z*lamda / (lamda**2 + omega[:,None]**2),
            axis=-1
        )
        
    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> float:
        if x is None:
            Ap = self.emb.A
        else:
            emb = self.emb
            Ap = emb.compute_drift_matrix(emb._inverse_map(x))
        return self._compute_spec_from_A(Ap, self.array)
    
    def grad_wrt_A(self, A: Optional[npt.NDArray[np.floating]]=None) -> npt.NDArray[np.floating]:
        if A is None:
            Ap = self.emb.A
        else:
            Ap = np.copy(A)
        #TODO from here