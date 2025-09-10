#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   multi.py
@Time    :   2025/09/09 09:29:05
@Author  :   George Trenins
@Desc    :   Direct sum of several Markovian embedders
'''

from __future__ import print_function, division, absolute_import
from ._base import BaseEmbedder, ScalarArr
from typing import Iterable, Optional
import numpy as np
import numpy.typing as npt


class MultiEmbedder(BaseEmbedder):

    def __init__(self, embs: Iterable[BaseEmbedder], *args, **kwargs):
        self._embs = embs
        self._naux = sum([len(emb) for emb in self._embs])
        self._nparam = sum([emb.nparam for emb in embs])
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        return self._naux
    
    def _get_nparam(self) -> int:
        return self._nparam 
    
    def kernel(self, time: ScalarArr, nu: Optional[int] = 0) -> npt.NDArray[np.floating]:
        super().kernel(time, nu=nu)
        if nu == 0:
            ans = 0.0
            for emb in self._embs:
                ans = ans + emb.kernel(time, nu=0)
            return ans
        elif nu == 1:
            # Concatenate along parameter axis (axis=0)
            ans = np.concatenate([emb.kernel(time, nu=1) for emb in self._embs], axis=0)
        else:
            raise ValueError(f"Invalid derivative order nu = {nu} in call to kernel. Valid values are 0 and 1.")
            
    def spectrum(self, frequency: ScalarArr, nu: Optional[int] = 0) -> npt.NDArray[np.floating]:
        """Compute the spectrum (cosine transform) of the memory kernel.
        
        Parameters
        ----------
        frequency : scalar or array-like
            The frequencies at which to evaluate the spectrum
        nu : int, optional
            Order of derivative with respect to parameters (default: 0)
            
        Returns
        -------
        numpy.ndarray
            For nu=0: Sum of spectra from all embedders
            For nu=1: Concatenated parameter derivatives from all embedders
        """
        if nu == 0:
            ans = 0.0
            for emb in self._embs:
                ans = ans + emb.spectrum(frequency, nu=0)
            return ans
        elif nu == 1:
            # Concatenate along parameter axis (axis=0)
            ans = np.concatenate([emb.spectrum(frequency, nu=1) for emb in self._embs], axis=0)
            return ans
        else:
            raise ValueError(f"Invalid derivative order nu = {nu} in call to spectrum. Valid values are 0 and 1.")
        
    def _compute_drift_matrix(self) -> npt.NDArray[np.floating]:
        """Construct the drift matrix for the combined embedding system.
    
        Returns
        -------
        numpy.ndarray
            Square matrix of shape (N+1, N+1) where N is the total number of
            auxiliary variables from all embedders. The matrix has a block structure:
            [  0    w₁ᵀ   w₂ᵀ   ...  ]
            [ w₁    A₁     0    ...  ]
            [ w₂     0     A₂   ...  ]
            [ ...   ...   ...   ...  ]
            where wᵢ are coupling vectors and Aᵢ are auxiliary-variable drift matrices of individual embedders.
    
        Notes
        -----
        The drift matrix is constructed by placing each embedder's drift matrix
        in a block-diagonal form, with coupling terms in the first row/column.
        The first row/column contains all couplings to the system coordinate.
        """
        self._A[:] = 0.0
        ub = 1
        for emb in self._embs:
            emb_A = emb.drift_matrix
            naux = len(emb)
            lb = ub
            ub = ub + naux
            self._A[0,lb:ub] = emb_A[0,1:]
            self._A[lb:ub,0] = emb_A[1:,0]
            self._A[lb:ub, lb:ub] = emb_A[1:,1:]
        return self._A
    
    def _compute_drift_matrix_gradient(self) -> npt.NDArray[np.floating]:
        """Construct the gradient of the drift matrix for the combined system.

        Returns
        -------
        numpy.ndarray
            Array of shape (P, N+1, N+1) where:
            - P is the total number of parameters from all embedders
            - N is the total number of auxiliary variables
            Each slice [i,:,:] contains the derivative with respect to the i-th
            parameter and has block structure:
            [  0    ∂w₁ᵀ  ∂w₂ᵀ  ...  ]
            [∂w₁   ∂A₁    0     ...  ]
            [∂w₂    0    ∂A₂    ...  ]
            [...   ...    ...   ...  ]

        Notes
        -----
        Parameters are ordered according to the sequence of embedders,
        with all parameters from each embedder appearing consecutively.
        """
        self._grad_A[:] = 0.0
        param_offset = 0
        ub = 1
        
        for emb in self._embs:
            # Get gradient from current embedder
            emb_grad_A = emb.drift_matrix_gradient
            naux = len(emb)
            nparam = emb_grad_A.shape[0]
            
            # Set block boundaries
            lb = ub
            ub = ub + naux
            
            # Fill gradient blocks for each parameter
            grad_A_slice = self._grad_A[param_offset:param_offset+nparam]
            grad_A_slice[:, 0, lb:ub] = emb_grad_A[:, 0, 1:]
            grad_A_slice[:, lb:ub, 0] = emb_grad_A[:, 1:, 0]
            grad_A_slice[:, lb:ub, lb:ub] = emb_grad_A[:, 1:, 1:]
                
            param_offset += nparam
            
        return self._grad_A


