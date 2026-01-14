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

#TODO: overwrite to_conventional / to_primitive

class MultiEmbedder(BaseEmbedder):

    def __init__(self, embedders: Iterable[BaseEmbedder], *args, **kwargs):
        self._embs = embedders
        self._naux = sum([len(emb) for emb in self._embs])
        self._nparam = sum([emb.nparam for emb in embedders])
        self._ndof = sum([emb.ndof for emb in embedders])
        super().__init__(*args, **kwargs)
        # gather the parameters from component embedders
        _ = self.conventional_params
        _ = self.x

    @classmethod
    def from_dict(
        cls, 
        parameters: dict
    ) -> "MultiEmbedder":
        from glefit.embedding import EMBEDDER_MAP
        embedder_dict: list[dict] = parameters["embedders"]
        embedders: list[BaseEmbedder] = []
        for d in embedder_dict:
            EmbedderClass: BaseEmbedder = EMBEDDER_MAP[d["type"]]
            embedder: BaseEmbedder = EmbedderClass.from_dict(d["parameters"])
            embedders.append(embedder)
        return cls(embedders)

    def __len__(self) -> int:
        return self._naux
    
    def _get_nparam(self) -> int:
        return self._nparam
    
    def _get_ndof(self) -> int:
        return self._ndof
    
    @BaseEmbedder.conventional_params.getter
    def conventional_params(self):
        # ----- collect params from component embedders ---- #
        primitive_parameters = []
        for emb in self._embs:
            # set slice boundaries
            primitive_parameters.append(emb.conventional_params)
        return np.concatenate(primitive_parameters)
    
    @conventional_params.setter
    def conventional_params(self, value):
        arr = np.asarray(value)
        # ---- set params in the component embedders ---- #
        primitive_offset = 0
        conventional_offset = 0
        for emb in self._embs:
            ndof = emb.ndof
            nparams = emb.nparams
            # set slice boundaries
            p_slc = slice(primitive_offset, primitive_offset+ndof)
            c_slc = slice(conventional_offset, conventional_offset+nparams)
            emb.conventional_params = arr[c_slc]
            self._primitive_params[p_slc] = emb._primitive_params
            self._x[p_slc] = emb._x   
            primitive_offset += ndof
            conventional_offset += nparams

    @BaseEmbedder.x.getter
    def x(self):
        # ----- collect mapped params from component embedders ---- #
        param_offset = 0
        for emb in self._embs:
            ndof = emb.ndof
            # set slice boundaries
            slc = slice(param_offset, param_offset+ndof)
            self._x[slc] = emb.x
            param_offset += ndof
        return np.copy(self._x)
    
    @x.setter
    def x(self, value):
        arr = np.asarray(value)
        # ---- set mapped params in the component embedders ---- #
        param_offset = 0
        for emb in self._embs:
            ndof = emb.ndof
            # set slice boundaries
            slc = slice(param_offset, param_offset+ndof)
            emb.x = arr[slc] 
            self._primitive_params[slc] = emb._primitive_params
            self._x[slc] = emb._x   
            param_offset += ndof

    @BaseEmbedder.primitive_params.getter
    def primitive_params(self) -> npt.NDArray[np.floating]:
        # ----- collect primitive params from component embedders ---- #
        param_offset = 0
        for emb in self._embs:
            ndof = emb.ndof
            # set slice boundaries
            slc = slice(param_offset, param_offset+ndof)
            self._primitive_params[slc] = emb.primitive_params
            self._x[slc] = emb._x
            param_offset += ndof
        return np.copy(self._primitive_params)

    @primitive_params.setter
    def primitive_params(self, value: npt.ArrayLike) -> None:
        arr = np.asarray(value)
        # ---- set primitive params in the component embedders ---- #
        param_offset = 0
        for emb in self._embs:
            ndof = emb.ndof
            # set slice boundaries
            slc = slice(param_offset, param_offset+ndof)
            emb.primitive_params = arr[slc] 
            self._primitive_params[slc] = emb._primitive_params
            self._x[slc] = emb._x   
            param_offset += ndof

    def _apply_to_params(self, func_name: str, arr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Apply a parameter transformation function to the primitive parameters of every component embedder.
        
        Parameters
        ----------
        func_name : str
            Name of the transformation method to call on each embedder
        arr : numpy.ndarray
            Input array to transform
            
        Returns
        -------
        numpy.ndarray
            Array of transformed parameters
        """
        ans = np.zeros_like(arr)
        param_offset = 0
        for emb in self._embs:
            ndof = emb.ndof
            slc = slice(param_offset, param_offset + ndof)
            # Get the method by name and call it
            transform = getattr(emb, func_name)
            ans[slc] = transform(arr[slc])
            param_offset += ndof
        return ans

    def _forward_map(self, params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform optimizable (primitive) params so that inequality constraints are automatically imposed"""
        return self._apply_to_params('_forward_map', params)

    def _inverse_map(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Undo the inequality constraint-imposing mapping."""
        return self._apply_to_params('_inverse_map', x)

    def jac_px(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Returns J_{n} = ∂ p_n / ∂ x_n, where p_n is an embedding parameter and x_n is its transform."""
        return self._apply_to_params('jac_px', x)

    def hess_px(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Returns H_{n} = ∂² p_n / ∂ x_n², where p_n is an embedding parameter and x_n is its transform."""
        return self._apply_to_params('hess_px', x)
    
    def compute_drift_matrix(
            self, 
            params: npt.NDArray[np.floating]
        ) -> npt.NDArray[np.floating]:
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
        A = np.zeros_like(self._A)
        params = np.asarray(params)
        ub = 1
        j = 0
        for emb in self._embs:
            naux = len(emb)
            nparam = emb.nparam
            i = j
            j = i+nparam
            lb = ub
            ub = ub + naux
            emb_A = emb.compute_drift_matrix(params[i:j])
            A[0,lb:ub] = emb_A[0,1:]
            A[lb:ub,0] = emb_A[1:,0]
            A[lb:ub, lb:ub] = emb_A[1:,1:]
        return A
    
    def drift_matrix_param_grad(
            self, 
            params: npt.NDArray[np.floating]
        ) -> npt.NDArray[np.floating]:
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
        params = np.asarray(params)
        grad_A = np.zeros_like(self._grad_A)
        ub = 1
        j = 0
        for emb in self._embs:
            nparam = emb.nparam
            naux = len(emb)
            # Set block boundaries
            lb = ub
            ub = ub + naux
            nparam = emb.nparam
            i = j
            j = i+nparam
            emb_grad_A = emb.drift_matrix_param_grad(params[i:j])
            # Fill gradient blocks for each parameter
            grad_A_slice = grad_A[i:j]
            grad_A_slice[:, 0, lb:ub] = emb_grad_A[:, 0, 1:]
            grad_A_slice[:, lb:ub, 0] = emb_grad_A[:, 1:, 0]
            grad_A_slice[:, lb:ub, lb:ub] = emb_grad_A[:, 1:, 1:]
        return grad_A
    
    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        ans = 0.0
        for emb in self._embs:
            ans = ans + emb.kernel_func(time)
        return ans
    
    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        # Concatenate along parameter axis (axis=0)
        ans = np.concatenate([emb.kernel_grad(time) for emb in self._embs], axis=0)
        return ans
    
    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        blocks = []
        indices = [0]
        for emb in self._embs:
            block = emb.kernel_hess(time)
            blocks.append(block)
            indices.append(indices[-1]+len(block))
        ans = np.zeros((indices[-1], indices[-1], blocks[-1].shape[-1]))
        # second pass over embs to pack the output
        for i, block in enumerate(blocks):
            lb, ub = indices[i:i+2]
            ans[lb:ub,lb:ub] = block
        return ans

    def spectrum_func(self, frequency: ScalarArr) -> npt.NDArray[np.floating]:
        ans = 0.0
        for emb in self._embs:
            ans = ans + emb.spectrum_func(frequency)
        return ans
    
    def spectrum_grad(self, frequency: ScalarArr) -> npt.NDArray[np.floating]:
        ans = np.concatenate([emb.spectrum_grad(frequency) for emb in self._embs], axis=0)
        return ans
    
    def spectrum_hess(self, frequency: ScalarArr) -> npt.NDArray[np.floating]:
        blocks = []
        indices = [0]
        for emb in self._embs:
            block = emb.spectrum_hess(frequency) 
            blocks.append(block)
            indices.append(indices[-1]+len(block))
        ans = np.zeros((indices[-1], indices[-1], blocks[-1].shape[-1]))
        # second pass over embs to pack the output
        for i, block in enumerate(blocks):
            lb, ub = indices[i:i+2]
            ans[lb:ub,lb:ub] = block
        return ans
