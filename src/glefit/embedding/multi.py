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
from itertools import accumulate
from typing import Iterable
import numpy as np
import numpy.typing as npt

class MultiEmbedder(BaseEmbedder):
    """Direct sum of component embedders.
    """

    def __init__(self, embedders: Iterable[BaseEmbedder], *args, **kwargs):
        if "mappers" in kwargs:
            raise TypeError(
                "MultiEmbedder takes no mappers: each component maps its own "
                "parameters")
        self._embs = embedders
        self._naux = sum([len(emb) for emb in self._embs])
        self._nparam = sum([emb.nparam for emb in embedders])
        self._ndof = sum([emb.ndof for emb in embedders])
        self._c_slices = self._slices([emb.nparam for emb in embedders])
        self._p_slices = self._slices([emb.ndof for emb in embedders])
        self._aux_slices = self._slices([len(emb) for emb in embedders], 1)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _slices(sizes: list[int], start: int = 0) -> list[slice]:
        """Consecutive slices of the given sizes, beginning at start."""
        bounds = list(accumulate(sizes, initial=start))
        return [slice(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]

    @classmethod
    def from_dict(
        cls,
        parameters: dict
    ) -> "MultiEmbedder":
        from glefit.embedding import EMBEDDER_MAP
        # no mappers: each component maps its own parameters
        cls._check_config_keys(parameters, ("embedders",), optional=())
        embedder_dict: list[dict] = parameters["embedders"]
        embedders: list[BaseEmbedder] = []
        for d in embedder_dict:
            EmbedderClass: BaseEmbedder = EMBEDDER_MAP[d["type"]]
            embedder: BaseEmbedder = EmbedderClass.from_dict(d["parameters"])
            embedders.append(embedder)
        return cls(embedders)

    def __len__(self) -> int:
        return self._naux

    @property
    def nparam(self) -> int:
        """Total number of conventional parameters of the components."""
        return self._nparam

    @property
    def ndof(self) -> int:
        """Total number of primitive parameters of the components."""
        return self._ndof

    def to_primitive(
        self,
        conventional_params: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Convert the 'conventional' embedding parameters to the primitive parameter set that removes the embedding degenracy
        """
        conventional_params = np.asarray(conventional_params, dtype=float)
        if conventional_params.shape != (self.nparam,):
           raise ValueError(f"The conventional parameters must be supplied as a length-{self.nparam} vector, instead got {conventional_params.shape = }")
        return np.concatenate([
            emb.to_primitive(conventional_params[c])
            for emb, c in zip(self._embs, self._c_slices)])

    def to_conventional(
        self,
        primitive_params: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Convert non-degenerate embedding parameters to conventional embedding variables
        """
        primitive_params = np.asarray(primitive_params, dtype=float)
        if primitive_params.shape != (self.ndof,):
           raise ValueError(f"The non-degenerate parameters must be supplied as a length-{self.ndof} vector, instead got {primitive_params.shape = }")
        return np.concatenate([
            emb.to_conventional(primitive_params[p])
            for emb, p in zip(self._embs, self._p_slices)])

    @property
    def conventional_params(self) -> npt.NDArray[np.floating]:
        """Conventional parameters of the components, concatenated."""
        return np.concatenate([emb.conventional_params for emb in self._embs])

    @conventional_params.setter
    def conventional_params(self, value: npt.ArrayLike) -> None:
        arr = self._checked(value, self.nparam, "conventional_params")
        self.primitive_params = self.to_primitive(arr)

    @property
    def x(self) -> npt.NDArray[np.floating]:
        """Mapped parameters of the components, concatenated."""
        return np.concatenate([emb.x for emb in self._embs])

    @x.setter
    def x(self, value: npt.ArrayLike) -> None:
        arr = self._checked(value, self.ndof, "x")
        for emb, p in zip(self._embs, self._p_slices):
            emb.x = arr[p]

    @property
    def primitive_params(self) -> npt.NDArray[np.floating]:
        """Primitive parameters of the components, concatenated."""
        return np.concatenate([emb.primitive_params for emb in self._embs])

    @primitive_params.setter
    def primitive_params(self, value: npt.ArrayLike) -> None:
        arr = self._checked(value, self.ndof, "primitive_params")
        # map every slice first, so that a value that a component's mapper
        # rejects raises before any component has changed
        self._forward_map(arr)
        for emb, p in zip(self._embs, self._p_slices):
            emb.primitive_params = arr[p]

    def _apply_to_params(
        self, func_name: str, arr: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
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
        ans = np.zeros_like(arr, dtype=float)
        for emb, p in zip(self._embs, self._p_slices):
            ans[p] = getattr(emb, func_name)(arr[p])
        return ans

    def _forward_map(self, params: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Transform optimizable (primitive) params so that inequality constraints are automatically imposed"""
        return self._apply_to_params('_forward_map', params)

    def _inverse_map(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Undo the inequality constraint-imposing mapping."""
        return self._apply_to_params('_inverse_map', x)

    def jac_px(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Returns J_{n} = ∂ p_n / ∂ x_n, where p_n is an embedding parameter and x_n is its transform."""
        return self._apply_to_params('jac_px', x)

    def hess_px(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Returns H_{n} = ∂² p_n / ∂ x_n², where p_n is an embedding parameter and x_n is its transform."""
        return self._apply_to_params('hess_px', x)

    @staticmethod
    def _place(
        out: npt.NDArray[np.floating],
        block: npt.NDArray[np.floating],
        aux: slice
    ) -> None:
        """Copy a component's drift matrix, or a stack of them, into out:
        its row 0, column 0 and auxiliary block go to row 0, column 0
        and the auxiliary block `aux` of out."""
        out[..., 0, aux] = block[..., 0, 1:]
        out[..., aux, 0] = block[..., 1:, 0]
        out[..., aux, aux] = block[..., 1:, 1:]

    def compute_drift_matrix(
        self,
        primitive_params: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Construct the drift matrix for the combined embedding system.

        Returns
        -------
        numpy.ndarray
            Square matrix of shape (N+1, N+1) where N is the total number of
            auxiliary variables from all embedders. The matrix has a block structure:
            [  0    w₁ᵀ   w₂ᵀ   ...  ]
            [-w₁    A₁     0    ...  ]
            [-w₂     0     A₂   ...  ]
            [ ...   ...   ...   ...  ]
            where wᵢ are coupling vectors and Aᵢ are auxiliary-variable drift matrices of individual embedders.

        Notes
        -----
        The drift matrix is constructed by placing each embedder's drift matrix
        in a block-diagonal form, with coupling terms in the first row/column.
        The first row/column contains all couplings to the system coordinate.
        """
        A = np.zeros((len(self) + 1, len(self) + 1))
        primitive_params = np.asarray(primitive_params)
        for emb, p, aux in zip(self._embs, self._p_slices, self._aux_slices):
            self._place(A, emb.compute_drift_matrix(primitive_params[p]), aux)
        return A

    def propagation_matrix(self) -> npt.NDArray[np.floating]:
        """The drift matrix for propagation, from those of the components.

        Returns
        -------
        numpy.ndarray
            Square matrix of shape (N+1, N+1), with the same block structure
            as compute_drift_matrix
        """
        Ap = np.zeros((len(self) + 1, len(self) + 1))
        for emb, aux in zip(self._embs, self._aux_slices):
            self._place(Ap, emb.propagation_matrix(), aux)
        return Ap

    def drift_matrix_param_grad(
        self,
        primitive_params: npt.ArrayLike
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
            [-∂w₁  ∂A₁    0     ...  ]
            [-∂w₂   0    ∂A₂    ...  ]
            [...   ...    ...   ...  ]

        Notes
        -----
        Parameters are ordered according to the sequence of embedders,
        with all parameters from each embedder appearing consecutively.
        """
        primitive_params = np.asarray(primitive_params)
        grad_A = np.zeros((self.ndof, len(self) + 1, len(self) + 1))
        for emb, p, aux in zip(self._embs, self._p_slices, self._aux_slices):
            self._place(grad_A[p],
                        emb.drift_matrix_param_grad(primitive_params[p]), aux)
        return grad_A

    def _block_diagonal(
        self, blocks: list[npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Hessian with the components' blocks, each of shape
        (ndof_i, ndof_i, ngrid), on its diagonal."""
        ans = np.zeros((self.ndof, self.ndof, blocks[-1].shape[-1]))
        for block, p in zip(blocks, self._p_slices):
            ans[p, p] = block
        return ans

    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        return sum((emb.kernel_func(time) for emb in self._embs), 0.0)

    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        # Concatenate along parameter axis (axis=0)
        return np.concatenate(
            [emb.kernel_grad(time) for emb in self._embs], axis=0)

    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        return self._block_diagonal(
            [emb.kernel_hess(time) for emb in self._embs])

    def spectrum_func(self, frequency: ScalarArr) -> npt.NDArray[np.floating]:
        return sum((emb.spectrum_func(frequency) for emb in self._embs), 0.0)

    def spectrum_grad(self, frequency: ScalarArr) -> npt.NDArray[np.floating]:
        return np.concatenate(
            [emb.spectrum_grad(frequency) for emb in self._embs], axis=0)

    def spectrum_hess(self, frequency: ScalarArr) -> npt.NDArray[np.floating]:
        return self._block_diagonal(
            [emb.spectrum_hess(frequency) for emb in self._embs])
