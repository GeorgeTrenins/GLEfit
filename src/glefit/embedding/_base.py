#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2025/09/08 10:35:48
@Author  :   George Trenins
@Desc    :   This defines the base class for a Markovian embedding scheme, mapping a memory-kernel of a one-dimensional generalized Langevin equation onto an extended Markovian system.
'''


from __future__ import print_function, division, absolute_import
from typing import Union, Iterable, Optional
from glefit.mappers import BaseMapper
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod


ScalarArr = Union[
    float,  # scalar float
    int,    # scalar int
    Iterable[Union[float, int]],  # iterable of numbers
    npt.NDArray[np.floating]      # 1D numpy array of floats
]


class BaseEmbedder(ABC):
    """Base class for Markovian embedding schemes.
    
    Roles of parameters:
    - Conventional params: user-facing (identical to primitive for simple embedders).
    - Primitive params: internal, non-degenerate parameters.
    - Mapped params x: one-to-one transforms of primitive params imposing constraints; used in optimization.
    """

    def __init__(self, *args, **kwargs):
        naux = self.__len__()
        self._A : np.ndarray = np.empty((naux+1, naux+1))
        nparam = self.nparam
        ndof = self.ndof
        self._grad_A : np.ndarray = np.empty((nparam, naux+1, naux+1))
        self._params : np.ndarray = np.empty(ndof)
        self._x : np.ndarray = np.empty(ndof)
        self._mappers: BaseMapper = kwargs.get("mappers")
        if self._mappers is not None:
            if len(self._mappers) != self.ndof:
                raise ValueError(f"The number of constraint mappers should match the number of optimizable parameters, instead got {ndof = } and {len(self._mappers) = }")
            
    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        parameters: dict
    ) -> "BaseEmbedder":
        pass


    @abstractmethod
    def __len__(self) -> int:
        """Return the number of auxiliary variables in the embedding.
        
        Returns
        -------
        int
            The number of auxiliary variables used in the embedding.
            
        Example
        -------
        >>> emb = ConcreteEmbedder(n=3)  # Some concrete implementation
        >>> len(emb)  # Returns 3
        3
        """
        pass

    @abstractmethod
    def _get_nparam(self) -> int:
        """Number of conventional parameters used to define the drift matrix.
        """
        pass

    @property
    def nparam(self) -> int:
        """Number of conventional parameters used to define the drift matrix.
        """
        return self._get_nparam()
    
    def _get_ndof(self) -> int:
        """Number of primitive (optimizable) parameters.
        """
        return self._get_nparam()

    @property
    def ndof(self) -> int:
        """Number of primitive (optimizable) parameters.
        """
        return self._get_ndof()
    
    @property
    def primitive_params(self) -> npt.NDArray[np.floating]:
        """Internal primitive parameters."""
        return np.copy(self._params)

    @primitive_params.setter
    def primitive_params(self, value: npt.ArrayLike) -> None:
        """Set primitive parameters directly (bypasses conventional mapping)."""
        cur = self._params
        arr = np.asarray(value)
        if arr.shape != cur.shape:
            raise ValueError(f"primitive_params must have shape {cur.shape}, got {arr.shape}")
        self._params[:] = arr
        self._x[:] = self._forward_map(arr)


    @property
    def params(self) -> npt.NDArray[np.floating]:
        """Conventional parameters [θ1, θ2, γ, δ, Ω]."""
        return self.to_conventional(self._params)
    
    @params.setter
    def params(self, value: npt.ArrayLike) -> None:
        """Set conventional parameters; store as primitive internally."""
        arr_conv = np.asarray(value)
        if arr_conv.shape != self.nparam:
            raise ValueError(f"params must have shape ({self.nparam},), got {arr_conv.shape}")
        prim = self.to_primitive(arr_conv)
        self._params[:] = prim
        self._x[:] = self._forward_map(prim)

    def to_primitive(
        self,
        cparams: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Convert the 'conventional' embedding parameters to the primitive parameter set that removes the embedding degenracy
        """
        if cparams.shape != (self.nparam,):
           raise ValueError(f"The conventional parameters must be supplied as a length-{self.nparam} vector, instead got {cparams.shape = }")
        return np.copy(cparams)
    
    def to_conventional(
        self,
        params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Convert non-degenerate embedding parameters to conventional embedding variables
        """
        if params.shape != (self.ndof,):
           raise ValueError(f"The non-degenerate parameters must be supplied as a length-{self.ndof} vector, instead got {params.shape = }")
        return np.copy(params)

    @property
    def x(self) -> npt.NDArray[np.floating]:
        """Transformed parameters, imposing inequality constraints"""
        return np.copy(self._x)
    
    @x.setter
    def x(self, value: npt.ArrayLike) -> None:
        cur = self._x
        arr = np.asarray(value)
        if arr.shape != cur.shape:
            raise ValueError(f"x must have shape {cur.shape}, got {arr.shape}")
        self._x[:] = arr
        self._params[:] = self._inverse_map(arr)
        
    def _forward_map(self, params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform optimizable params so that inequality constraints are automatically imposed"""
        if self._mappers is None:
            return np.copy(params)
        else:
            x = np.zeros_like(params)
            for i, (mapper,p) in enumerate(zip(self._mappers, params)):
                mapper: BaseMapper
                x[i] = mapper.forward(p)
            return x

    def _inverse_map(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Undo the inequality constraint-imposing mapping.
        """
        if self._mappers is None:
            return np.copy(x)
        else:
            p = np.zeros_like(x)
            for i, (mapper,xi) in enumerate(zip(self._mappers, x)):
                mapper: BaseMapper
                p[i] = mapper.inverse(xi)
            return p

    def jac_px(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Returns J_{n} = ∂ p_n / ∂ x_n, where p_n is an embedding parameter and x_n is its transform.
        """
        if self._mappers is None:
            return np.copy(x)
        else:
            p = np.zeros_like(x)
            for i, (mapper,xi) in enumerate(zip(self._mappers, x)):
                mapper: BaseMapper
                p[i] = mapper.grad(xi)
            return p
    
    def hess_px(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Returns H_{n} = ∂² p_n / ∂ x_n², where p_n is an embedding parameter and x_n is its transform.
        """
        if self._mappers is None:
            return np.copy(x)
        else:
            p = np.zeros_like(x)
            for i, (mapper,xi) in enumerate(zip(self._mappers, x)):
                mapper: BaseMapper
                p[i] = mapper.hess(xi)
            return p
    
    def grad_param_to_x(
            self, 
            grad: npt.NDArray[np.floating], 
            x: npt.NDArray[np.floating]
        ) -> npt.NDArray[np.floating]:
        """Convert a gradient with respect to the conventional parameters to a gradient w.r.t.
        mapped parameters imposing inequality constraints.
        """
        jac_px = self.jac_px(x)
        return np.einsum('i,i...->i...', jac_px, grad)
    
    def hess_param_to_x(
            self, 
            grad: npt.NDArray[np.floating], 
            hess: npt.NDArray[np.floating], 
            x: npt.NDArray[np.floating]
        ) -> npt.NDArray[np.floating]:
        """Compute the Hessian w.r.t.mapped parameters from the gradient and Hessian w.r.t conventional parameters.
        """
        jac_px = self.jac_px(x)
        hess_px = self.hess_px(x)
        ans = np.einsum('i,j,ij...->ij...', jac_px, jac_px, hess)
        diag = np.einsum('ii...->i...', ans) # view of the diagonal
        diag += np.einsum('i,i...->i...', hess_px, grad)
        return ans

    @abstractmethod
    def compute_drift_matrix(self, params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Calculate the drift matrix for a given set of conventional parameters.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (n+1, n+1) where n is the number of
            auxiliary variables. The matrix represents the drift term
            in the extended system's equations of motion.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        pass


    @property
    def drift_matrix(self) -> npt.NDArray[np.floating]:
        """The drift matrix of the extended Markovian system.
        
        Returns
        -------
        numpy.ndarray
            A 2D array of shape (n+1, n+1) where n is the number of
            auxiliary variables. The matrix represents the drift term
            in the extended system's equations of motion.
            
        Notes
        -----
        This matrix is also accessible through the alias 'A'.
        """
        return self.compute_drift_matrix(self.params)

    # Alias
    A = drift_matrix

    @abstractmethod
    def drift_matrix_param_grad(self, params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Calculate the gradient of the drift matrix for the current parametrization of the embedder
        (before the mapping)

        Returns
        -------
        numpy.ndarray
            A 3D array of shape (k, n+1, n+1) where k is the number of parameters and n is the number of auxiliary variables. 

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        pass

    def drift_matrix_x_grad(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Calculate the gradient of the drift matrix for the current parametrization of the embedder
        (after the mapping).

        Returns
        -------
        numpy.ndarray
            A 3D array of shape (k, n+1, n+1) where k is the number of parameters and n is the number of auxiliary variables. 
        """
        params = self._inverse_map(x)
        param_grad_A = self.drift_matrix_param_grad(params)
        return self.grad_param_to_x(param_grad_A, x)
        
    @property
    def drift_matrix_gradient(self) -> npt.NDArray[np.floating]:
        """The derivative of the drift matrix with respect to the parameters of the embedder.
        
        Returns
        -------
        numpy.ndarray
            A 3D array of shape (k, n+1, n+1) where k is the number of parameters and n is the number of auxiliary variables. 
            
        Notes
        -----
        This is also accessible through the alias 'grad_A'.
        """
        return self.drift_matrix_x_grad(self._x)

    # Alias
    grad_A = drift_matrix_gradient

    @abstractmethod
    def kernel_func(
        self, 
        time: ScalarArr, 
    )-> npt.NDArray[np.floating]:
        """
        Computes the memory kernel function for the embedding.

        Parameters
        ----------
        time : scalar or array-like
            The input time values for which to compute the kernel.

        Returns
        -------
        np.ndarray
            The computed kernel values as a NumPy array of floating point numbers.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def kernel_grad(
        self, 
        time: ScalarArr, 
    )-> npt.NDArray[np.floating]:
        """
        Computes the gradient of the memory kernel function w.r.t. conventional embedding parameters.

        Parameters
        ----------
        time : scalar or array-like
            The input time values for which to compute the kernel.

        Returns
        -------
        np.ndarray
            The computed kernel values as a NumPy array of floating point numbers.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def kernel_hess(
        self, 
        time: ScalarArr, 
    )-> npt.NDArray[np.floating]:
        """
        Computes the hessian of the memory kernel function w.r.t. conventional embedding parameters.

        Parameters
        ----------
        time : scalar or array-like
            The input time values for which to compute the kernel.

        Returns
        -------
        np.ndarray
            The computed kernel values as a NumPy array of floating point numbers.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        pass

    def kernel(
        self, 
        time: ScalarArr, 
        nu: Optional[int]=0, 
        mapped: Optional[bool]=False
    ) -> npt.NDArray[np.floating]:
        """
        Computes the memory kernel or its param derivatives.
        
        nu: 0 → kernel values; 1 → gradient w.r.t. primitive params; 2 → Hessian w.r.t. primitive params.
        mapped=True returns derivatives w.r.t. mapped params x.
        """
        time = np.atleast_1d(time)
        if time.ndim != 1:
            raise ValueError(f"Expecting `time` to be scalar or a 1D array, instead got {time.ndim = }.")
        if not isinstance(nu, int):
            raise TypeError(f"nu must be an integer, got {type(nu).__name__}")
        if nu == 0:
            return self.kernel_func(time)
        elif nu == 1:
            grad = self.kernel_grad(time)
            return self.grad_param_to_x(grad, self.x) if mapped else grad
        elif nu == 2:
            hess = self.kernel_hess(time)
            if mapped:
                grad = self.kernel_grad(time)
                return self.hess_param_to_x(grad, hess, self.x)
            return hess
        else:
            raise ValueError(f"Invalid value for nu = {nu}. Valid values are 0, 1, and 2.")
        

    @abstractmethod
    def spectrum_func(
        self, 
        frequency: ScalarArr, 
    )-> npt.NDArray[np.floating]:
        """
        Computes the spectrum for the embedding.

        Parameters
        ----------
        frequency : scalar or array-like
            The frequencies for which to compute the spectrum.

        Returns
        -------
        np.ndarray
            The computed spectrum values as a NumPy array of floating point numbers.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        pass

    
    @abstractmethod
    def spectrum_grad(
        self, 
        frequency: ScalarArr, 
    )-> npt.NDArray[np.floating]:
        """
        Computes the gradient of the spectrum w.r.t. conventional embedding parameters.

        Parameters
        ----------
        frequency : scalar or array-like
            The frequencies for which to compute the spectrum.

        Returns
        -------
        np.ndarray
            The computed spectrum values as a NumPy array of floating point numbers.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def spectrum_hess(
        self, 
        frequency: ScalarArr, 
    )-> npt.NDArray[np.floating]:
        """
        Computes the hessian of the spectrum w.r.t. conventional embedding parameters.

        Parameters
        ----------
        frequency : scalar or array-like
            The frequencies for which to compute the spectrum.

        Returns
        -------
        np.ndarray
            The computed kernel values as a NumPy array of floating point numbers.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        pass
        
    def spectrum(
        self, 
        frequency: ScalarArr, 
        nu: Optional[int]=0,
        mapped: Optional[bool]=False
    ) -> npt.NDArray[np.floating]:
        """
        Computes the spectrum (cosine transform of the kernel) or its derivatives.
        
        nu: 0 → spectrum values; 1 → gradient w.r.t. primitive params; 2 → Hessian w.r.t. primitive params.
        mapped=True returns derivatives w.r.t. mapped params x.
        """
        frequency = np.atleast_1d(frequency)
        if frequency.ndim != 1:
            raise ValueError(f"Expecting `time` to be scalar or a 1D array, instead got {frequency.ndim = }.")
        if not isinstance(nu, int):
            raise TypeError(f"nu must be an integer, got {type(nu).__name__}")
        if nu == 0:
            return self.spectrum_func(frequency)
        elif nu == 1:
            grad = self.spectrum_grad(frequency)
            return self.grad_param_to_x(grad, self.x) if mapped else grad
        elif nu == 2:
            hess = self.spectrum_hess(frequency)
            if mapped:
                grad = self.spectrum_grad(frequency)
                return self.hess_param_to_x(grad, hess, self.x)
            return hess
        else:
            raise ValueError(f"Invalid value for nu = {nu}. Valid values are 0, 1, and 2.")
        
    def spectral_density(
            self, 
            frequency: ScalarArr, 
            nu: Optional[int]=0,
            mapped: Optional[bool]=False
        ) -> npt.NDArray[np.floating]:
        """
        Computes the spectral density S(ω)=ω·K̂(ω) and its derivatives.
        """
        freq = np.asarray(frequency)
        return freq * self.spectrum(freq, nu=nu, mapped=mapped)
