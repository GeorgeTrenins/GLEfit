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
    
    This class defines the interface for embedding schemes that map
    a memory kernel onto an extended Markovian system using auxiliary
    variables.
    """

    def __init__(self, *args, **kwargs):
        naux = self.__len__()
        self._A : np.ndarray = np.empty((naux+1, naux+1))
        nparam = self.nparam
        self._grad_A : np.ndarray = np.empty((nparam, naux+1, naux+1))
        self._params : np.ndarray = np.empty(nparam)
        self._x : np.ndarray = np.empty(nparam) 

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
        """Number of independent parameters used to define the drift matrix.
        """
        pass

    @property
    def nparam(self) -> int:
        """Number of independent parameters used to define the drift matrix.
        """
        return self._get_nparam()
    
    @property
    def params(self) -> npt.NDArray[np.floating]:
        """Return vector of optimizable parameters in the embedding."""
        return np.copy(self._params)
    
    @params.setter
    def params(self, value: npt.ArrayLike) -> None:
        """Set vector of optimizable parameters in the embedding."""
        cur = self._params
        arr = np.asarray(value)
        if arr.shape != cur.shape:
            raise ValueError(f"params must have shape {cur.shape}, got {arr.shape}")
        self._params[:] = arr
        self._x[:] = self._forward_map(arr)

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
        
    @abstractmethod
    def _forward_map(self, params: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform optimizable params so that inequality constraints are automatically imposed"""
        return np.copy(params)

    @abstractmethod
    def _inverse_map(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Undo the inequality constraint-imposing mapping.
        """
        return np.copy(x)

    @abstractmethod
    def jac_px(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Returns J_{n} = ∂ p_n / ∂ x_n, where p_n is an embedding parameter and x_n is its transform.
        """
        return np.ones_like(x)
    
    @abstractmethod
    def hess_px(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Returns H_{n} = ∂² p_n / ∂ x_n², where p_n is an embedding parameter and x_n is its transform.
        """
        return np.zeros_like(x)
    
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
        """Calculate the drift matrix for the current parametrization of the embedder

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
        return self.compute_drift_matrix(self._params)

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
        Computes the memory kernel function for the embedding.

        Parameters
        ----------
        time : scalar or array-like
            The input time values for which to compute the kernel.
        nu : int (optional)
            Order of the derivative with respect to embedding parameters 
            (default is 0)
        mapped : bool (optional)
            True if the derivatives are with respect to mapped parameters. 
            Default is False.

        Returns
        -------
        np.ndarray
            The computed kernel values as a NumPy array of floating point numbers.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        TypeError
            If `nu` is not an integer
        """
        time = np.abs(np.atleast_1d(time))
        if time.ndim != 1:
            raise ValueError(f"Expecting `time` to be scalar or a 1D array, "
                             f"instead got {time.ndim = }.")
        if not isinstance(nu, int):
            raise TypeError(f"nu must be an integer, got {type(nu).__name__}")
        if nu == 0:
            return self.kernel_func(time)
        elif nu == 1:
            grad = self.kernel_grad(time)
            if mapped:
                return self.grad_param_to_x(grad, self.x)
            else:
                return grad
        elif nu == 2:
            hess = self.kernel_hess(time)
            if mapped:
                grad = self.kernel_grad(time)
                return self.hess_param_to_x(grad, hess, self.x)
            else:
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
        Computes the spectrum (Fourier transform of the kernel) for the embedding
        at the given frequencies.

        Parameters
        ----------
        frequency : scalar or array-like
            Array of frequency values at which to evaluate the spectrum.
        nu : int (optional)
            Order of the derivative with respect to embedding parameters 
            (default is 0)
        mapped : bool (optional)
            True if the derivatives are with respect to mapped parameters. 
            Default is False.

        Returns
        -------
        numpy.ndarray
            Array of floating point values representing the spectrum at the specified frequencies.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        TypeError
            If `nu` is not an integer
        """
        frequency = np.abs(np.atleast_1d(frequency))
        if frequency.ndim != 1:
            raise ValueError(f"Expecting `time` to be scalar or a 1D array, "
                             f"instead got {frequency.ndim = }.")
        if not isinstance(nu, int):
            raise TypeError(f"nu must be an integer, got {type(nu).__name__}")
        if nu == 0:
            return self.spectrum_func(frequency)
        elif nu == 1:
            grad = self.spectrum_grad(frequency)
            if mapped:
                return self.grad_param_to_x(grad, self.x)
            else:
                return grad
        elif nu == 2:
            hess = self.spectrum_hess(frequency)
            if mapped:
                grad = self.spectrum_grad(frequency)
                return self.hess_param_to_x(grad, hess, self.x)
            else:
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
        Computes the spectral density (spectrum multiplied by frequency) for the embedding
        at the given frequencies.

        Parameters
        ----------
        frequency : scalar or array-like
            Array of frequency values at which to evaluate the spectrum.
        nu : int (optional)
            Order of the derivative with respect to embedding parameters 
            (default is 0)
        mapped : bool (optional)
            True if the derivatives are with respect to mapped parameters. 
            Default is False.

        Returns
        -------
        numpy.ndarray
            Array of floating point values representing the spectral density at the specified frequencies.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        freq = np.asarray(frequency)
        return freq * self.spectrum_func(freq, nu=nu, mapped=mapped)
