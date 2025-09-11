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
    
    
    @abstractmethod
    def _compute_drift_matrix(self) -> npt.NDArray[np.floating]:
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
        return self._compute_drift_matrix()

    # Alias
    A = drift_matrix

    @abstractmethod
    def _compute_drift_matrix_gradient(self) -> npt.NDArray[np.floating]:
        """Calculate the gradient of the drift matrix for the current parametrization of the embedder

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
        return self._compute_drift_matrix_gradient()

    # Alias
    grad_A = drift_matrix_gradient

    
    
    @abstractmethod
    def kernel(self, time: ScalarArr, nu: Optional[int]=0) -> npt.NDArray[np.floating]:
        """
        Computes the memory kernel function for the embedding.

        Parameters
        ----------
        time : scalar or array-like
            The input time values for which to compute the kernel.
        nu : int (optional)
            Order of the derivative with respect to embedding parameters 
            (default is 0)

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
        if not isinstance(nu, int):
            raise TypeError(f"nu must be an integer, got {type(nu).__name__}")
    
    @abstractmethod
    def spectrum(self, frequency: ScalarArr, nu: Optional[int]=0) -> npt.NDArray[np.floating]:
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
        if not isinstance(nu, int):
            raise TypeError(f"nu must be an integer, got {type(nu).__name__}")
        
        
    def spectral_density(self, frequency: ScalarArr, nu: Optional[int]=0) -> npt.NDArray[np.floating]:
        """
        Computes the spectral density (spectrum multiplied by frequency) for the embedding
        at the given frequencies.

        Parameters
        ----------
        frequency : scalar or array-like
            Array of frequency values at which to evaluate the spectrum.

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
        return freq * self.spectrum(freq, nu=nu)
