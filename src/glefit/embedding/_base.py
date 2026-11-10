#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2025/09/08 10:35:48
@Author  :   George Trenins
@Desc    :   This defines the base class for a Markovian embedding scheme, mapping a memory-kernel of a one-dimensional generalized Langevin equation onto an extended Markovian system.
'''


from __future__ import print_function, division, absolute_import
from typing import Callable, Union, Iterable, Optional
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

    # names of the primitive parameters, in order, for error messages
    _primitive_names: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        # subclasses pass on what their own arguments leave over, which
        # may only be `mappers`
        name = type(self).__name__
        if args:
            raise TypeError(
                f"{name} got {len(args)} unexpected positional "
                f"argument(s): {args}")
        unexpected = sorted(set(kwargs) - {"mappers"})
        if unexpected:
            raise TypeError(
                f"{name} got unexpected keyword arguments {unexpected}; "
                f"besides its parameters, it takes only 'mappers'")
        ndof = self.ndof
        self._primitive_params : np.ndarray = np.empty(ndof)
        self._x : np.ndarray = np.empty(ndof)
        self._mappers: list[BaseMapper] = kwargs.get("mappers")
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

    def to_dict(self) -> dict:
        """Return the config entry that rebuilds this embedder.

        It has the form of the `embedder` section of a config file, with
        built-in Python types only, so that
        `EMBEDDER_MAP[d["type"]].from_dict(d["parameters"])` builds an
        embedder with the same kernel and spectrum. Mappers are left out,
        so that embedder has the default mappers.

        Returns
        -------
        dict
            `type`, the name of the class in EMBEDDER_MAP, and
            `parameters`, the conventional parameters as from_dict takes
            them
        """
        from glefit.embedding import EMBEDDER_MAP
        names = {cls: name for name, cls in EMBEDDER_MAP.items()}
        return {"type": names[type(self)],
                "parameters": self._config_parameters()}

    @abstractmethod
    def _config_parameters(self) -> dict:
        """The keys of from_dict, as built-in Python types."""
        pass

    @classmethod
    def _check_config_keys(
        cls,
        parameters: dict,
        required: tuple,
        optional: tuple = ("mappers",)
    ) -> None:
        """Reject keys of from_dict parameters that the embedder does not take.

        Parameters
        ----------
        parameters : dict
            The parameters given to from_dict
        required : tuple of str
            The keys that from_dict needs
        optional : tuple of str
            Other keys it accepts. By default `mappers`, which only Python
            code can set, since configs cannot hold mapper objects.

        Raises
        ------
        TypeError
            If parameters has a key that is neither required nor optional
        """
        unexpected = sorted(set(parameters) - set(required) - set(optional))
        if unexpected:
            raise TypeError(
                f"{cls.__name__}.from_dict got unexpected keys {unexpected}; "
                f"it takes {list(required + optional)}")


    @abstractmethod
    def __len__(self) -> int:
        """Return the number of auxiliary variables in the embedding.
        
        Returns
        -------
        int:
            The number of auxiliary variables used in the embedding.
        """
        pass

    @property
    @abstractmethod
    def nparam(self) -> int:
        """Number of conventional parameters used to define the drift matrix.
        """

    @property
    def ndof(self) -> int:
        """Number of primitive (optimizable) parameters, by default nparam.
        """
        return self.nparam

    def to_primitive(
        self,
        conventional_params: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Convert the 'conventional' embedding parameters to the primitive parameter set that removes the embedding degenracy
        """
        # By default, assume that the primitive and conventional parameters are identical
        conventional_params = np.asarray(conventional_params, dtype=float)
        if conventional_params.shape != (self.nparam,):
           raise ValueError(f"The conventional parameters must be supplied as a length-{self.nparam} vector, instead got {conventional_params.shape = }")
        return np.copy(conventional_params)
    
    def to_conventional(
        self,
        primitive_params: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Convert non-degenerate embedding parameters to conventional embedding variables
        """
        # By default, assume that the primitive and conventional parameters are identical
        primitive_params = np.asarray(primitive_params, dtype=float)
        if primitive_params.shape != (self.ndof,):
           raise ValueError(f"The non-degenerate parameters must be supplied as a length-{self.ndof} vector, instead got {primitive_params.shape = }")
        return np.copy(primitive_params)

    @staticmethod
    def _checked(value: npt.ArrayLike, size: int, name: str) -> np.ndarray:
        """Return value as an array, after checking that its shape is
        (size,). name is the parameter set, for the error message."""
        arr = np.asarray(value)
        if arr.shape != (size,):
            raise ValueError(
                f"{name} must have shape {(size,)}, got {arr.shape}")
        return arr

    @property
    def primitive_params(self) -> npt.NDArray[np.floating]:
        """Internal primitive parameters."""
        return np.copy(self._primitive_params)

    @primitive_params.setter
    def primitive_params(self, value: npt.ArrayLike) -> None:
        """Set primitive parameters directly (bypasses conventional mapping)."""
        arr = self._checked(value, self.ndof, "primitive_params")
        # map before storing anything, so that a `value` that a mapper
        # rejects leaves the embedder unchanged
        x = self._forward_map(arr)
        self._primitive_params[:] = arr
        self._x[:] = x

    @property
    def x(self) -> npt.NDArray[np.floating]:
        """Transformed parameters, imposing inequality constraints"""
        return np.copy(self._x)
    
    @x.setter
    def x(self, value: npt.ArrayLike) -> None:
        arr = self._checked(value, self.ndof, "x")
        p = self._inverse_map(arr)
        self._x[:] = arr
        self._primitive_params[:] = p
        
    @property
    def conventional_params(self) -> npt.NDArray[np.floating]:
        """Conventional parameters, as in config files and trajectory output."""
        return self.to_conventional(self._primitive_params)
    
    @conventional_params.setter
    def conventional_params(self, value: npt.ArrayLike) -> None:
        """Set conventional parameters; store as primitive internally."""
        arr = self._checked(value, self.nparam, "conventional_params")
        self.primitive_params = self.to_primitive(arr)

    def _apply_mappers(
        self,
        method: str,
        values: npt.ArrayLike,
        unmapped: Callable[[np.ndarray], np.ndarray]
    ) -> npt.NDArray[np.floating]:
        """Apply `method` of each mapper to the matching entry of values.

        Without mappers (`_mappers` is None, as analysis/plot_objective.py
        sets), return unmapped(values) instead.
        """
        values = np.asarray(values, dtype=float)
        if self._mappers is None:
            return unmapped(values)
        ans = np.zeros_like(values)
        names = self._primitive_names
        for i, (mapper, v) in enumerate(zip(self._mappers, values)):
            mapper: BaseMapper
            try:
                ans[i] = getattr(mapper, method)(v)
            except ValueError as err:
                name = names[i] if i < len(names) else f"#{i}"
                raise ValueError(
                    f"{type(self).__name__} parameter {name}: {err}"
                ) from err
        return ans

    def _forward_map(
        self, primitive_params: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Transform optimizable params so that inequality constraints are automatically imposed"""
        return self._apply_mappers("forward", primitive_params, np.copy)

    def _inverse_map(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Undo the inequality constraint-imposing mapping.
        """
        return self._apply_mappers("inverse", x, np.copy)

    def jac_px(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Returns J_{n} = ∂ p_n / ∂ x_n, where p_n is an embedding parameter and x_n is its transform.
        """
        return self._apply_mappers("grad", x, np.ones_like)

    def hess_px(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Returns H_{n} = ∂² p_n / ∂ x_n², where p_n is a primitive embedding parameter and x_n is its transform.
        """
        return self._apply_mappers("hess", x, np.zeros_like)
    
    def grad_param_to_x(
            self, 
            grad: npt.ArrayLike, 
            x: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Convert a gradient with respect to the primitive parameters to a gradient w.r.t.
        mapped parameters imposing inequality constraints.
        """
        jac_px = self.jac_px(x)
        return np.einsum('i,i...->i...', jac_px, grad)
    
    def hess_param_to_x(
            self, 
            grad: npt.ArrayLike, 
            hess: npt.ArrayLike, 
            x: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Compute the Hessian w.r.t.mapped parameters from the gradient and Hessian w.r.t primitive parameters.
        """
        jac_px = self.jac_px(x)
        hess_px = self.hess_px(x)
        ans = np.einsum('i,j,ij...->ij...', jac_px, jac_px, hess)
        diag = np.einsum('ii...->i...', ans) # view of the diagonal
        diag += np.einsum('i,i...->i...', hess_px, grad)
        return ans

    @abstractmethod
    def compute_drift_matrix(
        self, 
        primitive_params: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Calculate the drift matrix for a given set of primitive parameters.

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
        return self.compute_drift_matrix(self.primitive_params)

    # Alias
    A = drift_matrix

    def propagation_matrix(self) -> npt.NDArray[np.floating]:
        """The drift matrix to use for propagating the extended system.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (n+1, n+1), by default the drift matrix.

        Notes
        -----
        Embedders whose parameters fix the kernel but not the drift matrix,
        such as gen2x2, override this with the form best suited to
        propagation.
        """
        return self.drift_matrix

    @abstractmethod
    def drift_matrix_param_grad(
        self, 
        primitive_params: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Calculate the gradient of the drift matrix with respect to the primitive parameters

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

    def drift_matrix_x_grad(
        self, x: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """Calculate the gradient of the drift matrix with respect to the mapped parameters

        Returns
        -------
        numpy.ndarray
            A 3D array of shape (k, n+1, n+1) where k is the number of parameters and n is the number of auxiliary variables. 
        """
        primitive_params = self._inverse_map(x)
        param_grad_A = self.drift_matrix_param_grad(primitive_params)
        return self.grad_param_to_x(param_grad_A, x)
        
    @property
    def drift_matrix_gradient(self) -> npt.NDArray[np.floating]:
        """The derivative of the drift matrix with respect to the mapped
        parameters x, at the current state of the embedder.
        
        Returns
        -------
        numpy.ndarray
            A 3D array of shape (k, n+1, n+1) where k is the number of parameters and n is the number of auxiliary variables. 
            
        Notes
        -----
        This is also accessible through the alias 'grad_A'.
        """
        return self.drift_matrix_x_grad(self.x)

    # Alias
    grad_A = drift_matrix_gradient

    @abstractmethod
    def kernel_func(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """Return K at each entry of the 1-D array time, shape (ngrid,)."""

    @abstractmethod
    def kernel_grad(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """Return ∂K/∂p, for the primitive parameters p, at each entry of
        the 1-D array time, shape (ndof, ngrid)."""

    @abstractmethod
    def kernel_hess(self, time: ScalarArr) -> npt.NDArray[np.floating]:
        """Return ∂²K/∂p², for the primitive parameters p, at each entry of
        the 1-D array time, shape (ndof, ndof, ngrid)."""

    def kernel(
        self,
        time: ScalarArr,
        nu: Optional[int]=0,
        mapped: Optional[bool]=False
    ) -> npt.NDArray[np.floating]:
        """Return the memory kernel K(t), or its derivatives.

        Parameters
        ----------
        time : float or array-like
            Times, as a scalar or a 1D array
        nu : int
            0 for K, 1 for its gradient and 2 for its Hessian
        mapped : bool
            Differentiate with respect to the mapped parameters x if True,
            and with respect to the primitive parameters otherwise

        Returns
        -------
        numpy.ndarray
            Shape (ngrid,) for nu = 0, (ndof, ngrid) for nu = 1 and
            (ndof, ndof, ngrid) for nu = 2. A scalar time gives ngrid = 1.

        Raises
        ------
        ValueError
            If time has more than one dimension, or nu is not 0, 1 or 2
        TypeError
            If nu is not an integer
        """
        return self._closed_form(
            "time", time, nu, mapped,
            self.kernel_func, self.kernel_grad, self.kernel_hess)

    @abstractmethod
    def spectrum_func(self, frequency: ScalarArr) -> npt.NDArray[np.floating]:
        """Return Λ at each entry of the 1-D array frequency, shape (ngrid,).
        """

    @abstractmethod
    def spectrum_grad(self, frequency: ScalarArr) -> npt.NDArray[np.floating]:
        """Return ∂Λ/∂p, for the primitive parameters p, at each entry of
        the 1-D array frequency, shape (ndof, ngrid)."""

    @abstractmethod
    def spectrum_hess(self, frequency: ScalarArr) -> npt.NDArray[np.floating]:
        """Return ∂²Λ/∂p², for the primitive parameters p, at each entry of
        the 1-D array frequency, shape (ndof, ndof, ngrid)."""

    def spectrum(
        self,
        frequency: ScalarArr,
        nu: Optional[int]=0,
        mapped: Optional[bool]=False
    ) -> npt.NDArray[np.floating]:
        """Return the spectrum Λ(ω), the cosine transform of K, or its
        derivatives.

        Parameters
        ----------
        frequency : float or array-like
            Frequencies, as a scalar or a 1D array
        nu : int
            0 for Λ, 1 for its gradient and 2 for its Hessian
        mapped : bool
            Differentiate with respect to the mapped parameters x if True,
            and with respect to the primitive parameters otherwise

        Returns
        -------
        numpy.ndarray
            Shape (ngrid,) for nu = 0, (ndof, ngrid) for nu = 1 and
            (ndof, ndof, ngrid) for nu = 2. A scalar frequency gives
            ngrid = 1.

        Raises
        ------
        ValueError
            If frequency has more than one dimension, or nu is not 0, 1 or 2
        TypeError
            If nu is not an integer
        """
        return self._closed_form(
            "frequency", frequency, nu, mapped,
            self.spectrum_func, self.spectrum_grad, self.spectrum_hess)

    def _closed_form(
        self,
        name: str,
        grid: ScalarArr,
        nu: int,
        mapped: bool,
        func: Callable[[np.ndarray], np.ndarray],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Callable[[np.ndarray], np.ndarray]
    ) -> npt.NDArray[np.floating]:
        """The body of kernel() and spectrum(), which pass their closed
        forms as func, grad and hess, and name their grid argument."""
        grid = np.atleast_1d(grid)
        if grid.ndim != 1:
            raise ValueError(
                f"Expecting `{name}` to be scalar or a 1D array, "
                f"instead got {name}.ndim = {grid.ndim}.")
        if not isinstance(nu, int):
            raise TypeError(f"nu must be an integer, got {type(nu).__name__}")
        if nu == 0:
            return func(grid)
        if nu == 1:
            g = grad(grid)
            return self.grad_param_to_x(g, self.x) if mapped else g
        if nu == 2:
            h = hess(grid)
            return self.hess_param_to_x(grad(grid), h, self.x) if mapped else h
        raise ValueError(
            f"Invalid value for nu = {nu}. Valid values are 0, 1, and 2.")

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
