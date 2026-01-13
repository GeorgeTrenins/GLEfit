#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2025/09/11 10:38:05
@Author  :   George Trenins
@Desc    :   Base classes for merit functions used in GLE parameter optimization.

This module provides the framework for defining objective functions (merit functions)
that quantify the discrepancy between computed GLE properties and target values.

Key concepts:
- Deviation metrics: Quantify discrepancy between computed and target values
- Properties: Physical quantities computed from GLE embeddings (e.g., memory kernels, 
              spectra, or linear combinations of such properties)
- Parameters vs transformed parameters: Raw parameters may have restricted domains,
  which are mapped to unrestricted transformed parameters via constraint mappers
'''


from __future__ import print_function, division, absolute_import, annotations
from glefit.embedding import BaseEmbedder
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Union, TypeVar, Callable, Any, Optional, cast
from glefit.utils.numderiv import jacobian
import functools
from glefit.config.config_handler import ConfigError

# ---- Types ---------------------------------------------------------------

ScalarArr = Union[float, npt.NDArray[np.floating]]  # numpy array of floats
T = TypeVar("T", bound=ScalarArr)
F = TypeVar("F", bound=Callable[..., Any])

# ---- Validation helpers --------------------------------------------------

def _validate_scalararr(x: ScalarArr, *, name: str) -> None:
    """Ensure x is float or 1-D np.ndarray of floating dtype.
    
    Args:
        x: Value to validate (float or 1D array)
        name: Parameter name for error messages
        
    Raises:
        TypeError: If x is neither float nor ndarray, or has wrong dtype
        ValueError: If array is not 1-dimensional
    """
    if isinstance(x, float):
        return
    if isinstance(x, np.ndarray):
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"{name} must have a floating dtype; got {x.dtype!r}")
        if x.ndim != 1:
            raise ValueError(f"{name} must be 1-D; got shape {x.shape!r} with ndim={x.ndim}")
        return
    # If it's neither float nor ndarray, it's invalid.
    raise TypeError(f"{name} must be float or a 1-D numpy array of floats; got {type(x).__name__}")

# ---- Decorators -----------------------------------------------------------

def check_value_target(func: F) -> F:
    """Decorator validating (value, target) argument pairs in deviation metric methods.
    
    Automatically wraps deviation metric implementations to validate that value and target
    are both floats or 1D floating-point arrays with matching dimensionality.
    
    Args:
        func: Deviation metric method to wrap
        
    Returns:
        Wrapped function with input validation
    """
    @functools.wraps(func)
    def wrapper(self, value: ScalarArr, target: ScalarArr, *args, **kwargs):
        _validate_scalararr(value, name="value")
        _validate_scalararr(target, name="target")
        return func(self, value, target, *args, **kwargs)
    return cast(F, wrapper)

# ---- Base classes ----------------------------------------------------------

class BaseDeviation(ABC):
    """Abstract base class for deviation metrics.
    
    Deviation metrics quantify the discrepancy between a computed value and a target value.
    They must provide derivatives (gradient, Hessian) for gradient-based optimization.
    
    The metric handles both scalar and array inputs, with derivatives computed element-wise.
    Subclass implementations are automatically wrapped with input validation.
    
    Example:
        >>> metric = SquaredDeviation()
        >>> value = np.array([1.0, 2.0])
        >>> target = np.array([1.5, 1.8])
        >>> deviation = metric(value, target)  # Returns (0.5**2 + 0.2**2)
        >>> grad = metric.gradient(value, target)  # Returns 2*(value - target)
    """

    # Automatically wrap subclass implementations of __call__ and gradient, etc
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in ("__call__", "gradient", "hessian", "both", "all"):
            if name in cls.__dict__:               # only wrap methods defined on the subclass
                setattr(cls, name, check_value_target(cls.__dict__[name]))

    @abstractmethod
    def __call__(self, value: ScalarArr, target: ScalarArr, *args, **kwargs) -> ScalarArr:
        """Deviation between value and target.
        
        Args:
            value: Computed value(s) (float or 1D array)
            target: Target value(s) (float or 1D array, same shape as value)
            
        Returns:
            Deviation d(value, target), same shape as inputs
        """
        raise NotImplementedError
    
    @abstractmethod
    def gradient(self, value: T, target: T) -> T:
        """Gradient of deviation w.r.t. value: ∂d/∂value.
        
        Args:
            value: Computed value(s) (float or 1D array)
            target: Target value(s) (float or 1D array)
            
        Returns:
            Gradient array, same shape as inputs
        """
        raise NotImplementedError

    def both(self, value: T, target: T) -> tuple[ScalarArr, T]:
        """Deviation and its gradient in one call (more efficient).
        
        Args:
            value: Computed value(s)
            target: Target value(s)
            
        Returns:
            Tuple of (deviation, gradient)
        """
        return self.__call__(value, target), self.gradient(value, target)
    
    @abstractmethod
    def hessian(self, value: T, target: T) -> T:
        """Hessian of deviation w.r.t. value: ∂²d/∂value².
        
        For array inputs, returns diagonal Hessian (element-wise second derivatives).
        
        Args:
            value: Computed value(s)
            target: Target value(s)
            
        Returns:
            Hessian array, same shape as inputs
        """
        raise NotImplementedError
    
    def all(self, value: T, target: T) -> tuple[ScalarArr, T, T]:
        """Deviation, gradient, and Hessian in one call.
        
        Args:
            value: Computed value(s)
            target: Target value(s)
            
        Returns:
            Tuple of (deviation, gradient, hessian)
        """
        val, grad = self.both(value, target)
        return val, grad, self.hessian(value, target)


class SquaredDeviation(BaseDeviation):
    """Defined as the squared difference (element-wise for array properties):
    d = (value - target)².
    
    Provides analytical first and second derivatives for efficient optimization.
    Works element-wise for both scalar and array inputs.
    """

    def __call__(self, value: T, target: T, *args, **kwargs) -> T:
        """Compute d = (value - target)²."""
        d = value - target
        ans = d*d
        return ans
        
    def gradient(self, value: T, target: T) -> T:
        """Compute ∂d/∂value = 2(value - target)."""
        d = value - target
        return 2*d
    
    def both(self, value: T, target: T) -> tuple[T, T]:
        """Compute deviation and gradient simultaneously."""
        grad = value - target
        d2 = grad*grad
        grad *= 2
        return d2, grad
    
    def hessian(self, value: T, target: T) -> T:
        """Compute the hessian ∂²d/∂value² = 2."""
        return 2*np.ones_like(value)
    

METRICS = {
    "squared" : SquaredDeviation
}


class BaseProperty(ABC):
    """Abstract base class for GLE properties.
    
    A property is a physical quantity computed from the extended system's drift matrix A.
    This can be a scalar (e.g., a single kernel value) or an array (e.g., kernel over 
    a time grid or spectrum over a frequency grid).
    
    Properties support automatic differentiation via the chain rule:
        ∂property/∂x = ∂property/∂A · ∂A/∂x
    
    where x are the transformed parameters (with constraints imposed via mappers).
    
    Args:
        target: Target value(s) for this property
        emb: GLE embedding providing the drift matrix A
        metric: Name of deviation metric to use (e.g., "squared")
        weight: Weight factor for this property in combined objectives
        
    Attributes:
        deviation_metric: Instance of BaseDeviation for measuring errors
        emb: Reference to the embedding
        weight: Weight factor applied to the deviation and its derivatives
    """
    
    def __init__(
            self, 
            target: ScalarArr,
            emb: BaseEmbedder,
            metric: str,
            weight: Optional[ScalarArr] = 1.0,
            *args, **kwargs
    ) -> None:
        self._target = target
        self.emb = emb
        self._set_weight(weight)
        self.deviation_metric: BaseDeviation = METRICS[metric]()

    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        parameters: dict,
        data: dict,
        embedder: BaseEmbedder
    ) -> "BaseProperty":
        pass


    def _set_weight(self, weight: ScalarArr) -> None:
        """Set and validate weight factor(s).
        """
        raise NotImplementedError
    
    @property
    def target(self) -> ScalarArr:
        """Target value(s) for this property."""
        return self._target
    
    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> ScalarArr:
        """Compute the property value from GLE parameters.
        
        Args:
            x: Transformed parameters. If None, uses current embedding parameters.
               
        Returns:
            Current value of the property
        """
        raise NotImplementedError
    
    @property
    def value(self) -> ScalarArr:
        """Current value of the property."""
        return self.function()
    
    @property
    def distance(self) -> float:
        """Weighted distance of current property value from target.
        
        For scalar properties: distance = weight * deviation(value, target)
        For array properties: distance = sum(weights * deviation(values, targets))
        """
        raise NotImplementedError

    @abstractmethod
    def grad_wrt_A(self, A: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of property w.r.t. drift matrix A: ∂property/∂A.
        
        Args:
            A: Full drift matrix of shape (n+1, n+1). If None, uses current embedding.
            
        Returns:
            Gradient array. For scalar property: shape (n+1, n+1).
            For array property of length m: shape (m, n+1, n+1).
        """
        raise NotImplementedError

    def grad_wrt_params(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of property w.r.t. transformed parameters via chain rule: ∂property/∂x.
        
        Uses the chain rule:
            ∂property/∂x = ∂property/∂A · ∂A/∂x
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Gradient array. For scalar property: shape (p,).
            For array property of length m: shape (p, m), where p = number of parameters.
        """
        emb = self.emb
        if x is None:
            x = emb.x
            Ap = emb.A
        else:
            Ap = emb.compute_drift_matrix(emb._inverse_map(x))
        # Gradient of the property w.r.t drift matrix A
        grad_h_A: npt.NDArray[np.floating] = self.grad_wrt_A(A=Ap)
        # Gradient of drift matrix w.r.t. transformed parameters
        grad_A_params: npt.NDArray[np.floating] = emb.drift_matrix_x_grad(x)
        # Chain rule: multiply gradients
        ans = np.einsum('...ij,pij->p...', grad_h_A, grad_A_params)
        return ans
    
    def gradient(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of weighted distance to target w.r.t. transformed parameters.
        
        Uses chain rule: ∂(distance)/∂x = (∂distance/∂property) · (∂property/∂x)
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Gradient array of shape (p,), where p = number of parameters
        """
        raise NotImplementedError
    
    def both(self, x: Optional[npt.NDArray[np.floating]] = None) -> tuple[float, npt.NDArray[np.floating]]:
        """Distance and gradient in one call (more efficient).
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Tuple of (distance, gradient) where gradient has shape (p,)
        """
        raise NotImplementedError
    
    def gradhess_wrt_params(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Gradient and Hessian of property w.r.t. transformed parameters.
        
        The Hessian is computed numerically via finite differences (4th order) 
        of the gradient.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Tuple of (gradient, hessian) with shapes:
            - gradient: (p,) or (p, m) for array properties
            - hessian: (p, p) or (p, p, m) for array properties
            where p = number of parameters, m = property array length
        """
        raise NotImplementedError
    
    def hessian(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Hessian of weighted distance to target w.r.t. transformed parameters.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Hessian matrix of shape (p, p), where p = number of parameters
        """
        raise NotImplementedError

    def all(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Distance, gradient, and Hessian in one call.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Tuple of (distance, gradient, hessian) where:
            - distance: float
            - gradient: shape (p,)
            - hessian: shape (p, p)
        """
        raise NotImplementedError


class BaseScalarProperty(BaseProperty):
    """Base class for scalar-valued GLE properties.
    
    Scalar properties evaluate to a single value (e.g., memory kernel at t=0).
    
    Example:
        A property that fits K(t=0.5) to a target value.
    """

    @classmethod
    def from_dict(
        cls,
        parameters: dict,
        data: dict,
        embedder: BaseEmbedder
    ) -> "BaseScalarProperty":
        
        try:
            data_reference: str = parameters["data"]
        except KeyError:
            raise ConfigError(f"The configuration for the objective function of type {cls.__name__} is missing the entry 'data' that should reference the numerical data used to construct the objective.")
        try:
            _, target = data[data_reference]
        except KeyError:
            raise ConfigError(f"The data reference {data_reference} for the objective function of type {cls.__name__} is invalid.")
        try:
            metric: str = parameters["metric"]
        except KeyError:
            raise ConfigError(f"The configuration for the objective function of type {cls.__name__} is missing the entry 'metric' that determines how to quantify the deviation of the embedder from target.")
        weight = parameters.get("weight", default=1.0)
        return cls(target, embedder, metric, weight=weight)

    def _set_weight(self, weight: ScalarArr) -> None:
        """Validate and set scalar weight."""
        if np.ndim(weight) != 0:
            raise ValueError(f"Weight should be a scalar, instead got ndim = {np.ndim(weight)}")
        self.weight = float(weight)
    
    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> float:
        """Compute the scalar property value.
        
        Args:
            x: Transformed parameters. If None, uses current embedding.
            
        Returns:
            Scalar value of the property
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement a property function")
    
    @property
    def distance(self) -> float:
        """Weighted distance from target."""
        return self.weight * self.deviation_metric(self.value, self.target)
    
    def gradient(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of weighted distance to target w.r.t. parameters.
        
        Computes: weight · (∂d/∂property) · (∂property/∂x)
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Gradient vector of shape (p,)
        """
        # Derivative of distance metric w.r.t. property
        grad_deviation_h: float = self.deviation_metric.gradient(self.value, self.target)
        grad_deviation_h *= self.weight
        # Gradient of the property w.r.t. optimizable parameters
        grad_h_params: npt.NDArray[np.floating] = self.grad_wrt_params(x=x)
        # Chain rule
        return grad_deviation_h * grad_h_params
    
    def both(self, x: Optional[npt.NDArray[np.floating]] = None) -> tuple[float, npt.NDArray[np.floating]]:
        """Distance and gradient in one call.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Tuple of (distance, gradient)
        """
        deviation, grad_deviation_h = self.deviation_metric.both(self.function(x=x), self.target)
        deviation *= self.weight
        grad_deviation_h *= self.weight
        grad_h_params: npt.NDArray[np.floating] = self.grad_wrt_params(x=x)
        grad = grad_deviation_h * grad_h_params
        return deviation, grad
    
    def gradhess_wrt_params(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Gradient and numerical Hessian of property w.r.t. parameters.
        
        Hessian computed via 4th-order finite differences of gradient.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Tuple of (gradient, hessian) with shapes (p,) and (p, p)
        """
        emb = self.emb
        if x is None:
            x = emb.x
        grad = self.grad_wrt_params(x=x)
        
        # Compute Hessian via finite differences
        def fun(v):
            return self.grad_wrt_params(x=v)
        
        hess = jacobian(fun, x, order=4)
        # Symmetrize
        hess += hess.T
        hess /= 2
        return grad, hess
    
    def hessian(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> npt.NDArray[np.floating]:
        """Hessian of weighted distance to target w.r.t. parameters.
        
        Uses Hessian of property and distance metric derivatives.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Hessian matrix of shape (p, p)
        """
        _, grad_deviation_h, hess_deviation_h = self.deviation_metric.all(self.function(x=x), self.target)
        grad_deviation_h *= self.weight
        hess_deviation_h *= self.weight
        
        grad_h_params, hess_h_params = self.gradhess_wrt_params(x=x)
        ans = hess_deviation_h * grad_h_params[:,None] * grad_h_params[None,:]
        ans += grad_deviation_h * hess_h_params
        return ans

    def all(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Distance, gradient, and Hessian in one call.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Tuple of (distance, gradient, hessian)
        """
        deviation, grad_deviation_h, hess_deviation_h = self.deviation_metric.all(
            self.function(x=x), self.target)
        deviation *= self.weight
        grad_deviation_h *= self.weight
        hess_deviation_h *= self.weight
        
        grad_h_params, hess_h_params = self.gradhess_wrt_params(x=x)
        hess = hess_deviation_h * grad_h_params[:,None] * grad_h_params[None,:]
        hess += grad_deviation_h * hess_h_params
        grad = grad_deviation_h * grad_h_params
        return deviation, grad, hess

    
class BaseArrayProperty(BaseProperty):
    """Base class for array-valued GLE properties.
    
    Array properties evaluate to multiple values over a grid (e.g., memory kernel K(t)
    at multiple times, or spectrum Λ(ω) at multiple frequencies).
    
    Distance is computed as a weighted sum over the deviation at each grid points.
    
    Example:
        A property that fits K(t₁), K(t₂), ..., K(tₙ) to target values.
    
    Args:
        grid: Grid points (time or frequency) where property is evaluated
        target: Target values at each grid point
        emb: GLE embedding
        metric: Deviation metric name
        weight: Scalar or 1D array of weights for each grid point
    """
    
    def __init__(
            self, 
            grid: npt.NDArray[np.floating],
            target: npt.NDArray[np.floating],
            emb: BaseEmbedder,
            metric: str,
            weight: Optional[ScalarArr] = 1.0,
            *args, **kwargs):
        
        self._grid = np.copy(grid)
        super().__init__(target, emb, metric, weight, *args, **kwargs)

    @classmethod
    def from_dict(
        cls,
        parameters: dict,
        data: dict,
        embedder: BaseEmbedder
    ) -> "BaseScalarProperty":
        
        try:
            data_reference: str = parameters["data"]
        except KeyError:
            raise ConfigError(f"The configuration for the objective function of type {cls.__name__} is missing the entry 'data' that should reference the numerical data used to construct the objective.")
        try:
            grid, target = data[data_reference]
        except KeyError:
            raise ConfigError(f"The data reference {data_reference} for the objective function of type {cls.__name__} is invalid.")
        try:
            metric: str = parameters["metric"]
        except KeyError:
            raise ConfigError(f"The configuration for the objective function of type {cls.__name__} is missing the entry 'metric' that determines how to quantify the deviation of the embedder from target.")
        weight = parameters.get("weight", 1.0)
        return cls(grid, target, embedder, metric, weight=weight)

    def _set_weight(self, weight: ScalarArr) -> None:
        """Set and validate element-wise weights.
        
        Weights can be scalar (applied to all elements) or 1D array
        (individual weight per grid point).
        """
        if np.ndim(weight) not in {0, 1}:
            raise ValueError(f"Weight should be a scalar or 1D array, instead got ndim = {np.ndim(weight)}")
        try:
            self.weight = np.ones_like(self._grid) * weight
        except ValueError as e:
            raise ValueError("Weights could not be broadcast correctly") from e

    @property
    def grid(self) -> npt.NDArray[np.floating]:
        """Grid points where property is evaluated."""
        return np.copy(self._grid)
    
    @property
    def target(self) -> npt.NDArray[np.floating]:
        """Target values at each grid point."""
        return np.copy(self._target)
    
    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Compute property values at all grid points.
        
        Args:
            x: Transformed parameters. If None, uses current embedding.
            
        Returns:
            Array of property values with same shape as grid
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement a property function")
    
    @property
    def distance(self) -> float:
        """Weighted sum of element-wise deviations.
        
        Implements: sum(weights * deviation(values, targets))
        
        Note:
            Currently uses weighted sum (1-norm). Can be generalized later
            to support other aggregation methods (e.g., weighted L2-norm).
        """
        return np.sum(self.weight * self.deviation_metric(self.value, self.target))
    
    def gradient(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of distance sum w.r.t. parameters.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Gradient vector of shape (p,)
        """
        grad_deviation_h: npt.NDArray[np.floating] = self.deviation_metric.gradient(
            self.function(x=x), self.target)
        grad_deviation_h *= self.weight
        grad_h_params: npt.NDArray[np.floating] = self.grad_wrt_params(x=x)
        # Aggregate over array dimension using einsum
        ans = np.einsum('t,pt->p', grad_deviation_h, grad_h_params)
        return ans
    
    def both(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
        ) -> tuple[float, npt.NDArray[np.floating]]:
        """Distance and gradient in one call.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Tuple of (sum of distances, aggregated gradient)
        """
        deviation, grad_deviation_h = self.deviation_metric.both(self.function(x=x), self.target)
        deviation *= self.weight
        grad_deviation_h *= self.weight
        grad_h_params: npt.NDArray[np.floating] = self.grad_wrt_params(x=x)
        return np.sum(deviation), np.einsum('t,pt->p', grad_deviation_h, grad_h_params)

    def hessian(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Hessian of weighted distance sum w.r.t. parameters.
        
        Aggregates Hessian contributions across all grid points:
            ∂²(sum)/∂x² = sum(∂²d/∂property² · ∇property · ∇propertyᵀ + ∂d/∂property · ∇²property)
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Hessian matrix of shape (p, p)
        """
        _, grad_deviation_h, hess_deviation_h = self.deviation_metric.all(
            self.function(x=x), self.target)
        grad_deviation_h *= self.weight
        hess_deviation_h *= self.weight
        
        grad_h_params, hess_h_params = self.gradhess_wrt_params(x=x)
        
        ans = np.einsum('t,at,bt->ab', hess_deviation_h, grad_h_params, grad_h_params)
        ans += np.einsum('t,abt->ab', grad_deviation_h, hess_h_params)
        return ans

    def all(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Distance, gradient, and Hessian in one call.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Tuple of (sum of distances, gradient, Hessian)
        """
        deviation, grad_deviation_h, hess_deviation_h = self.deviation_metric.all(
            self.function(x=x), self.target)
        deviation *= self.weight
        grad_deviation_h *= self.weight
        hess_deviation_h *= self.weight
        
        grad_h_params, hess_h_params = self.gradhess_wrt_params(x=x)
        
        grad = np.einsum('t,at->a', grad_deviation_h, grad_h_params)
        hess = np.einsum('t,at,bt->ab', hess_deviation_h, grad_h_params, grad_h_params)
        hess += np.einsum('t,abt->ab', grad_deviation_h, hess_h_params)
        return np.sum(deviation), grad, hess
    
    def gradhess_wrt_params(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Gradient and numerical Hessian of property w.r.t. parameters.
        
        Hessian computed via 4th-order finite differences of gradient.
        
        Args:
            x: Transformed parameters. If None, uses current values.
            
        Returns:
            Tuple of (gradient, hessian) with shapes (p, m) and (p, p, m)
            where m is the property array length
        """
        emb = self.emb
        if x is None:
            x = emb.x
        grad = self.grad_wrt_params(x=x)
        grad_shape = grad.shape

        def fun(v):
            ans = self.grad_wrt_params(x=v).reshape(-1)
            return ans
        
        hess = jacobian(fun, x, order=4)
        hess.shape = grad_shape + (len(x),)
        # reorder axes
        hess = np.einsum('a...b->ab...', hess)
        # symmetrize
        hess += np.swapaxes(hess, 0, 1)
        hess /= 2
        return grad, hess
        