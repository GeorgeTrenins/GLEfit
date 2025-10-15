#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2025/09/11 10:38:05
@Author  :   George Trenins
@Desc    :   Base class for merit functions.
'''


from __future__ import print_function, division, absolute_import, annotations
from glefit.embedding import BaseEmbedder
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Union, TypeVar, Callable, Any, Optional, cast
from glefit.utils.numderiv import jacobian
import functools

# ---- Types ---------------------------------------------------------------

ScalarArr = Union[float, npt.NDArray[np.floating]]  # numpy array of floats
T = TypeVar("T", bound=ScalarArr)
F = TypeVar("F", bound=Callable[..., Any])

# ---- Validation helpers --------------------------------------------------

def _validate_scalararr(x: ScalarArr, *, name: str) -> None:
    """Ensure x is float or 1-D np.ndarray of floating dtype."""
    if isinstance(x, float):
        return
    if isinstance(x, np.ndarray):
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"{name} must have a floating dtype; got {x.dtype!r}")
        if x.ndim != 1:
            raise ValueError(f"{name} must be 1-D; got shape {x.shape!r} with ndim={x.ndim}")
        return
    # If it’s neither float nor ndarray, it’s invalid.
    raise TypeError(f"{name} must be float or a 1-D numpy array of floats; got {type(x).__name__}")

# ---- Decorators -----------------------------------------------------------

def check_value_target(func: F) -> F:
    """
    Decorator for methods whose first two arguments after `self` are
    (value, target). Validates type and dimensionality.
    """
    @functools.wraps(func)
    def wrapper(self, value: ScalarArr, target: ScalarArr, *args, **kwargs):
        _validate_scalararr(value, name="value")
        _validate_scalararr(target, name="target")
        return func(self, value, target, *args, **kwargs)
    return cast(F, wrapper)

# ---- Base class ----------------------------------------------------------

class BaseDistance(ABC):

    # Automatically wrap subclass implementations of __call__ and gradient, etc
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in ("__call__", "gradient", "hessian", "both", "all"):
            if name in cls.__dict__:               # only wrap methods defined on the subclass
                setattr(cls, name, check_value_target(cls.__dict__[name]))

    @abstractmethod
    def __call__(self, value: ScalarArr, target: ScalarArr, *args, **kwargs) -> float:
        """Distance between the current value of a function and its target"""
        raise NotImplementedError
    
    @abstractmethod
    def gradient(self, value: T, target: T) -> T:
        """Derivative of the distance metric with respect to the current value"""
        raise NotImplementedError

    def both(self, value: T, target: T) -> tuple[float, T]:
        """Distance between the current value of a function and its target and its derivative 
        with respect to the current value"""
        return self.__call__(value, target), self.gradient(value, target)
    
    @abstractmethod
    def hessian(self, value: T, target: T) -> T:
        """Hessian of the distance metric with respect to the current value"""
        raise NotImplementedError
    
    def all(self, value: T, target: T) -> tuple[float, T, T]:
        """Distance between the current value of a function and its target, its derivative 
        and its hessian with respect to the current value"""
        val, grad = self.both(value, target)
        return val, grad, self.hessian(value, target)


class SquaredDistance(BaseDistance):

    def __call__(self, value: T, target: T, *args, **kwargs) -> T:
        d = value - target
        ans = d*d
        return ans
        
    def gradient(self, value: T, target: T) -> T:
        d = value - target
        return 2*d
    
    def both(self, value: T, target: T) -> tuple[T, T]:
        grad = value - target
        d2 = grad*grad
        grad *= 2
        return d2, grad
    
    def hessian(self, value: T, target: T) -> T:
        return 2*np.ones_like(value)
    

METRICS = {
    "squared" : SquaredDistance
}

class BaseScalarProperty(ABC):
    
    def __init__(
            self, 
            target: float,
            emb: BaseEmbedder,
            metric: str,
            weight: Optional[ScalarArr] = 1.0,
            *args, **kwargs
    ) -> None:
        self._target = target
        self.emb = emb
        self._set_weight(weight)
        self.distance_metric: BaseDistance = METRICS[metric]()

    def _set_weight(self, weight):
        if np.ndim(weight) != 0:
            raise ValueError(f"Weight should be a scalar, instead got ndim = {np.ndim(weight)}")
        self.weight = float(weight)

    @property
    def target(self) -> float:
        return self._target
    
    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> float:
        """Actual value of the property parametrized by the GLE matrice"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement a property function")
    
    @property
    def value(self) -> float:
        return self.function()
    
    @property
    def distance(self) -> float:
        """Weighted distance of the current property value from the target"""
        return self.weight * self.distance_metric(self.value, self.target)

    @abstractmethod
    def grad_wrt_A(self, A: Optional[npt.NDArray[np.floating]]=None) -> npt.NDArray[np.floating]:
        """Gradient of the current property value w.r.t. GLE matrix"""
        pass

    def grad_wrt_params(self, x: Optional[npt.NDArray[np.floating]]=None) -> npt.NDArray[np.floating]:
        """Gradient of the current property value w.r.t. optimizable parameters
        Output shape (p,...) where p is the number of parameters
        """
        emb = self.emb
        if x is None:
            x = emb.x
            Ap = emb.A
        else:
            Ap = emb.compute_drift_matrix(emb._inverse_map(x))
        # Gradient of the property w.r.t A matrix
        grad_h_A: npt.NDArray[np.floating] = self.grad_wrt_A(A=Ap)
        # Gradient of the p + auxvar A matrix w.r.t. optimizable parameters
        grad_A_params: npt.NDArray[np.floating] = emb.drift_matrix_x_grad(x)
        # Gradient of the property w.r.t. optimizable parameters
        ans = np.einsum('...ij,pij->p...', grad_h_A, grad_A_params)
        return ans
    
    def gradient(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of the distance to target w.r.t. optimizable parameters.
        Output shape (p,) where p is the number of parameters
        """
        # Derivative of the distance metric w.r.t. property
        grad_distance_h: float = self.distance_metric.gradient(self.value, self.target)
        grad_distance_h *= self.weight
        # Gradient of the property w.r.t. optimizable parameters
        grad_h_params: npt.NDArray[np.floating] = self.grad_wrt_params(x=x)
        # Chain rule:
        ans = grad_distance_h * grad_h_params
        return ans
    
    def both(self, x: Optional[npt.NDArray[np.floating]] = None) -> tuple[float,npt.NDArray[np.floating]]:
        """Distance from target and its gradient w.r.t. optimizable params.
        """
        distance, grad_distance_h = self.distance_metric.both(self.function(x=x), self.target)
        distance *= self.weight
        grad_distance_h *= self.weight
        grad_h_params: npt.NDArray[np.floating] = self.grad_wrt_params()
        grad = grad_distance_h * grad_h_params
        return distance, grad
    
    def gradhess_wrt_params(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    )-> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Gradient and hessian of the current property w.r.t. optimizable parameters.
        Output shapes are (p, ...) and (p, p, ...), where p is the number of parameters and
        ... is the shape of the property -- can be scalar or array
        """
        emb = self.emb
        if x is None:
            x = emb.x
        grad = self.grad_wrt_params(x=x)
        grad_shape = grad.shape
        if np.ndim(grad) == 1:
            def fun(v):
                ans = self.grad_wrt_params(x=v)
                return ans
            hess = jacobian(fun, x, order=4)
            # symmetrize
            hess += hess.T
            hess /= 2
        else:
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
    
    def hessian(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> npt.NDArray[np.floating]:
        """Hessian of the distance to target w.r.t. optimizable parameters.
        Output shape (p, p) where p is the number of parameters
        """
        grad_distance_h: float
        hess_distance_h: float
        _, grad_distance_h, hess_distance_h = self.distance_metric.all(self.function(x=x), self.target)
        grad_distance_h *= self.weight
        hess_distance_h *= self.weight
        grad_h_params : np.ndarray
        hess_h_params : np.ndarray
        grad_h_params, hess_h_params = self.gradhess_wrt_params(x=x)
        ans = hess_distance_h * grad_h_params[:,None] * grad_h_params[None,:]
        ans += grad_distance_h * hess_h_params
        return ans

    def all(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> tuple[float,npt.NDArray[np.floating],npt.NDArray[np.floating]]:
        """Distance of property to target, its gradient, and its hessian w.r.t. optimizable params.
        """
        grad_distance_h: float
        hess_distance_h: float
        distance, grad_distance_h, hess_distance_h = self.distance_metric.all(
            self.function(x=x), self.target)
        distance *= self.weight
        grad_distance_h *= self.weight
        hess_distance_h *= self.weight
        grad_h_params : np.ndarray
        hess_h_params : np.ndarray
        grad_h_params, hess_h_params = self.gradhess_wrt_params(x=x)
        hess = hess_distance_h * grad_h_params[:,None] * grad_h_params[None,:]
        hess += grad_distance_h * hess_h_params
        grad = grad_distance_h * grad_h_params
        return distance, grad, hess
    
BaseMerit = BaseScalarProperty  # alias


class BaseArrayProperty(BaseScalarProperty):
    
    def __init__(
            self, 
            array: npt.NDArray[np.floating],
            target: npt.NDArray[np.floating],
            emb: BaseEmbedder,
            metric: str,
            weight: Optional[ScalarArr] = 1.0,
            *args, **kwargs):
        
        self._array = np.copy(array)
        super().__init__(target, emb, metric, weight, *args, **kwargs)

    def _set_weight(self, weight):
        if np.ndim(weight) not in {0, 1}:
            raise ValueError(f"Weight should be a scalar ot 1D array, instead got ndim = {np.ndim(weight)}")
        try:
            self.weight = np.ones_like(self._array) * weight
        except ValueError as e:
            raise e("Weights could not be broadcast correctly")

    @property
    def array(self) -> npt.NDArray[np.floating]:
        return np.copy(self._array)
    
    @property
    def target(self) -> npt.NDArray[np.floating]:
        return np.copy(self._target)
    
    @property
    def value(self) -> npt.NDArray[np.floating]:
        return self.function()
    
    def function(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement a property function")
    
    @property
    def distance(self) -> float:
        """Weighted distance of the current property value from the target"""
        #TODO: this currently implements the "1-norm" only, generalise later
        return np.sum(self.weight * self.distance_metric(self.value, self.target))
    
    def gradient(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Gradient of the distance to target w.r.t. optimizable parameters.
        Output shape (p,) where p is the number of parameters
        """
        # Derivative of the distance metric w.r.t. property
        grad_distance_h: npt.NDArray[np.floating] = self.distance_metric.gradient(
            self.function(x=x), self.target)
        grad_distance_h *= self.weight
        # Gradient of the property w.r.t. optimizable paramet_spectrum_hessers
        grad_h_params: npt.NDArray[np.floating] = self.grad_wrt_params(x=x)
        # Chain rule:
        ans = np.einsum('t,pt->p', grad_distance_h, grad_h_params)
        return ans
    
    def both(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
        ) -> tuple[float,npt.NDArray[np.floating]]:
        """Distance from target and its gradient w.r.t. optimizable params.
        """
        distance, grad_distance_h = self.distance_metric.both(self.function(x=x), self.target)
        distance *= self.weight
        grad_distance_h *= self.weight
        grad_h_params: npt.NDArray[np.floating] = self.grad_wrt_params(x=x)
        return np.sum(distance), np.einsum('t,pt->p', grad_distance_h, grad_h_params)

    def hessian(self, x: Optional[npt.NDArray[np.floating]] = None) -> npt.NDArray[np.floating]:
        """Hessian of the property w.r.t. optimizable parameters.
        Output shape (p, p) where p is the number of parameters
        """
        # First and second derivative of the distance metric w.r.t. property, shape (t,)
        grad_distance_h: np.ndarray
        hess_distance_h: np.ndarray
        _, grad_distance_h, hess_distance_h = self.distance_metric.all(self.function(x=x), self.target)
        grad_distance_h *= self.weight
        hess_distance_h *= self.weight
        # Gradient and hessian of the property w.r.t. optimizable parameters
        grad_h_params : np.ndarray  # shape (p,t)
        hess_h_params : np.ndarray  # shape (p,p,t)
        grad_h_params, hess_h_params = self.gradhess_wrt_params(x=x)
        ans = np.einsum('t,at,bt->ab', hess_distance_h, grad_h_params, grad_h_params)
        ans += np.einsum('t,abt->ab', grad_distance_h, hess_h_params)
        return ans

    def all(
            self, 
            x: Optional[npt.NDArray[np.floating]] = None
    ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        # Zeroth, first and second derivative of the distance metric w.r.t. property, shape (t,)
        distance: np.ndarray
        grad_distance_h: np.ndarray
        hess_distance_h: np.ndarray
        distance, grad_distance_h, hess_distance_h = self.distance_metric.all(
            self.function(x=x), self.target)
        distance *= self.weight
        grad_distance_h *= self.weight
        hess_distance_h *= self.weight
        # Gradient and hessian of the property w.r.t. optimizable parameters
        grad_h_params : np.ndarray  # shape (p,t)
        hess_h_params : np.ndarray  # shape (p,p,t)
        grad_h_params, hess_h_params = self.gradhess_wrt_params(x=x)
        grad = np.einsum('t,at->a', grad_distance_h, grad_h_params)
        hess = np.einsum('t,at,bt->ab', hess_distance_h, grad_h_params, grad_h_params)
        hess += np.einsum('t,abt->ab', grad_distance_h, hess_h_params)
        return np.sum(distance), grad, hess
