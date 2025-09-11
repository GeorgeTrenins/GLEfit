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

# ---- Decorator -----------------------------------------------------------

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

    def __init__(self, weights: Optional[ScalarArr] =  1.0) -> None:
        self.weights = weights

    # Automatically wrap subclass implementations of __call__ and gradient
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in ("__call__", "gradient"):
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


class SquaredDistance(BaseDistance):

    def __call__(self, value: ScalarArr, target: ScalarArr, *args, **kwargs) -> float:
        d = value - target
        ans = self.weights * (d*d)
        try:
            return ans.item()
        except ValueError:
            return np.sum(ans)
        
    def gradient(self, value: T, target: T) -> T:
        d = value - target
        return self.weights * (2*d)
    

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
        self.distance_metric: BaseDistance = METRICS[metric](weights=weight)

    @property
    def target(self) -> float:
        return self._target
    
    @property
    def value(self) -> float:
        return self._get_value()

    @abstractmethod
    def _get_value(self) -> float:
        pass

    @abstractmethod
    def _grad_wrt_A(self) -> npt.NDArray[np.floating]:
        pass

    def _grad_wrt_params(self) -> npt.NDArray[np.floating]:
        # Gradient of the p + auxvar A matrix w.r.t. optimizable parameters
        grad_A_params: npt.NDArray[np.floating] = self.emb.grad_A
        # Gradient of the property w.r.t A matrix
        grad_h_A: npt.NDArray[np.floating] = self._grad_wrt_A()
        # Gradient of the property w.r.t. optimizable parameters
        ans = np.einsum('...ij,pij->p...', grad_h_A, grad_A_params)
        return ans
    
    def gradient(self) -> npt.NDArray[np.floating]:
        # Gradient of the property w.r.t. optimizable parameters
        grad_h_params: npt.NDArray[np.floating] = self._grad_wrt_params()
        # Derivative of the distance metric w.r.t. property
        grad_distance_h: float = self.distance_metric.gradient(self.value, self.target)
        # Chain rule:
        ans = grad_distance_h * grad_h_params
        return ans


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

    @property
    def array(self) -> npt.NDArray[np.floating]:
        return np.copy(self._array)
    
    @property
    def target(self) -> npt.NDArray[np.floating]:
        return np.copy(self._target)
    
    @property
    def value(self) -> npt.NDArray[np.floating]:
        return self._get_value()
    

    @abstractmethod
    def _get_value(self) -> npt.NDArray[np.floating]:
        pass
    
    def gradient(self) -> npt.NDArray[np.floating]:
        # Gradient of the property w.r.t. optimizable parameters
        grad_h_params: npt.NDArray[np.floating] = self._grad_wrt_params()
        # Derivative of the distance metric w.r.t. property
        grad_distance_h: npt.NDArray[np.floating] = self.distance_metric.gradient(self.value, self.target)
        # Chain rule:
        ans = np.einsum('t,pt->p', grad_distance_h, grad_h_params)
        return ans

