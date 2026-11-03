#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_numderiv.py
@Time    :   2025/10/09 14:09:57
@Author  :   George Trenins
@Desc    :   Test numerical differentiation
'''

from __future__ import print_function, division, absolute_import
from glefit.utils.numderiv import diff, jacobian
import sympy as sp
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_diff_first_derivative():
    """Test first derivatives using 2nd and 4th order stencils."""
    x = sp.Symbol('x')
    funcs = [
        sp.Lambda(x, x**2),          
        sp.Lambda(x, sp.exp(-x**2)), 
        sp.Lambda(x, sp.sin(x))      
    ]
    
    num_funcs = []
    for f in funcs:
        df = sp.diff(f(x), x)
        num_f = sp.lambdify(x, f(x), modules='numpy')
        num_df = sp.lambdify(x, df, modules='numpy')
        num_funcs.append((num_f, num_df, str(f(x))))
    
    # Test points
    x_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for f, dfdx, expr in num_funcs:
        for order in [2, 4]:
            for x in x_vals:
                numerical = diff(f, x, order=order, nu=1)
                analytical = dfdx(x)
                assert_allclose(
                    numerical, analytical,
                    rtol=1e-6, atol=1e-8,
                    err_msg=f"First derivative of {expr} at x={x} "
                           f"with order={order} does not match analytical result"
                )

def test_diff_second_derivative():
    """Test second derivatives using 2nd and 4th order stencils."""
    x = sp.Symbol('x')
    funcs = [
        sp.Lambda(x, x**2),          
        sp.Lambda(x, sp.exp(-x**2)), 
        sp.Lambda(x, sp.sin(x))      
    ]
    
    num_funcs = []
    for f in funcs:
        d2f = sp.diff(f(x), x, 2)
        num_f = sp.lambdify(x, f(x), modules='numpy')
        num_d2f = sp.lambdify(x, d2f, modules='numpy')
        num_funcs.append((num_f, num_d2f, str(f(x))))
    
    # Test points
    x_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for f, d2fdx2, expr in num_funcs:
        for order in [2, 4, 6]:
            for x in x_vals:
                numerical = diff(f, x, order=order, nu=2)
                analytical = d2fdx2(x)
                assert_allclose(
                    numerical, analytical,
                    rtol=1e-6, atol=1e-8,
                    err_msg=f"Second derivative of {expr} at x={x} "
                           f"with order={order} does not match analytical result"
                )

def test_jacobian_values():
    """Test Jacobian computation for different vector functions."""
    def f1(x):
        """f: R^2 -> R^2, f(x,y) = [x^2 + y, x*y]"""
        return np.array([x[0]**2 + x[1], x[0]*x[1]])
    
    def j1(x):
        """Jacobian of f1"""
        return np.array([[2*x[0], 1], [x[1], x[0]]])
    
    def f2(x):
        """f: R^3 -> R^2, f(x,y,z) = [sin(x)*cos(y), exp(-z)*y]"""
        return np.array([np.sin(x[0])*np.cos(x[1]), np.exp(-x[2])*x[1]])
    
    def j2(x):
        """Jacobian of f2"""
        return np.array([
            [np.cos(x[0])*np.cos(x[1]), -np.sin(x[0])*np.sin(x[1]), 0],
            [0, np.exp(-x[2]), -np.exp(-x[2])*x[1]]
        ])
    
    test_cases = [
        (f1, j1, np.array([1.0, 2.0])),
        (f1, j1, np.array([-0.5, 0.5])),
        (f2, j2, np.array([0.1, 0.2, 0.3])),
        (f2, j2, np.array([np.pi/4, np.pi/3, 1.0]))
    ]
    
    for f, j_analytical, x in test_cases:
        # Test both 2nd and 4th order methods
        for order in [2, 4]:
            j_numerical = jacobian(f, x, order=order)
            j_exact = j_analytical(x)
            
            assert_allclose(
                j_numerical, j_exact,
                rtol=1e-6, atol=1e-8,
                err_msg=f"Jacobian at x={x} with order={order} "
                       f"does not match analytical result"
            )

def test_jacobian_with_value():
    """Test that jacobian returns correct function value when requested."""
    def f(x):
        return np.array([x[0]**2 + x[1], x[0]*x[1]])
    
    x = np.array([1.0, 2.0])
    f_expected = f(x)
    
    J, f_value = jacobian(f, x, value=True)
    
    assert_allclose(
        f_value, f_expected,
        rtol=1e-15,
        err_msg="Function value returned by jacobian does not match direct evaluation"
    )

def test_jacobian_input_validation():
    """Test that jacobian properly validates input arguments."""
    def f(x):
        return np.array([x[0]**2])
    
    # Test invalid x dimension
    with pytest.raises(ValueError, match="x must be a 1D array-like"):
        jacobian(f, np.array([[1.0]]))
    
    # Test invalid h dimension
    with pytest.raises(ValueError, match="h must have the same shape as x"):
        jacobian(f, np.array([1.0]), h=np.array([[0.1]]))
    
    # Test invalid function output dimension
    def f_invalid(x):
        return np.array([[x[0]**2]])
    
    with pytest.raises(ValueError, match=r"f\(x\) must return a 1D array-like"):
        jacobian(f_invalid, np.array([1.0]))
    
    # Test invalid fshape
    with pytest.raises(ValueError, match=r"f\(x\) must return a 1D array-like"):
        jacobian(f, np.array([1.0]), fshape=(2,2))

if __name__ == "__main__":
    pytest.main([__file__])