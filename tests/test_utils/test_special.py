#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_special.py
@Time    :   2025/10/15 15:27:39
@Author  :   George Trenins
@Desc    :   Test special functions
'''


from __future__ import print_function, division, absolute_import
import pytest
import numpy as np
import sympy as sp
from numpy.testing import assert_allclose
from glefit.utils.special import coscosh, sincsinhc


def test_coscosh_invalid_nu():
    """Test that coscosh raises appropriate errors for invalid nu."""
    x = np.array([-1.0, -1.0e-3, -1.0e-6, 0.0, 1.0e-5, 0.1, 1.0])
    with pytest.raises(TypeError, match="must be integer"):
        coscosh(x, nu=1.5)
    with pytest.raises(ValueError, match="must be non-negative"):
        coscosh(x, nu=-1)

def test_coscosh_derivatives():
    """Test derivatives of coscosh against symbolic differentiation."""
    x = sp.Symbol('x', real=True)
    f = sp.Piecewise((sp.cos(x), x < 0), (sp.cosh(x), True))
    df = [f]  # 0th derivative is function itself
    for n in range(4):  # test up to 4th derivative
        df.append(sp.diff(df[-1], x))
    f_num = [sp.lambdify(x, d, 'numpy') for d in df]
    x_vals = np.array(np.array([-1.0, -1.0e-3, -1.0e-6, 0.0, 1.0e-5, 0.1, 1.0]))
    for nu in range(5):
        assert_allclose(
            coscosh(x_vals, nu=nu), f_num[nu](x_vals),
            rtol=1e-15,
            err_msg=f"{nu}-th derivative of coscosh does not match symbolic results"
        )

def test_sincsinhc_invalid_nu():
    """Test that sincsinhc raises appropriate errors for invalid nu."""
    x = np.array(np.array([-1.0, -1.0e-3, -1.0e-6, 0.0, 1.0e-5, 0.1, 1.0]))
    with pytest.raises(TypeError, match="must be integer"):
        sincsinhc(x, nu=1.5)
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        sincsinhc(x, nu=3)

def test_sincsinhc_derivatives():
    """Test that sincsinhc matches symbolic piecewise function."""
    x = sp.Symbol('x', real=True)
    eps = 0.01
    # Define piecewise function including Taylor expansion near zero
    sinc_taylor = sp.simplify(sum((sp.sin.taylor_term(n, x) for n in range(7)))/x)
    sinhc_taylor = sp.simplify(sum((sp.sinh.taylor_term(n, x) for n in range(7)))/x) 
    f = sp.Piecewise(
        (sp.sin(x)/(x), x < -eps),
        (sinc_taylor, x < 0),
        (sinhc_taylor, x < eps),
        (sp.sinh(x)/x, True)
    )
    df = [f]  # 0th derivative is function itself
    for n in range(2):  # test up to 2nd derivative
        df.append(sp.diff(df[-1], x))    
    f_num = [sp.lambdify(x, d, 'numpy') for d in df]
    x_vals = np.array([-1.0, -1.0e-3, -1.0e-6, 0.0, 1.0e-5, 0.1, 1.0])
    for nu in range(3):
        assert_allclose(
            sincsinhc(x_vals, nu=nu), f_num[nu](x_vals),
            rtol=1e-12, atol=1e-14,
            err_msg=f"{nu}-th derivative of coscosh does not match symbolic results"
        )

if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        # the lambdified functions divide by 0, which we can safely ignore
        warnings.simplefilter("ignore")  
        pytest.main([__file__])
