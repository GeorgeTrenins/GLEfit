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
from glefit.utils.special import coscosh, sincsinhc, Softplus, Softmax, Softabs, expcoscosh, expsincsinhc


def test_coscosh_invalid_nu():
    """Test that coscosh raises appropriate errors for invalid nu."""
    G = np.array([-1.0, -0.5, 0.5, 1.0, 0.5, -1.0, 2.0])
    t = np.array([-1.0, -1.0e-3, -1.0e-6, 0.0, 1.0e-5, 0.1, 1.0])
    with pytest.raises(TypeError, match="must be integer"):
        coscosh(G, t, nu=1.5)
    with pytest.raises(ValueError, match="must be non-negative"):
        coscosh(G, t, nu=-1)

def test_coscosh_derivatives():
    """Test derivatives of coscosh against symbolic differentiation."""
    t, G = sp.symbols('x Gamma', real=True)
    f = sp.Piecewise((sp.cos(G*t), G < 0), (sp.cosh(G*t), True))
    df = [f]  # 0th derivative is function itself
    for _ in range(4):  # test up to 4th derivative
        df.append(sp.diff(df[-1], t))
    f_num = [sp.lambdify((G, t), d, 'numpy') for d in df]
    rng = np.random.default_rng(seed=31415)
    G_vals = rng.uniform(low=-1, high=1, size=2*100)
    t_vals = np.concatenate([
        rng.uniform(low=-1, high=1, size=100),
        rng.uniform(low=-0.01, high=0.01, size=100)])
    for nu in range(5):
        assert_allclose(
            coscosh(G_vals, t_vals, nu=nu), f_num[nu](G_vals, t_vals),
            rtol=1e-15,
            err_msg=f"{nu}-th derivative of coscosh does not match symbolic results"
        )

def test_sincsinhc_invalid_nu():
    """Test that sincsinhc raises appropriate errors for invalid nu."""
    G = np.array([-1.0, -0.5, 0.5, 1.0, 0.5, -1.0, 2.0])
    t = np.array(np.array([-1.0, -1.0e-3, -1.0e-6, 0.0, 1.0e-5, 0.1, 1.0]))
    with pytest.raises(TypeError, match="must be integer"):
        sincsinhc(G, t, nu=1.5)
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        sincsinhc(G, t, nu=3)

def test_sincsinhc_derivatives():
    """Test that sincsinhc matches symbolic piecewise function."""
    G, t = sp.symbols('Gamma t', real=True)
    Gt = G*t
    eps = 0.01
    # Define piecewise function including Taylor expansion near zero
    sinc_taylor = sp.simplify(sum((sp.sin.taylor_term(n, Gt) for n in range(8)))/Gt)
    sinhc_taylor = sp.simplify(sum((sp.sinh.taylor_term(n, Gt) for n in range(8)))/Gt) 
    f = sp.Piecewise(
        (sp.sin(Gt)/(Gt), G*abs(t) < -eps),
        (sinc_taylor, G*abs(t) < 0),
        (sinhc_taylor, G*abs(t) < eps),
        (sp.sinh(Gt)/Gt, True)
    )
    df = [f]  # 0th derivative is function itself
    for n in range(2):  # test up to 2nd derivative
        df.append(sp.diff(df[-1], t))    
    f_num = [sp.lambdify((G, t), d, 'numpy') for d in df]
    rng = np.random.default_rng(seed=31415)
    G_vals = rng.uniform(low=-1, high=1, size=2*100)
    t_vals = np.concatenate([
        rng.uniform(low=-1, high=1, size=100),
        rng.uniform(low=-0.01, high=0.01, size=100)])
    for nu in range(3):
        assert_allclose(
            sincsinhc(G_vals, t_vals, nu=nu), 
            f_num[nu](G_vals, t_vals),
            rtol=1e-10, atol=1e-12,
            err_msg=f"{nu}-th derivative of coscosh does not match symbolic results"
        )

def test_expcoscosh_invalid_nu():
    """Test that expcoscosh raises appropriate errors for invalid nu."""
    G = np.array([-1.0, -0.5, 0.5, 1.0])
    t = np.array([-1.0, -0.5, 0.5, 1.0])
    lamda = 0.5
    with pytest.raises(TypeError, match="must be integer"):
        expcoscosh(G, t, lamda, nu=1.5)
    with pytest.raises(ValueError, match="must be non-negative"):
        expcoscosh(G, t, lamda, nu=-1)


def test_expcoscosh_invalid_lamda():
    """Test that expcoscosh raises errors for invalid lamda."""
    G = np.array([-1.0, 0.5])
    t = np.array([-1.0, 1.0])
    with pytest.raises(ValueError, match="lamda must be non-negative"):
        expcoscosh(G, t, lamda=-0.5)


def test_expcoscosh_invalid_wrt():
    """Test that expcoscosh raises errors for invalid wrt."""
    G = np.array([-1.0, 0.5])
    t = np.array([-1.0, 1.0])
    lamda = 0.5
    with pytest.raises(ValueError, match="must be one of"):
        expcoscosh(G, t, lamda, wrt='invalid')


def test_expcoscosh_values():
    """Test that expcoscosh values match symbolic function."""
    G, t, lamda = sp.symbols('Gamma t lambda', real=True)
    exp_decay = sp.exp(-lamda * t)
    coscosh_expr = sp.Piecewise(
        (sp.cos(G * t), G < 0),
        (sp.cosh(G * t), True)
    )
    expcoscosh_expr = exp_decay * coscosh_expr
    f_num = sp.lambdify((G, t, lamda), expcoscosh_expr, 'numpy')
    rng = np.random.default_rng(seed=31415)
    G_vals = rng.uniform(low=-2.0, high=2.0, size=50)
    t_vals = rng.uniform(low=-2.0, high=2.0, size=50)
    lamda_vals = rng.uniform(low=0.1, high=2.0, size=50)
    result = expcoscosh(G_vals, t_vals, lamda_vals, nu=0)
    expected = f_num(G_vals, t_vals, lamda_vals)
    assert_allclose(
        result, expected,
        rtol=1e-10, atol=1e-12,
        err_msg="expcoscosh values do not match symbolic results"
    )


def test_expcoscosh_first_derivative_time():
    """Test first derivative of expcoscosh w.r.t. time against symbolic differentiation."""
    G, lamda = sp.symbols('Gamma lambda', real=True)
    t_pos = sp.symbols('t', real=True, positive=True)
    expcoscosh_pos = sp.exp(-lamda * t_pos) * sp.Piecewise(
        (sp.cos(G * t_pos), G < 0),
        (sp.cosh(G * t_pos), True)
    )
    d_expr = sp.diff(expcoscosh_pos, t_pos)
    f_num = sp.lambdify((G, t_pos, lamda), d_expr, 'numpy')
    
    # Test points (positive t only for symbolic derivative)
    rng = np.random.default_rng(seed=31415)
    G_vals = rng.uniform(low=-2.0, high=2.0, size=30)
    t_vals = rng.uniform(low=0.01, high=2.0, size=30)
    lamda_vals = rng.uniform(low=0.1, high=2.0, size=30)    
    result = expcoscosh(G_vals, t_vals, lamda_vals, nu=1, wrt='time')
    expected = f_num(G_vals, t_vals, lamda_vals)
    assert_allclose(
        result, expected,
        rtol=1e-10, atol=1e-12,
        err_msg="First derivative of expcoscosh w.r.t. time does not match symbolic results"
    )


def test_expcoscosh_second_derivative_time():
    """Test second derivative of expcoscosh w.r.t. time against symbolic differentiation."""
    G, lamda = sp.symbols('Gamma lambda', real=True)
    t_pos = sp.symbols('t', real=True, positive=True)
    expcoscosh_pos = sp.exp(-lamda * t_pos) * sp.Piecewise(
        (sp.cos(G * t_pos), G < 0),
        (sp.cosh(G * t_pos), True)
    )
    d2_expr = sp.diff(expcoscosh_pos, t_pos, 2)
    f_num = sp.lambdify((G, t_pos, lamda), d2_expr, 'numpy')
    rng = np.random.default_rng(seed=31415)
    G_vals = rng.uniform(low=-2.0, high=2.0, size=30)
    t_vals = rng.uniform(low=0.01, high=2.0, size=30)
    lamda_vals = rng.uniform(low=0.1, high=2.0, size=30)
    result = expcoscosh(G_vals, t_vals, lamda_vals, nu=2, wrt='time')
    expected = f_num(G_vals, t_vals, lamda_vals)
    assert_allclose(
        result, expected,
        rtol=1e-10, atol=1e-12,
        err_msg="Second derivative of expcoscosh w.r.t. time does not match symbolic results"
    )

def test_expcoscosh_numerical_stability():
    """Test numerical stability of expcoscosh for large decay rates."""
    # Test that expcoscosh doesn't overflow/underflow for large lamda

    t_vals = np.array([10.0, 50.0, 100.0, 1000.0])
    G_vals = 7.9
    lamda = 8.0
    result = expcoscosh(G_vals, t_vals, lamda, nu=0)
    # Check that result is finite and doesn't contain NaN
    assert np.all(np.isfinite(result)), "expcoscosh produced non-finite values"
    

def test_expsincsinhc_invalid_nu():
    """Test that expsincsinhc raises appropriate errors for invalid nu."""
    G = np.array([-1.0, -0.5, 0.5, 1.0])
    t = np.array([-1.0, -0.5, 0.5, 1.0])
    lamda = 0.5
    with pytest.raises(TypeError, match="must be integer"):
        expsincsinhc(G, t, lamda, nu=1.5)
    with pytest.raises(ValueError, match="must be non-negative"):
        expsincsinhc(G, t, lamda, nu=-1)


def test_expsincsinhc_invalid_lamda():
    """Test that expsincsinhc raises errors for invalid lamda."""
    G = np.array([-1.0, 0.5])
    t = np.array([-1.0, 1.0])
    with pytest.raises(ValueError, match="lamda must be non-negative"):
        expsincsinhc(G, t, lamda=-0.5)


def test_expsincsinhc_invalid_wrt():
    """Test that expsincsinhc raises errors for invalid wrt."""
    G = np.array([-1.0, 0.5])
    t = np.array([-1.0, 1.0])
    lamda = 0.5
    with pytest.raises(ValueError, match="must be one of"):
        expsincsinhc(G, t, lamda, wrt='invalid')


def test_expsincsinhc_values():
    """Test that expsincsinhc values match symbolic function."""
    from glefit.utils.special import expsincsinhc
    
    G, t, lamda = sp.symbols('Gamma t lambda', real=True)
    Gt = G * t
    eps = 0.01
    
    # Define piecewise function including Taylor expansion near zero
    sinc_taylor = sp.simplify(sum((sp.sin.taylor_term(n, Gt) for n in range(8))) / Gt)
    sinhc_taylor = sp.simplify(sum((sp.sinh.taylor_term(n, Gt) for n in range(8))) / Gt)
    
    sincsinhc_expr = sp.Piecewise(
        (sp.sin(Gt) / Gt, G * sp.Abs(t) < -eps),
        (sinc_taylor, G * sp.Abs(t) < 0),
        (sinhc_taylor, G * sp.Abs(t) < eps),
        (sp.sinh(Gt) / Gt, True)
    )
    
    expsincsinhc_expr = sp.exp(-lamda * t) * sincsinhc_expr
    f_num = sp.lambdify((G, t, lamda), expsincsinhc_expr, 'numpy')
    
    rng = np.random.default_rng(seed=31415)
    G_vals = rng.uniform(low=-1, high=1, size=100)
    t_vals = np.concatenate([
        rng.uniform(low=-1, high=1, size=50),
        rng.uniform(low=-0.01, high=0.01, size=50)
    ])
    lamda_vals = rng.uniform(low=0.1, high=2.0, size=100)
    
    result = expsincsinhc(G_vals, t_vals, lamda_vals, nu=0)
    expected = f_num(G_vals, t_vals, lamda_vals)
    
    assert_allclose(
        result, expected,
        rtol=1e-10, atol=1e-12,
        err_msg="expsincsinhc values do not match symbolic results"
    )


def test_expsincsinhc_first_derivative_time():
    """Test first derivative of expsincsinhc w.r.t. time against symbolic differentiation."""
    from glefit.utils.special import expsincsinhc
    
    G, t, lamda = sp.symbols('Gamma t lambda', real=True)
    Gt = G * t
    eps = 0.01
    
    # Define piecewise function including Taylor expansion near zero
    sinc_taylor = sp.simplify(sum((sp.sin.taylor_term(n, Gt) for n in range(8))) / Gt)
    sinhc_taylor = sp.simplify(sum((sp.sinh.taylor_term(n, Gt) for n in range(8))) / Gt)
    
    sincsinhc_expr = sp.Piecewise(
        (sp.sin(Gt) / Gt, G * sp.Abs(t) < -eps),
        (sinc_taylor, G * sp.Abs(t) < 0),
        (sinhc_taylor, G * sp.Abs(t) < eps),
        (sp.sinh(Gt) / Gt, True)
    )
    
    expsincsinhc_expr = sp.exp(-lamda * t) * sincsinhc_expr
    d_expr = sp.diff(expsincsinhc_expr, t)
    f_num = sp.lambdify((G, t, lamda), d_expr, 'numpy')
    
    rng = np.random.default_rng(seed=31415)
    G_vals = rng.uniform(low=-1, high=1, size=100)
    t_vals = np.concatenate([
        rng.uniform(low=-1, high=1, size=50),
        rng.uniform(low=-0.01, high=0.01, size=50)
    ])
    lamda_vals = rng.uniform(low=0.1, high=2.0, size=100)
    
    result = expsincsinhc(G_vals, t_vals, lamda_vals, nu=1, wrt='time')
    expected = f_num(G_vals, t_vals, lamda_vals)
    
    assert_allclose(
        result, expected,
        rtol=1e-10, atol=1e-12,
        err_msg="First derivative of expsincsinhc w.r.t. time does not match symbolic results"
    )


def test_expsincsinhc_second_derivative_time():
    """Test second derivative of expsincsinhc w.r.t. time against symbolic differentiation."""
    from glefit.utils.special import expsincsinhc
    
    G, t, lamda = sp.symbols('Gamma t lambda', real=True)
    Gt = G * t
    eps = 0.01
    
    # Define piecewise function including Taylor expansion near zero
    sinc_taylor = sp.simplify(sum((sp.sin.taylor_term(n, Gt) for n in range(8))) / Gt)
    sinhc_taylor = sp.simplify(sum((sp.sinh.taylor_term(n, Gt) for n in range(8))) / Gt)
    
    sincsinhc_expr = sp.Piecewise(
        (sp.sin(Gt) / Gt, G * sp.Abs(t) < -eps),
        (sinc_taylor, G * sp.Abs(t) < 0),
        (sinhc_taylor, G * sp.Abs(t) < eps),
        (sp.sinh(Gt) / Gt, True)
    )
    
    expsincsinhc_expr = sp.exp(-lamda * t) * sincsinhc_expr
    d2_expr = sp.diff(expsincsinhc_expr, t, 2)
    f_num = sp.lambdify((G, t, lamda), d2_expr, 'numpy')
    
    rng = np.random.default_rng(seed=31415)
    G_vals = rng.uniform(low=-1, high=1, size=100)
    t_vals = np.concatenate([
        rng.uniform(low=-1, high=1, size=50),
        rng.uniform(low=-0.01, high=0.01, size=50)
    ])
    lamda_vals = rng.uniform(low=0.1, high=2.0, size=100)
    
    result = expsincsinhc(G_vals, t_vals, lamda_vals, nu=2, wrt='time')
    expected = f_num(G_vals, t_vals, lamda_vals)
    
    assert_allclose(
        result, expected,
        rtol=1e-10, atol=1e-12,
        err_msg="Second derivative of expsincsinhc w.r.t. time does not match symbolic results"
    )


def test_expsincsinhc_numerical_stability():
    """Test numerical stability of expsincsinhc for large arguments."""
    from glefit.utils.special import expsincsinhc
    
    t_vals = np.array([10.0, 50.0, 100.0, 1000.0])
    G_vals = 7.9
    lamda = 8.0
    
    result = expsincsinhc(G_vals, t_vals, lamda, nu=0)
    
    # Check that result is finite and doesn't contain NaN
    assert np.all(np.isfinite(result)), "expsincsinhc produced non-finite values"
    
def test_softplus_init_invalid_sigma():
    """Test that Softplus raises errors for invalid sigma."""
    with pytest.raises(TypeError, match="sigma must be a float"):
        Softplus(sigma="invalid")
    with pytest.raises(ValueError, match="sigma must be positive"):
        Softplus(sigma=0.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        Softplus(sigma=-1.0)


def test_softplus_init_invalid_threshold():
    """Test that Softplus raises errors for invalid threshold."""
    with pytest.raises(TypeError, match="threshold must be a float"):
        Softplus(threshold="invalid")
    with pytest.raises(ValueError, match="threshold must be positive"):
        Softplus(threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be positive"):
        Softplus(threshold=-1.0)


def test_softplus_values():
    """Test that softplus values match symbolic function."""
    sigma = 5.0
    threshold = 20.0
    softplus = Softplus(sigma=sigma, threshold=threshold)
    x = sp.Symbol('x', real=True)
    s = sigma * x
    f = sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (x, True)
    )
    f_num = sp.lambdify(x, f, 'numpy')
    
    x_vals = np.array([-5.0, -3.0, -2.5, -1.0, 0.0, 1.0, 2.5, 3.0, 5.0])
    
    assert_allclose(
        softplus(x_vals), f_num(x_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="Softmax values do not match symbolic results"
    )


def test_softplus_first_derivative():
    """Test first derivative of softplus against symbolic differentiation."""
    sigma = 5.0
    threshold = 20.0
    softplus = Softplus(sigma=sigma, threshold=threshold)
    x = sp.Symbol('x', real=True)
    s = sigma * x
    f = sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (x, True)
    )
    df = sp.diff(f, x)
    df_num = sp.lambdify(x, df, 'numpy')
    x_vals = np.array([-5.0, -3.0, -2.5, -1.0, 0.0, 1.0, 2.5, 3.0, 5.0])
    
    assert_allclose(
        softplus(x_vals, nu=1), df_num(x_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="First derivative of softplus does not match symbolic results"
    )


def test_softplus_second_derivative():
    sigma = 5.0
    threshold = 20.0
    softplus = Softplus(sigma=sigma, threshold=threshold)
    x = sp.Symbol('x', real=True)
    s = sigma * x
    f = sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (x, True)
    )
    d2f = sp.diff(f, x, 2)
    d2f_num = sp.lambdify(x, d2f, 'numpy')
    x_vals = np.array([-5.0, -3.0, -2.5, -1.0, 0.0, 1.0, 2.5, 3.0, 5.0])
    
    assert_allclose(
        softplus(x_vals, nu=2), d2f_num(x_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="Second derivative of softplus does not match symbolic results"
    )

def test_softplus_invalid_nu():
    softplus = Softplus()
    x = np.array([0.0, 1.0])
    
    with pytest.raises(ValueError, match="Expected nu = 0, 1, or 2"):
        softplus(x, nu=3)
    
    with pytest.raises(ValueError, match="Expected nu = 0, 1, or 2"):
        softplus(x, nu="invalid")


def test_softmax_input_validation():
    softmax = Softmax()
    x1 = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Expecting a 1d input"):
        softmax(np.array([[1.0]]), x1)
    with pytest.raises(ValueError, match="same shape"):
        softmax(x1, np.array([1.0]))


def test_softmax_values():
    sigma = 5.0
    threshold = 20.0
    softmax = Softmax(sigma=sigma, threshold=threshold)
    x1_sym = sp.Symbol('x1', real=True)
    x2_sym = sp.Symbol('x2', real=True)
    diff = x2_sym - x1_sym
    s = sigma * diff
    softplus_expr = sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (diff, True)
    )
    f = x1_sym + softplus_expr
    f_num = sp.lambdify((x1_sym, x2_sym), f, 'numpy')
    x1_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x2_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert_allclose(
        softmax(x1_vals, x2_vals), f_num(x1_vals, x2_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="Softmax values do not match symbolic results"
    )


def test_softmax_first_derivative():
    """Test first derivative of softmax against symbolic differentiation."""
    sigma = 5.0
    threshold = 20.0
    softmax = Softmax(sigma=sigma, threshold=threshold)
    x1_sym = sp.Symbol('x1', real=True)
    x2_sym = sp.Symbol('x2', real=True)
    diff = x2_sym - x1_sym
    s = sigma * diff
    softplus_expr = sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (diff, True)
    )
    f = x1_sym + softplus_expr
    df_dx1 = sp.diff(f, x1_sym)
    df_dx2 = sp.diff(f, x2_sym)
    df_dx1_num = sp.lambdify((x1_sym, x2_sym), df_dx1, 'numpy')
    df_dx2_num = sp.lambdify((x1_sym, x2_sym), df_dx2, 'numpy')
    x1_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x2_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    deriv = softmax(x1_vals, x2_vals, nu=1)
    assert_allclose(
        deriv[0], df_dx1_num(x1_vals, x2_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="First derivative w.r.t. x1 does not match symbolic results"
    )
    assert_allclose(
        deriv[1], df_dx2_num(x1_vals, x2_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="First derivative w.r.t. x2 does not match symbolic results"
    )


def test_softmax_second_derivative():
    """Test second derivative of softmax against symbolic differentiation."""
    sigma = 5.0
    threshold = 20.0
    softmax = Softmax(sigma=sigma, threshold=threshold)
    x1_sym = sp.Symbol('x1', real=True)
    x2_sym = sp.Symbol('x2', real=True)
    diff = x2_sym - x1_sym
    s = sigma * diff
    softplus_expr = sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (diff, True)
    )
    f = x1_sym + softplus_expr
    d2f_dx1dx1 = sp.diff(f, x1_sym, 2)
    d2f_dx1dx2 = sp.diff(f, x1_sym, x2_sym)
    d2f_dx2dx2 = sp.diff(f, x2_sym, 2)
    
    d2f_dx1dx1_num = sp.lambdify((x1_sym, x2_sym), d2f_dx1dx1, 'numpy')
    d2f_dx1dx2_num = sp.lambdify((x1_sym, x2_sym), d2f_dx1dx2, 'numpy')
    d2f_dx2dx2_num = sp.lambdify((x1_sym, x2_sym), d2f_dx2dx2, 'numpy')
    
    x1_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x2_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    
    hess = softmax(x1_vals, x2_vals, nu=2)
    
    assert_allclose(
        hess[0, 0], d2f_dx1dx1_num(x1_vals, x2_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="Second derivative d²/dx1² does not match symbolic results"
    )
    assert_allclose(
        hess[0, 1], d2f_dx1dx2_num(x1_vals, x2_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="Second derivative d²/dx1dx2 does not match symbolic results"
    )
    assert_allclose(
        hess[1, 0], d2f_dx1dx2_num(x1_vals, x2_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="Second derivative d²/dx1dx2 does not match symbolic results"
    )
    assert_allclose(
        hess[1, 1], d2f_dx2dx2_num(x1_vals, x2_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="Second derivative d²/dx2² does not match symbolic results"
    )


def test_softmax_invalid_nu():
    """Test that Softmax raises error for invalid nu."""
    softmax = Softmax()
    x1 = np.array([0.0, 1.0])
    x2 = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Expected nu = 0, 1, or 2"):
        softmax(x1, x2, nu=3)
    with pytest.raises(ValueError, match="Expected nu = 0, 1, or 2"):
        softmax(x1, x2, nu="invalid")


def test_softmax_scalar_input():
    """Test that Softmax handles scalar inputs correctly."""
    softmax = Softmax()
    result = softmax(1.0, 2.0)
    assert np.isscalar(result), "Scalar input should return scalar output"
    deriv = softmax(np.array([1.0]), np.array([2.0]), nu=1)
    assert deriv.shape == (2, 1), "Derivative of scalar input should have shape (2, 1)"


def test_softabs_values():
    """Test that softabs values match symbolic function."""
    sigma = 5.0
    threshold = 20.0
    softabs = Softabs(sigma=sigma, threshold=threshold)
    x = sp.Symbol('x', real=True)
    s = 2 * sigma * x
    softplus_expr = sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (2*x, True)
    )
    f = -x + softplus_expr
    f_num = sp.lambdify(x, f, 'numpy')
    x_vals = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    assert_allclose(
        softabs(x_vals), f_num(x_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="Softabs values do not match symbolic results"
    )


def test_softabs_first_derivative():
    """Test first derivative of softabs against symbolic differentiation."""
    sigma = 5.0
    threshold = 20.0
    softabs = Softabs(sigma=sigma, threshold=threshold)
    
    x = sp.Symbol('x', real=True)
    s = 2 * sigma * x
    softplus_expr = sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (2*x, True)
    )
    f = -x + softplus_expr
    df = sp.diff(f, x)
    df_num = sp.lambdify(x, df, 'numpy')
    
    x_vals = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    
    assert_allclose(
        softabs(x_vals, nu=1), df_num(x_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="First derivative of softabs does not match symbolic results"
    )


def test_softabs_second_derivative():
    """Test second derivative of softabs against symbolic differentiation."""
    sigma = 5.0
    threshold = 20.0
    softabs = Softabs(sigma=sigma, threshold=threshold)
    
    x = sp.Symbol('x', real=True)
    s = 2 * sigma * x
    softplus_expr = sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (2*x, True)
    )
    f = -x + softplus_expr
    d2f = sp.diff(f, x, 2)
    d2f_num = sp.lambdify(x, d2f, 'numpy')
    
    x_vals = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    
    assert_allclose(
        softabs(x_vals, nu=2), d2f_num(x_vals),
        rtol=1e-12, atol=1e-14,
        err_msg="Second derivative of softabs does not match symbolic results"
    )


def test_softabs_invalid_nu():
    """Test that Softabs raises error for invalid nu."""
    softabs = Softabs()
    x = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="Expected nu = 0, 1, or 2"):
        softabs(x, nu=3)
    with pytest.raises(ValueError, match="Expected nu = 0, 1, or 2"):
        softabs(x, nu="invalid")

if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        # the lambdified functions divide by 0, which we can safely ignore
        warnings.simplefilter("ignore")  
        pytest.main([__file__])
