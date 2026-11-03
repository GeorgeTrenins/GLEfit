#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_gen2x2.py
@Time    :   2025/10/15 11:00:39
@Author  :   George Trenins
@Desc    :   Test the generalized 2-by-2 Markovian embedder
'''


from __future__ import print_function, division, absolute_import
from glefit.embedding import TwoAuxEmbedder
from scipy.linalg import expm
import pytest
import numpy as np
from numpy.testing import assert_allclose
import sympy as sp


#--- utility functions
def symbolic_softplus(x, sigma, threshold):
    """Symbolic Softplus: log(1 + exp(sigma*x)) / sigma with piecewise approximation."""
    s = sigma * x
    return sp.Piecewise(
        (0, s < -threshold),
        (sp.log(1 + sp.exp(s)) / sigma, s <= threshold),
        (x, True)
    )

def symbolic_softmax(x1, x2, sigma, threshold):
    """Symbolic Softmax: x1 + Softplus(x2 - x1)."""
    return x1 + symbolic_softplus(x2 - x1, sigma, threshold)

def symbolic_softabs(x, sigma, threshold):
    """Symbolic Softabs: -x + Softplus(2*x)."""
    return -x + symbolic_softplus(2*x, sigma, threshold)

#--- unit tests
def kernel_from_matrix(time, theta, A):
    if np.ndim(time) == 0:
        return np.linalg.multi_dot(
            [theta, expm(-A*time), theta]
        )
    else:
        return np.einsum('i,...ij,j->...',
            theta, expm(-A*time[...,None,None]), theta
        )

def test_matrix_rotation():
    """test basis rotation"""
    rng = np.random.default_rng(seed=31415)
    times = np.linspace(0, 1, 25)
    for _ in range(100):
        theta = rng.uniform(low=-10, high=10, size=2)
        A = rng.uniform(low=-10, high=10, size=(2,2))
        K0 = kernel_from_matrix(times, theta, A)
        try:
            th_new, A_new = TwoAuxEmbedder.rotate_matrix(theta, A)
        except ValueError:
            continue
        K1 = kernel_from_matrix(times, th_new, A_new)
        assert_allclose(
            K0, K1,
            rtol=1e-12, atol=1e-14,
            err_msg=f"Kernels computed for orthogonal transformations of the auxiliary variables do not match!"
        )
        Asym = (A_new + A_new.T)
        assert_allclose(
            0.0, [Asym[0,1], Asym[1,0]],
            rtol=1e-12, atol=1e-14,
            err_msg=f"The symmetric part of A is not diagonal"
        )

def test_param_conversion():
    """test kernel equivalence for degenerate parametrizations"""
    rng = np.random.default_rng(seed=31415)
    times = np.linspace(0, 1, 25)
    for _ in range(1000):
        theta = rng.uniform(low=-10, high=10, size=2)
        A = rng.uniform(low=-10, high=10, size=(2,2))
        # kernel before rotation or reparametrization
        K0 = kernel_from_matrix(times, theta, A)  
        try:
            emb = TwoAuxEmbedder.from_matrix(theta, A, sigma=10.0, threshold=30.0)
        except ValueError:
            # ignore non-positive definite drift matrices
            continue
        # kernel after rotation and reparametrization
        K1 = emb.kernel(times)
        assert_allclose(
            K0, K1,
            rtol=1e-10, atol=1e-12,
            err_msg=f"Kernels computed for equivalent parametrizations do not match!"
        )

def test_to_conventional():
    rng = np.random.default_rng(seed=31415)
    for _ in range(100):
        *theta, gamma, delta, Omega = rng.uniform(low=-10, high=10, size=5)
        gamma = abs(gamma)
        if gamma < abs(delta):
            continue
        emb = TwoAuxEmbedder(theta, gamma, delta, Omega, sigma=10.0, threshold=30.0)
        times = np.linspace(0, 1, 25)
        K0 = emb.kernel(times)
        new_params = emb.conventional_params
        emb2 = TwoAuxEmbedder(new_params[:2], *new_params[2:], sigma=10.0, threshold=30.0)
        K1 = emb2.kernel(times)
        assert_allclose(
                K0, K1,
                rtol=1e-10, atol=1e-12,
                err_msg=f"Kernels computed for equivalent parametrizations do not match!"
            )

def test_jac_conventional():
    """Verify conversion of primitive -> conventional parameters and its jacobian."""
    from glefit.utils.numderiv import jacobian
    rng = np.random.default_rng(seed=31415)
    # dummy embedder instance (constructor params not used by jac_conventional)
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=10.0, threshold=30.0)

    for _ in range(100):
        r = float(rng.uniform(0.1, 5.0))
        Gamma = float(rng.uniform(-5.0, 5.0))
        alpha = float(rng.uniform(-10.0, 10.0))
        lamda = float(rng.uniform(0.0, 2.0))
        p = np.array([r, alpha, lamda, Gamma], dtype=float)
        cparams, J_analytic = emb.jac_conventional(p)
        cparams_ref = emb.to_conventional(p)
        # numerical jacobian of the conversion function
        def g(q):
            return emb.to_conventional(q)

        # compare conventional parameters
        assert_allclose(cparams, cparams_ref, rtol=1e-15, atol=1e-14,
                        err_msg="jac_conventional returned different conventional params than to_conventional()")

        # compare analytic and numeric jacobians
        J_numeric = jacobian(g, p, order=4, h=0.0001)
        assert_allclose(J_analytic, J_numeric, rtol=1e-6, atol=1e-8,
                        err_msg="Analytic jacobian from jac_conventional does not match numeric jacobian")

def test_drift_matrix_param_grad():
    """Verify analytic drift-matrix derivatives w.r.t primitive params against numeric jacobian."""
    from glefit.utils.numderiv import jacobian
    rng = np.random.default_rng(seed=31415)
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=10.0, threshold=30.0)
    def A_flat(q):
        return emb.compute_drift_matrix(q).reshape(-1)
    for _ in range(100):
        r = float(rng.uniform(0.1, 5.0))
        Gamma = float(rng.uniform(-5.0, 5.0))
        alpha = float(rng.uniform(-10.0, 10.0))
        lamda = float(rng.uniform(0.0, 2.0))
        p = np.array([r, alpha, lamda, Gamma], dtype=float)
        grad_analytic = emb.drift_matrix_param_grad(p)
        # numeric jacobian of flattened A: shape (9,4) -> reshape to (3,3,4) then moveaxis to (4,3,3)
        J = jacobian(A_flat, p, order=4, h=1.0e-4)
        grad_numeric = np.moveaxis(J.reshape(3,3,4), 2, 0)
        assert grad_analytic.shape == (4, 3, 3)
        assert_allclose(
            grad_analytic, grad_numeric,
            rtol=1e-6, atol=1e-8,
            err_msg="drift_matrix_param_grad analytic gradient does not match numeric jacobian"
        )

def test_kernel_values_overdamped():
    """Test that kernel values match symbolic function."""
    rng = np.random.default_rng(seed=31415)
    t, r, lamda, Gamma = sp.symbols('t r lambda Gamma', real=True, positive=True)
    alpha, = sp.symbols('alpha,', real=True)
    sigma_val = 10.0
    threshold_val = 30.0
    gamma_sym = lamda + symbolic_softmax(
        Gamma, 
        symbolic_softabs(alpha, sigma_val, threshold_val), sigma_val, threshold_val)
    c_sym = sp.cosh(Gamma * t)
    s_sym = sp.sinh(Gamma * t) / (Gamma * t)
    kernel_sym = r**2 * sp.exp(-gamma_sym * abs(t)) * (
        c_sym - alpha * abs(t) * s_sym)
    f_kernel = sp.lambdify((t, r, alpha, lamda, Gamma), kernel_sym, 'numpy')
    times = np.linspace(0, 1, 11)
    times[0] += 1.0e-8
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=sigma_val, threshold=threshold_val)
    for _ in range(10):
        r_val = float(rng.uniform(0.1, 5.0))
        Gamma_val = float(rng.uniform(0.5, 5.0))
        alpha_val = float(rng.uniform(-10.0, 10.0))
        lamda_val = float(rng.uniform(0.0, 2.0))
        emb.primitive_params = np.array([r_val, alpha_val, lamda_val, Gamma_val])
        kernel_vals = emb.kernel_func(times)
        expected_kernel = f_kernel(times, r_val, alpha_val, lamda_val, Gamma_val)
        assert_allclose(
            kernel_vals, expected_kernel,
            rtol=1e-12, atol=1e-14,
            err_msg="Kernel values do not match symbolic expression"
        )

def test_kernel_values_underdamped():
    """Test that kernel values match symbolic function."""
    rng = np.random.default_rng(seed=31415)
    t, r, lamda = sp.symbols('t r lambda', real=True, positive=True)
    alpha, Gamma = sp.symbols('alpha Gamma', real=True)
    sigma_val = 10.0
    threshold_val = 30.0
    gamma_sym = lamda + symbolic_softmax(
        Gamma, 
        symbolic_softabs(alpha, sigma_val, threshold_val), sigma_val, threshold_val)
    c_sym = sp.cos(Gamma * t)
    s_sym = sp.sin(Gamma * t) / (Gamma * t)
    kernel_sym = r**2 * sp.exp(-gamma_sym * abs(t)) * (c_sym - alpha * abs(t) * s_sym)
    f_kernel = sp.lambdify((t, r, alpha, lamda, Gamma), kernel_sym, 'numpy')
    times = np.linspace(0, 1, 11)
    times[0] += 1.0e-6
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=sigma_val, threshold=threshold_val)
    for _ in range(10):
        r_val = float(rng.uniform(0.1, 5.0))
        Gamma_val = float(rng.uniform(-5.0, -0.5))
        alpha_val = float(rng.uniform(-10.0, 10.0))
        lamda_val = float(rng.uniform(0.0, 2.0))
        emb.primitive_params = np.array([r_val, alpha_val, lamda_val, Gamma_val])
        kernel_vals = emb.kernel_func(times)
        expected_kernel = f_kernel(times, r_val, alpha_val, lamda_val, Gamma_val)
        assert_allclose(
            kernel_vals, expected_kernel,
            rtol=1e-12, atol=1e-14,
            err_msg="Kernel values do not match symbolic expression"
        )

def test_kernel_grad_overdamped():
    sigma_val = 10.0
    threshold_val = 30.0
    t, r, alpha, lamda, Gamma = sp.symbols('t r alpha lambda Gamma', real=True)
    gamma_sym = lamda + symbolic_softmax(
        Gamma,
        symbolic_softabs(alpha, sigma_val, threshold_val),
        sigma_val,
        threshold_val
    )
    c_sym = sp.cosh(Gamma * t)
    s_sym = sp.sinh(Gamma * t) / (Gamma * t)
    kernel_sym = r**2 * sp.exp(-gamma_sym * abs(t)) * (
        c_sym - alpha * abs(t) * s_sym)
    # derivatives w.r.t primitive params [r, alpha, lambda, Gamma]
    d_r = sp.diff(kernel_sym, r)
    d_alpha = sp.diff(kernel_sym, alpha)
    d_lamda = sp.diff(kernel_sym, lamda)
    d_Gamma = sp.diff(kernel_sym, Gamma)
    f_r = sp.lambdify((t, r, alpha, lamda, Gamma), d_r, 'numpy')
    f_alpha = sp.lambdify((t, r, alpha, lamda, Gamma), d_alpha, 'numpy')
    f_lamda = sp.lambdify((t, r, alpha, lamda, Gamma), d_lamda, 'numpy')
    f_Gamma = sp.lambdify((t, r, alpha, lamda, Gamma), d_Gamma, 'numpy')
    rng = np.random.default_rng(seed=31415)
    times = np.linspace(0, 1, 11)
    times[0] += 1.0e-8
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=sigma_val, threshold=threshold_val)
    for _ in range(10):
        r_val = float(rng.uniform(0.1, 5.0))
        Gamma_val = float(rng.uniform(0.5, 5.0))   # overdamped / hyperbolic regime
        alpha_val = float(rng.uniform(-10.0, 10.0))
        lamda_val = float(rng.uniform(0.0, 2.0))
        emb.primitive_params = np.array([r_val, alpha_val, lamda_val, Gamma_val])
        grad = emb.kernel_grad(times)  
        expected0 = f_r(times, r_val, alpha_val, lamda_val, Gamma_val)
        expected1 = f_alpha(times, r_val, alpha_val, lamda_val, Gamma_val)
        expected2 = f_lamda(times, r_val, alpha_val, lamda_val, Gamma_val)
        expected3 = f_Gamma(times, r_val, alpha_val, lamda_val, Gamma_val)
        assert_allclose(grad[0], expected0, rtol=1e-12, atol=1e-14,
                        err_msg="dK/dr mismatch in overdamped regime")
        assert_allclose(grad[1], expected1, rtol=1e-12, atol=1e-14,
                        err_msg="dK/dα mismatch in overdamped regime")
        assert_allclose(grad[2], expected2, rtol=1e-12, atol=1e-14,
                        err_msg="dK/dλ mismatch in overdamped regime")
        assert_allclose(grad[3], expected3, rtol=1e-12, atol=1e-14,
                        err_msg="dK/dΓ mismatch in overdamped regime")

def test_kernel_grad_underdamped():
    sigma_val = 10.0
    threshold_val = 30.0
    t, r, alpha, lamda, Gamma = sp.symbols('t r alpha lambda Gamma', real=True)
    gamma_sym = lamda + symbolic_softmax(
        Gamma,
        symbolic_softabs(alpha, sigma_val, threshold_val),
        sigma_val,
        threshold_val
    )
    c_sym = sp.cos(Gamma * t)
    s_sym = sp.sin(Gamma * t) / (Gamma * t)
    kernel_sym = r**2 * sp.exp(-gamma_sym * abs(t)) * (c_sym - alpha * abs(t) * s_sym)
    d_r = sp.diff(kernel_sym, r)
    d_alpha = sp.diff(kernel_sym, alpha)
    d_lamda = sp.diff(kernel_sym, lamda)
    d_Gamma = sp.diff(kernel_sym, Gamma)
    f_r = sp.lambdify((t, r, alpha, lamda, Gamma), d_r, 'numpy')
    f_alpha = sp.lambdify((t, r, alpha, lamda, Gamma), d_alpha, 'numpy')
    f_lamda = sp.lambdify((t, r, alpha, lamda, Gamma), d_lamda, 'numpy')
    f_Gamma = sp.lambdify((t, r, alpha, lamda, Gamma), d_Gamma, 'numpy')
    rng = np.random.default_rng(seed=31415)
    times = np.linspace(0, 1, 11)
    times[0] += 1.0e-8  # avoid t=0 singularities for symbolic sinc form
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=sigma_val, threshold=threshold_val)
    for _ in range(10):
        r_val = float(rng.uniform(0.1, 5.0))
        Gamma_val = float(rng.uniform(-5.0, -0.5))  # underdamped / oscillatory regime
        alpha_val = float(rng.uniform(-10.0, 10.0))
        lamda_val = float(rng.uniform(0.0, 2.0))
        emb.primitive_params = np.array([r_val, alpha_val, lamda_val, Gamma_val])
        grad = emb.kernel_grad(times)
        expected0 = f_r(times, r_val, alpha_val, lamda_val, Gamma_val)
        expected1 = f_alpha(times, r_val, alpha_val, lamda_val, Gamma_val)
        expected2 = f_lamda(times, r_val, alpha_val, lamda_val, Gamma_val)
        expected3 = f_Gamma(times, r_val, alpha_val, lamda_val, Gamma_val)
        assert_allclose(grad[0], expected0, rtol=1e-12, atol=1e-14,
                        err_msg="dK/dr mismatch in underdamped regime")
        assert_allclose(grad[1], expected1, rtol=1e-12, atol=1e-14,
                        err_msg="dK/dα mismatch in underdamped regime")
        assert_allclose(grad[2], expected2, rtol=1e-12, atol=1e-14,
                        err_msg="dK/dλ mismatch in underdamped regime")
        assert_allclose(grad[3], expected3, rtol=1e-12, atol=1e-14,
                        err_msg="dK/dΓ mismatch in underdamped regime")

def test_spectrum_value():
    from scipy.integrate import quad
    from glefit.utils.special import expcoscosh, expsincsinhc
    sigma_val = 10.0
    threshold_val = 30.0
    rng = np.random.default_rng(seed=31415)
    frequencies = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=sigma_val, threshold=threshold_val)
    
    def spectrum_by_quadrature(w, r, alpha, lamda, Gamma):
        gamma_val = lamda + emb._gamma_lower_bound(Gamma, alpha)
        def kernel_t(t):
            C_val = expcoscosh(Gamma, t, gamma_val)
            S_val = expsincsinhc(Gamma, t, gamma_val)
            return r**2 * (C_val - alpha * t * S_val)
        if Gamma > 0:
            envelope = gamma_val - Gamma
        else:
            envelope = gamma_val
        upper_bound = 50.0/envelope
        if w == 0.0:
            # For w=0, compute regular integral
            result, _ = quad(kernel_t, 0, upper_bound, limit=100)
        else:
            result, _ = quad(kernel_t, 0, upper_bound, weight='cos', wvar=w, limit=100)
        return result
    
    for _ in range(100):
        r_val = float(rng.uniform(0.1, 5.0))
        Gamma_val = float(rng.uniform(-5.0, 5.0)) 
        alpha_val = float(rng.uniform(-10.0, 10.0))
        lamda_val = float(rng.uniform(0.1, 2.0))
        emb.primitive_params = np.array([r_val, alpha_val, lamda_val, Gamma_val])
        spectrum_vals = emb.spectrum_func(frequencies)
        expected_spectrum = np.array([
            spectrum_by_quadrature(w, r_val, alpha_val, lamda_val, Gamma_val)
            for w in frequencies
        ])
        assert_allclose(
            spectrum_vals, expected_spectrum,
            rtol=1e-6, atol=1e-8,
            err_msg="Spectrum values do not match numeric quadrature in underdamped regime"
        )

def test_spectrum_grad_overdamped():
    """Test spectrum gradient matches symbolic differentiation in overdamped regime."""
    sigma_val = 10.0
    threshold_val = 30.0
    
    w, r, alpha, lamda, Gamma = sp.symbols('omega r alpha lambda Gamma', real=True, positive=True)
    
    # Build symbolic spectrum expression
    gamma_sym = lamda + symbolic_softmax(
        Gamma,
        symbolic_softabs(alpha, sigma_val, threshold_val),
        sigma_val,
        threshold_val
    )
    w2 = w**2
    g2 = gamma_sym**2
    y = Gamma**2  # overdamped: Gamma > 0, so Gamma*|Gamma| = Gamma^2
    
    numerator = (alpha + gamma_sym)*w2 - (alpha - gamma_sym)*(g2 - y)
    denominator = g2**2 + 2*g2*(w2 - y) + (w2 + y)**2
    spectrum_sym = r**2 * numerator / denominator
    
    # Compute symbolic derivatives
    d_r = sp.diff(spectrum_sym, r)
    d_alpha = sp.diff(spectrum_sym, alpha)
    d_lamda = sp.diff(spectrum_sym, lamda)
    d_Gamma = sp.diff(spectrum_sym, Gamma)
    
    # Convert to numeric functions
    f_r = sp.lambdify((w, r, alpha, lamda, Gamma), d_r, 'numpy')
    f_alpha = sp.lambdify((w, r, alpha, lamda, Gamma), d_alpha, 'numpy')
    f_lamda = sp.lambdify((w, r, alpha, lamda, Gamma), d_lamda, 'numpy')
    f_Gamma = sp.lambdify((w, r, alpha, lamda, Gamma), d_Gamma, 'numpy')
    
    rng = np.random.default_rng(seed=31415)
    frequencies = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=sigma_val, threshold=threshold_val)
    
    for _ in range(10):
        r_val = float(rng.uniform(0.1, 5.0))
        Gamma_val = float(rng.uniform(0.5, 5.0))  # overdamped regime
        alpha_val = float(rng.uniform(-10.0, 10.0))
        lamda_val = float(rng.uniform(0.1, 2.0))
        
        emb.primitive_params = np.array([r_val, alpha_val, lamda_val, Gamma_val])
        
        grad = emb.spectrum_grad(frequencies)
        
        expected0 = f_r(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected1 = f_alpha(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected2 = f_lamda(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected3 = f_Gamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        
        assert_allclose(grad[0], expected0, rtol=1e-10, atol=1e-12,
                        err_msg="dS/dr mismatch in overdamped regime")
        assert_allclose(grad[1], expected1, rtol=1e-10, atol=1e-12,
                        err_msg="dS/dα mismatch in overdamped regime")
        assert_allclose(grad[2], expected2, rtol=1e-10, atol=1e-12,
                        err_msg="dS/dλ mismatch in overdamped regime")
        assert_allclose(grad[3], expected3, rtol=1e-10, atol=1e-12,
                        err_msg="dS/dΓ mismatch in overdamped regime")

def test_spectrum_grad_underdamped():
    """Test spectrum gradient matches symbolic differentiation in underdamped regime."""
    sigma_val = 10.0
    threshold_val = 30.0
    
    w, r, alpha, lamda = sp.symbols('omega r alpha lambda', real=True, positive=True)
    Gamma = sp.symbols('Gamma', real=True, negative=True)
    
    # Build symbolic spectrum expression
    gamma_sym = lamda + symbolic_softmax(
        Gamma,
        symbolic_softabs(alpha, sigma_val, threshold_val),
        sigma_val,
        threshold_val
    )
    w2 = w**2
    g2 = gamma_sym**2
    y = -Gamma**2  # underdamped: Gamma < 0, so Gamma*|Gamma| = -Gamma^2
    
    numerator = (alpha + gamma_sym)*w2 - (alpha - gamma_sym)*(g2 - y)
    denominator = g2**2 + 2*g2*(w2 - y) + (w2 + y)**2
    spectrum_sym = r**2 * numerator / denominator
    
    # Compute symbolic derivatives
    d_r = sp.diff(spectrum_sym, r)
    d_alpha = sp.diff(spectrum_sym, alpha)
    d_lamda = sp.diff(spectrum_sym, lamda)
    d_Gamma = sp.diff(spectrum_sym, Gamma)
    
    # Convert to numeric functions
    f_r = sp.lambdify((w, r, alpha, lamda, Gamma), d_r, 'numpy')
    f_alpha = sp.lambdify((w, r, alpha, lamda, Gamma), d_alpha, 'numpy')
    f_lamda = sp.lambdify((w, r, alpha, lamda, Gamma), d_lamda, 'numpy')
    f_Gamma = sp.lambdify((w, r, alpha, lamda, Gamma), d_Gamma, 'numpy')
    
    rng = np.random.default_rng(seed=31415)
    frequencies = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=sigma_val, threshold=threshold_val)
    
    for _ in range(10):
        r_val = float(rng.uniform(0.1, 5.0))
        Gamma_val = float(rng.uniform(-5.0, -0.5))  # underdamped regime
        alpha_val = float(rng.uniform(-10.0, 10.0))
        lamda_val = float(rng.uniform(0.1, 2.0))
        
        emb.primitive_params = np.array([r_val, alpha_val, lamda_val, Gamma_val])
        
        grad = emb.spectrum_grad(frequencies)
        
        expected0 = f_r(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected1 = f_alpha(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected2 = f_lamda(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected3 = f_Gamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        
        assert_allclose(grad[0], expected0, rtol=1e-10, atol=1e-12,
                        err_msg="dS/dr mismatch in underdamped regime")
        assert_allclose(grad[1], expected1, rtol=1e-10, atol=1e-12,
                        err_msg="dS/dα mismatch in underdamped regime")
        assert_allclose(grad[2], expected2, rtol=1e-10, atol=1e-12,
                        err_msg="dS/dλ mismatch in underdamped regime")
        assert_allclose(grad[3], expected3, rtol=1e-10, atol=1e-12,
                        err_msg="dS/dΓ mismatch in underdamped regime")

def test_spectrum_hess_overdamped():
    """Test spectrum Hessian matches symbolic differentiation in overdamped regime."""
    sigma_val = 10.0
    threshold_val = 30.0
    
    w, r, alpha, lamda, Gamma = sp.symbols('omega r alpha lambda Gamma', real=True, positive=True)
    
    # Build symbolic spectrum expression
    gamma_sym = lamda + symbolic_softmax(
        Gamma,
        symbolic_softabs(alpha, sigma_val, threshold_val),
        sigma_val,
        threshold_val
    )
    w2 = w**2
    g2 = gamma_sym**2
    y = Gamma**2  # overdamped: Gamma > 0, so Gamma*|Gamma| = Gamma^2
    
    numerator = (alpha + gamma_sym)*w2 - (alpha - gamma_sym)*(g2 - y)
    denominator = g2**2 + 2*g2*(w2 - y) + (w2 + y)**2
    spectrum_sym = r**2 * numerator / denominator
    
    # Compute symbolic second derivatives
    d2_rr = sp.diff(spectrum_sym, r, r)
    d2_ralpha = sp.diff(spectrum_sym, r, alpha)
    d2_rlamda = sp.diff(spectrum_sym, r, lamda)
    d2_rGamma = sp.diff(spectrum_sym, r, Gamma)
    d2_alphaalpha = sp.diff(spectrum_sym, alpha, alpha)
    d2_alphalamda = sp.diff(spectrum_sym, alpha, lamda)
    d2_alphaGamma = sp.diff(spectrum_sym, alpha, Gamma)
    d2_lamdalamda = sp.diff(spectrum_sym, lamda, lamda)
    d2_lamdaGamma = sp.diff(spectrum_sym, lamda, Gamma)
    d2_GammaGamma = sp.diff(spectrum_sym, Gamma, Gamma)
    
    # Convert to numeric functions
    f_rr = sp.lambdify((w, r, alpha, lamda, Gamma), d2_rr, 'numpy')
    f_ralpha = sp.lambdify((w, r, alpha, lamda, Gamma), d2_ralpha, 'numpy')
    f_rlamda = sp.lambdify((w, r, alpha, lamda, Gamma), d2_rlamda, 'numpy')
    f_rGamma = sp.lambdify((w, r, alpha, lamda, Gamma), d2_rGamma, 'numpy')
    f_alphaalpha = sp.lambdify((w, r, alpha, lamda, Gamma), d2_alphaalpha, 'numpy')
    f_alphalamda = sp.lambdify((w, r, alpha, lamda, Gamma), d2_alphalamda, 'numpy')
    f_alphaGamma = sp.lambdify((w, r, alpha, lamda, Gamma), d2_alphaGamma, 'numpy')
    f_lamdalamda = sp.lambdify((w, r, alpha, lamda, Gamma), d2_lamdalamda, 'numpy')
    f_lamdaGamma = sp.lambdify((w, r, alpha, lamda, Gamma), d2_lamdaGamma, 'numpy')
    f_GammaGamma = sp.lambdify((w, r, alpha, lamda, Gamma), d2_GammaGamma, 'numpy')
    
    rng = np.random.default_rng(seed=31415)
    frequencies = np.array([0.5, 1.0, 2.0, 5.0])
    
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=sigma_val, threshold=threshold_val)
    
    for _ in range(5):
        r_val = float(rng.uniform(0.1, 5.0))
        Gamma_val = float(rng.uniform(0.5, 5.0))  # overdamped regime
        alpha_val = float(rng.uniform(-10.0, 10.0))
        lamda_val = float(rng.uniform(0.1, 2.0))
        
        emb.primitive_params = np.array([r_val, alpha_val, lamda_val, Gamma_val])
        
        hess = emb.spectrum_hess(frequencies)
        
        expected_rr = f_rr(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_ralpha = f_ralpha(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_rlamda = f_rlamda(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_rGamma = f_rGamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_alphaalpha = f_alphaalpha(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_alphalamda = f_alphalamda(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_alphaGamma = f_alphaGamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_lamdalamd = f_lamdalamda(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_lamdaGamma = f_lamdaGamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_GammaGamma = f_GammaGamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        
        assert_allclose(hess[0, 0], expected_rr, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dr² mismatch in overdamped regime")
        assert_allclose(hess[0, 1], expected_ralpha, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/drdα mismatch in overdamped regime")
        assert_allclose(hess[0, 2], expected_rlamda, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/drdλ mismatch in overdamped regime")
        assert_allclose(hess[0, 3], expected_rGamma, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/drdΓ mismatch in overdamped regime")
        assert_allclose(hess[1, 1], expected_alphaalpha, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dα² mismatch in overdamped regime")
        assert_allclose(hess[1, 2], expected_alphalamda, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dαdλ mismatch in overdamped regime")
        assert_allclose(hess[1, 3], expected_alphaGamma, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dαdΓ mismatch in overdamped regime")
        assert_allclose(hess[2, 2], expected_lamdalamd, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dλ² mismatch in overdamped regime")
        assert_allclose(hess[2, 3], expected_lamdaGamma, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dλdΓ mismatch in overdamped regime")
        assert_allclose(hess[3, 3], expected_GammaGamma, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dΓ² mismatch in overdamped regime")

def test_spectrum_hess_underdamped():
    """Test spectrum Hessian matches symbolic differentiation in underdamped regime."""
    sigma_val = 10.0
    threshold_val = 30.0
    
    w, r, alpha, lamda = sp.symbols('omega r alpha lambda', real=True, positive=True)
    Gamma = sp.symbols('Gamma', real=True, negative=True)
    
    # Build symbolic spectrum expression
    gamma_sym = lamda + symbolic_softmax(
        Gamma,
        symbolic_softabs(alpha, sigma_val, threshold_val),
        sigma_val,
        threshold_val
    )
    w2 = w**2
    g2 = gamma_sym**2
    y = -Gamma**2  # underdamped: Gamma < 0, so Gamma*|Gamma| = -Gamma^2
    
    numerator = (alpha + gamma_sym)*w2 - (alpha - gamma_sym)*(g2 - y)
    denominator = g2**2 + 2*g2*(w2 - y) + (w2 + y)**2
    spectrum_sym = r**2 * numerator / denominator
    
    # Compute symbolic second derivatives
    d2_rr = sp.diff(spectrum_sym, r, r)
    d2_ralpha = sp.diff(spectrum_sym, r, alpha)
    d2_rlamda = sp.diff(spectrum_sym, r, lamda)
    d2_rGamma = sp.diff(spectrum_sym, r, Gamma)
    d2_alphaalpha = sp.diff(spectrum_sym, alpha, alpha)
    d2_alphalamda = sp.diff(spectrum_sym, alpha, lamda)
    d2_alphaGamma = sp.diff(spectrum_sym, alpha, Gamma)
    d2_lamdalamd = sp.diff(spectrum_sym, lamda, lamda)
    d2_lamdaGamma = sp.diff(spectrum_sym, lamda, Gamma)
    d2_GammaGamma = sp.diff(spectrum_sym, Gamma, Gamma)
    
    # Convert to numeric functions
    f_rr = sp.lambdify((w, r, alpha, lamda, Gamma), d2_rr, 'numpy')
    f_ralpha = sp.lambdify((w, r, alpha, lamda, Gamma), d2_ralpha, 'numpy')
    f_rlamda = sp.lambdify((w, r, alpha, lamda, Gamma), d2_rlamda, 'numpy')
    f_rGamma = sp.lambdify((w, r, alpha, lamda, Gamma), d2_rGamma, 'numpy')
    f_alphaalpha = sp.lambdify((w, r, alpha, lamda, Gamma), d2_alphaalpha, 'numpy')
    f_alphalamda = sp.lambdify((w, r, alpha, lamda, Gamma), d2_alphalamda, 'numpy')
    f_alphaGamma = sp.lambdify((w, r, alpha, lamda, Gamma), d2_alphaGamma, 'numpy')
    f_lamdalamda = sp.lambdify((w, r, alpha, lamda, Gamma), d2_lamdalamd, 'numpy')
    f_lamdaGamma = sp.lambdify((w, r, alpha, lamda, Gamma), d2_lamdaGamma, 'numpy')
    f_GammaGamma = sp.lambdify((w, r, alpha, lamda, Gamma), d2_GammaGamma, 'numpy')
    
    rng = np.random.default_rng(seed=31415)
    frequencies = np.array([0.5, 1.0, 2.0, 5.0])
    
    emb = TwoAuxEmbedder(np.array([1.0, 0.0]), 1.0, 0.0, 0.0, sigma=sigma_val, threshold=threshold_val)
    
    for _ in range(5):
        r_val = float(rng.uniform(0.1, 5.0))
        Gamma_val = float(rng.uniform(-5.0, -0.5))  # underdamped regime
        alpha_val = float(rng.uniform(-10.0, 10.0))
        lamda_val = float(rng.uniform(0.1, 2.0))
        
        emb.primitive_params = np.array([r_val, alpha_val, lamda_val, Gamma_val])
        
        hess = emb.spectrum_hess(frequencies)
        
        expected_rr = f_rr(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_ralpha = f_ralpha(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_rlamda = f_rlamda(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_rGamma = f_rGamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_alphaalpha = f_alphaalpha(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_alphalamda = f_alphalamda(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_alphaGamma = f_alphaGamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_lamdalamd = f_lamdalamda(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_lamdaGamma = f_lamdaGamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        expected_GammaGamma = f_GammaGamma(frequencies, r_val, alpha_val, lamda_val, Gamma_val)
        
        assert_allclose(hess[0, 0], expected_rr, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dr² mismatch in underdamped regime")
        assert_allclose(hess[0, 1], expected_ralpha, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/drdα mismatch in underdamped regime")
        assert_allclose(hess[0, 2], expected_rlamda, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/drdλ mismatch in underdamped regime")
        assert_allclose(hess[0, 3], expected_rGamma, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/drdΓ mismatch in underdamped regime")
        assert_allclose(hess[1, 1], expected_alphaalpha, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dα² mismatch in underdamped regime")
        assert_allclose(hess[1, 2], expected_alphalamda, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dαdλ mismatch in underdamped regime")
        assert_allclose(hess[1, 3], expected_alphaGamma, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dαdΓ mismatch in underdamped regime")
        assert_allclose(hess[2, 2], expected_lamdalamd, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dλ² mismatch in underdamped regime")
        assert_allclose(hess[2, 3], expected_lamdaGamma, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dλdΓ mismatch in underdamped regime")
        assert_allclose(hess[3, 3], expected_GammaGamma, rtol=1e-9, atol=1e-11,
                        err_msg="d²S/dΓ² mismatch in underdamped regime")

if __name__ == "__main__":
    pytest.main([__file__])