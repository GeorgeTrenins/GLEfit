#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_pronycos.py
@Time    :   2025/10/08 11:47:15
@Author  :   George Trenins
@Desc    :   Unit tests for the oscillatory Prony embedding (https://doi.org/10.1103/PhysRevB.89.134303)
'''


from __future__ import print_function, division, absolute_import


from __future__ import print_function, division, absolute_import
import pytest
from glefit.embedding import PronyCosineEmbedder
import numpy as np
import sympy as sp
from numpy.testing import assert_allclose


def test_kernel_values():
    """Test that kernel values match symbolic function."""
    # Setup symbolic expressions
    t, theta, gamma, omega = sp.symbols('t theta gamma omega', real=True)
    kernel = theta**2 * sp.exp(-gamma * sp.Abs(t)) * sp.cos(omega * t)
    
    # Convert to numeric functions
    f_kernel = sp.lambdify((t, theta, gamma, omega), kernel, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    omega_val = 2.5
    times = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    
    # Initialize embedder
    emb = PronyCosineEmbedder(theta_val, gamma_val, omega_val)
    
    # Get numerical derivatives
    kernel_vals = emb.kernel(times)
    
    # Compare with symbolic results
    expected_kernel = f_kernel(times, theta_val, gamma_val, omega_val)
    
    assert_allclose(
        kernel_vals, expected_kernel,
        rtol=1e-15, 
        err_msg="Friction kernel does not match symbolic result"
    )

def test_kernel_gradient():
    """Test that kernel derivatives match symbolic differentiation results."""
    # Setup symbolic expressions
    t, theta, gamma, omega = sp.symbols('t theta gamma omega', real=True)
    kernel = theta**2 * sp.exp(-gamma * sp.Abs(t)) * sp.cos(omega*t)
    
    # Compute symbolic derivatives
    d_theta = sp.diff(kernel, theta)
    d_gamma = sp.diff(kernel, gamma)
    d_omega = sp.diff(kernel, omega)
    
    # Convert to numeric functions
    f_theta = sp.lambdify((t, theta, gamma, omega), d_theta, 'numpy')
    f_gamma = sp.lambdify((t, theta, gamma, omega), d_gamma, 'numpy')
    f_omega = sp.lambdify((t, theta, gamma, omega), d_omega, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    omega_val = 2.0
    times = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    
    # Initialize embedder
    emb = PronyCosineEmbedder(theta_val, gamma_val, omega_val)
    
    # Get numerical derivatives
    derivatives = emb.kernel(times, nu=1)
    
    # Compare with symbolic results
    expected_theta = f_theta(times, theta_val, gamma_val, omega_val)
    expected_gamma = f_gamma(times, theta_val, gamma_val, omega_val)
    expected_omega = f_omega(times, theta_val, gamma_val, omega_val)
    
    assert_allclose(
        derivatives[0], expected_theta,
        rtol=1e-15, 
        err_msg="Derivative with respect to θ does not match symbolic result"
    )
    assert_allclose(
        derivatives[1], expected_gamma,
        rtol=1e-15,
        err_msg="Derivative with respect to γ does not match symbolic result"
    )
    assert_allclose(
        derivatives[2], expected_omega,
        rtol=1e-15,
        err_msg="Derivative with respect to ω does not match symbolic result"
    )

def test_kernel_x_gradient():
    """Test that kernel derivatives for mapped variables match symbolic differentiation results."""
    # Setup symbolic expressions
    t, x1, x2, x3 = sp.symbols('t x1 x2 x3', real=True)
    kernel = sp.exp(x1)**2 * sp.exp(-sp.exp(x2) * sp.Abs(t)) * sp.cos(sp.exp(x3)*t)
    
    # Compute symbolic derivatives
    d_x1 = sp.diff(kernel, x1)
    d_x2 = sp.diff(kernel, x2)
    d_x3 = sp.diff(kernel, x3)
    
    # Convert to numeric functions
    f_theta = sp.lambdify((t, x1, x2, x3), d_x1, 'numpy')
    f_gamma = sp.lambdify((t, x1, x2, x3), d_x2, 'numpy')
    f_omega = sp.lambdify((t, x1, x2, x3), d_x3, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    omega_val = 2.5
    times = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    
    # Initialize embedder
    emb = PronyCosineEmbedder(theta_val, gamma_val, omega_val)
    
    # Get numerical derivatives
    derivatives = emb.kernel(times, nu=1, mapped=True)
    
    # Compare with symbolic results
    x1, x2, x3 = emb.x
    expected_x1 = f_theta(times, x1, x2, x3)
    expected_x2 = f_gamma(times, x1, x2, x3)
    expected_x3 = f_omega(times, x1, x2, x3)
    
    assert_allclose(
        derivatives[0], expected_x1,
        rtol=1e-15, 
        err_msg="Derivative with respect to ln(θ) does not match symbolic result"
    )
    assert_allclose(
        derivatives[1], expected_x2,
        rtol=1e-15,
        err_msg="Derivative with respect to ln(γ) does not match symbolic result"
    )
    assert_allclose(
        derivatives[2], expected_x3,
        rtol=1e-15,
        err_msg="Derivative with respect to ln(ω) does not match symbolic result"
    )

def test_kernel_hessian():
    """Test that the second derivates of the kernel match symbolic differentiation."""
    # Setup symbolic expressions
    t, theta, gamma, omega = sp.symbols('t theta gamma omega', real=True)
    kernel = theta**2 * sp.exp(-gamma * sp.Abs(t)) * sp.cos(omega * t)
    
    # Compute symbolic derivatives
    d2_theta2 = sp.diff(kernel, theta, theta)
    d2_theta_gamma = sp.diff(kernel, theta, gamma)
    d2_theta_omega = sp.diff(kernel, theta, omega)
    d2_gamma2 = sp.diff(kernel, gamma, gamma)
    d2_gamma_omega = sp.diff(kernel, gamma, omega)
    d2_omega2 = sp.diff(kernel, omega, omega)
    
    # Convert to numeric functions
    f_theta2 = sp.lambdify((t, theta, gamma, omega), d2_theta2, 'numpy')
    f_theta_gamma = sp.lambdify((t, theta, gamma, omega), d2_theta_gamma, 'numpy')
    f_theta_omega = sp.lambdify((t, theta, gamma, omega), d2_theta_omega, 'numpy')
    f_gamma2 = sp.lambdify((t, theta, gamma, omega), d2_gamma2, 'numpy')
    f_gamma_omega = sp.lambdify((t, theta, gamma, omega), d2_gamma_omega, 'numpy')
    f_omega2 = sp.lambdify((t, theta, gamma, omega), d2_omega2, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    omega_val = 0.25
    times = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    
    # Initialize embedder
    emb = PronyCosineEmbedder(theta_val, gamma_val, omega_val)
    
    # Get numerical derivatives
    hessian = emb.kernel(times, nu=2)
    
    # Compare with symbolic results
    expected_theta2 = f_theta2(times, theta_val, gamma_val, omega_val)
    expected_theta_gamma = f_theta_gamma(times, theta_val, gamma_val, omega_val)
    expected_theta_omega = f_theta_omega(times, theta_val, gamma_val, omega_val)
    expected_gamma2 = f_gamma2(times, theta_val, gamma_val, omega_val)
    expected_gamma_omega = f_gamma_omega(times, theta_val, gamma_val, omega_val)
    expected_omega2 = f_omega2(times, theta_val, gamma_val, omega_val)
    
    assert_allclose(
        hessian[0,0], expected_theta2,
        rtol=1e-15, 
        err_msg="Second derivative with respect to θ does not match symbolic result"
    )
    assert_allclose(
        hessian[0,1], expected_theta_gamma,
        rtol=1e-15, 
        err_msg="Mixed derivative with respect to θ and γ does not match symbolic result"
    )
    assert_allclose(
        hessian[0,2], expected_theta_omega,
        rtol=1e-15, 
        err_msg="Mixed derivative with respect to θ and ω does not match symbolic result"
    )
    assert_allclose(
        hessian[1,1], expected_gamma2,
        rtol=1e-15,
        err_msg="Second derivative with respect to γ does not match symbolic result"
    )
    assert_allclose(
        hessian[1,2], expected_gamma_omega,
        rtol=1e-15,
        err_msg="Mixed derivative with respect to γ and ω does not match symbolic result"
    )
    assert_allclose(
        hessian[2,2], expected_omega2,
        rtol=1e-15,
        err_msg="Second derivative with respect to ω does not match symbolic result"
    )

def test_kernel_x_hessian():
    """Test that the second derivates of the kernel match symbolic differentiation for mapped variables."""
    # Setup symbolic expressions
    t, x1, x2, x3 = sp.symbols('t x1 x2 x3', real=True)
    kernel = sp.exp(x1)**2 * sp.exp(-sp.exp(x2) * sp.Abs(t)) * sp.cos(sp.exp(x3) * t)
    
    # Compute symbolic derivatives
    dx1x1 = sp.diff(kernel, x1, x1)
    dx1x2 = sp.diff(kernel, x1, x2)
    dx1x3 = sp.diff(kernel, x1, x3)
    dx2x2 = sp.diff(kernel, x2, x2)
    dx2x3 = sp.diff(kernel, x2, x3)
    dx3x3 = sp.diff(kernel, x3, x3)
    
    # Convert to numeric functions
    fx1x1 = sp.lambdify((t, x1, x2, x3), dx1x1, 'numpy')
    fx1x2 = sp.lambdify((t, x1, x2, x3), dx1x2, 'numpy')
    fx1x3 = sp.lambdify((t, x1, x2, x3), dx1x3, 'numpy')
    fx2x2 = sp.lambdify((t, x1, x2, x3), dx2x2, 'numpy')
    fx2x3 = sp.lambdify((t, x1, x2, x3), dx2x3, 'numpy')
    fx3x3 = sp.lambdify((t, x1, x2, x3), dx3x3, 'numpy')
    
    # Setup numerical test
    x1, x2, x3 = 0.0, -0.5, -1.0  # ln of theta, gamma, omega
    times = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    
    # Initialize embedder
    emb = PronyCosineEmbedder(np.exp(x1), np.exp(x2), np.exp(x3))
    
    # Get numerical derivatives
    hessian = emb.kernel(times, nu=2, mapped=True)
    
    # Compare with symbolic results
    x1, x2, x3 = emb.x
    expected_x1x1 = fx1x1(times, x1, x2, x3)
    expected_x1x2 = fx1x2(times, x1, x2, x3)
    expected_x1x3 = fx1x3(times, x1, x2, x3)
    expected_x2x2 = fx2x2(times, x1, x2, x3)
    expected_x2x3 = fx2x3(times, x1, x2, x3)
    expected_x3x3 = fx3x3(times, x1, x2, x3)
    
    assert_allclose(
        hessian[0,0], expected_x1x1,
        rtol=1e-15, 
        err_msg="Second derivative with respect to ln(θ) does not match symbolic result"
    )
    assert_allclose(
        hessian[0,1], expected_x1x2,
        rtol=1e-15, 
        err_msg="Mixed derivative with respect to ln(θ) and ln(γ) does not match symbolic result"
    )
    assert_allclose(
        hessian[0,2], expected_x1x3,
        rtol=1e-15, 
        err_msg="Mixed derivative with respect to ln(θ) and ln(ω) does not match symbolic result"
    )
    assert_allclose(
        hessian[1,1], expected_x2x2,
        rtol=1e-15,
        err_msg="Second derivative with respect to ln(γ) does not match symbolic result"
    )
    assert_allclose(
        hessian[1,2], expected_x2x3,
        rtol=1e-15,
        err_msg="Mixed derivative with respect to ln(γ) and ln(ω) does not match symbolic result"
    )
    assert_allclose(
        hessian[2,2], expected_x3x3,
        rtol=1e-15,
        err_msg="Second derivative with respect to ln(ω) does not match symbolic result"
    )

def test_spectrum_values():
    """Test that spectrum values match symbolic function."""
    # Setup symbolic expressions
    w, theta, gamma, omega = sp.symbols('omega theta gamma Omega', real=True)

    spectrum = (gamma * theta**2 / (gamma**2 + (w - omega)**2) + 
                gamma * theta**2 / (gamma**2 + (w + omega)**2))/2
    
    # Convert to numeric functions
    f_spectrum = sp.lambdify((w, theta, gamma, omega), spectrum, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    omega_val = 0.25
    freqs = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    
    # Initialize embedder
    emb = PronyCosineEmbedder(theta_val, gamma_val, omega_val)
    
    # Get numerical spectrum
    spectrum_vals = emb.spectrum(freqs)
    
    # Compare with symbolic results
    expected_spectrum = f_spectrum(freqs, theta_val, gamma_val, omega_val)
    
    assert_allclose(
        spectrum_vals, expected_spectrum,
        rtol=1e-15, 
        err_msg="Spectrum does not match symbolic result"
    )

def test_spectrum_gradient():
    """Test that spectrum derivatives match symbolic Fourier transforms."""
    # Setup symbolic expressions
    w, theta, gamma, omega = sp.symbols('omega theta gamma Omega', real=True)

    spectrum = (gamma * theta**2 / (gamma**2 + (w - omega)**2) + 
                gamma * theta**2 / (gamma**2 + (w + omega)**2))/2
    
    # Take derivatives of spectrum with respect to parameters
    d_theta = sp.diff(spectrum, theta)
    d_gamma = sp.diff(spectrum, gamma)
    d_omega = sp.diff(spectrum, omega)
    
    # Convert to numeric functions
    f_theta = sp.lambdify((w, theta, gamma, omega), d_theta, 'numpy')
    f_gamma = sp.lambdify((w, theta, gamma, omega), d_gamma, 'numpy')
    f_omega = sp.lambdify((w, theta, gamma, omega), d_omega, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    omega_val = 0.25
    frequencies = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    
    # Initialize embedder
    emb = PronyCosineEmbedder(theta_val, gamma_val, omega_val)
    
    # Get numerical derivatives
    derivatives = emb.spectrum(frequencies, nu=1)
    
    # Compare with symbolic results
    expected_theta = f_theta(frequencies, theta_val, gamma_val, omega_val)
    expected_gamma = f_gamma(frequencies, theta_val, gamma_val, omega_val)
    expected_omega = f_omega(frequencies, theta_val, gamma_val, omega_val)
    
    assert_allclose(
        derivatives[0], expected_theta,
        rtol=1e-14, 
        err_msg="Spectrum derivative with respect to θ does not match symbolic result"
    )
    assert_allclose(
        derivatives[1], expected_gamma,
        rtol=1e-14,
        err_msg="Spectrum derivative with respect to γ does not match symbolic result"
    )
    assert_allclose(
        derivatives[2], expected_omega,
        rtol=1e-14,
        err_msg="Spectrum derivative with respect to ω does not match symbolic result"
    )

def test_spectrum_hessian():
    """Test that the second derivatives of the spectrum match symbolic Fourier transforms."""
    w, theta, gamma, omega = sp.symbols('omega theta gamma Omega', real=True)

    spectrum = (gamma * theta**2 / (gamma**2 + (w - omega)**2) + 
                gamma * theta**2 / (gamma**2 + (w + omega)**2))/2
    
    
    # Take second derivatives of spectrum
    d2_theta2 = sp.diff(spectrum, theta, theta)
    d2_theta_gamma = sp.diff(spectrum, theta, gamma)
    d2_theta_omega = sp.diff(spectrum, theta, omega)
    d2_gamma2 = sp.diff(spectrum, gamma, gamma)
    d2_gamma_omega = sp.diff(spectrum, gamma, omega)
    d2_omega2 = sp.diff(spectrum, omega, omega)
    
    # Convert to numeric functions
    f_theta2 = sp.lambdify((w, theta, gamma, omega), d2_theta2, 'numpy')
    f_theta_gamma = sp.lambdify((w, theta, gamma, omega), d2_theta_gamma, 'numpy')
    f_theta_omega = sp.lambdify((w, theta, gamma, omega), d2_theta_omega, 'numpy')
    f_gamma2 = sp.lambdify((w, theta, gamma, omega), d2_gamma2, 'numpy')
    f_gamma_omega = sp.lambdify((w, theta, gamma, omega), d2_gamma_omega, 'numpy')
    f_omega2 = sp.lambdify((w, theta, gamma, omega), d2_omega2, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    omega_val = 0.25
    frequencies = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    
    # Initialize embedder
    emb = PronyCosineEmbedder(theta_val, gamma_val, omega_val)
    
    # Get numerical derivatives
    hessian = emb.spectrum(frequencies, nu=2)
    
    # Compare with symbolic results
    expected_theta2 = f_theta2(frequencies, theta_val, gamma_val, omega_val)
    expected_theta_gamma = f_theta_gamma(frequencies, theta_val, gamma_val, omega_val)
    expected_theta_omega = f_theta_omega(frequencies, theta_val, gamma_val, omega_val)
    expected_gamma2 = f_gamma2(frequencies, theta_val, gamma_val, omega_val)
    expected_gamma_omega = f_gamma_omega(frequencies, theta_val, gamma_val, omega_val)
    expected_omega2 = f_omega2(frequencies, theta_val, gamma_val, omega_val)
    
    assert_allclose(
        hessian[0,0], expected_theta2,
        rtol=1e-14, 
        err_msg="Second spectrum derivative with respect to θ does not match symbolic result"
    )
    assert_allclose(
        hessian[0,1], expected_theta_gamma,
        rtol=1e-14,
        err_msg="Mixed spectrum derivative with respect to θ and γ does not match symbolic result"
    )
    assert_allclose(
        hessian[0,2], expected_theta_omega,
        rtol=1e-14,
        err_msg="Mixed spectrum derivative with respect to θ and ω does not match symbolic result"
    )
    assert_allclose(
        hessian[1,1], expected_gamma2,
        rtol=1e-14,
        err_msg="Second spectrum derivative with respect to γ does not match symbolic result"
    )
    assert_allclose(
        hessian[1,2], expected_gamma_omega,
        rtol=1e-14,
        err_msg="Mixed spectrum derivative with respect to γ and ω does not match symbolic result"
    )
    assert_allclose(
        hessian[2,2], expected_omega2,
        rtol=1e-14,
        err_msg="Second spectrum derivative with respect to ω does not match symbolic result"
    )

if __name__ == "__main__":
    pytest.main([__file__])