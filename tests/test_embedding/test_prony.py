#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_prony.py
@Time    :   2025/09/08 15:06:27
@Author  :   George Trenins
@Desc    :   Test the Prony series embedder object
'''


from __future__ import print_function, division, absolute_import
import pytest
from glefit.embedding import PronyEmbedder
import numpy as np
import sympy as sp
from numpy.testing import assert_allclose


def test_kernel_invalid_nu():
    """Test that kernel raises ValueError when nu is not 0 or 1."""
    # Setup
    theta = 1.0
    gamma = 0.5
    emb = PronyEmbedder(theta, gamma)
    time = [-0.5, 0.0, 1.0]
    
    # Verify ValueError is raised with invalid nu
    nu = 3
    with pytest.raises(ValueError, match=f"Invalid value for nu = {nu}. Valid values are 0 and 1."):
        emb.kernel(time, nu=nu)


def test_kernel_values():
    """Test that kernel values match symbolic function."""
    # Setup symbolic expressions
    t, theta, gamma = sp.symbols('t theta gamma', real=True)
    kernel = theta**2 * sp.exp(-gamma * sp.Abs(t))
    
    # Convert to numeric functions
    f_kernel = sp.lambdify((t, theta, gamma), kernel, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    times = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    
    # Initialize embedder
    emb = PronyEmbedder(theta_val, gamma_val)
    
    # Get numerical derivatives
    kernel_vals = emb.kernel(times)
    
    # Compare with symbolic results
    expected_kernel = f_kernel(times, theta_val, gamma_val)
    
    assert_allclose(
        kernel_vals, expected_kernel,
        rtol=1e-15, 
        err_msg="Friction kernel does not match symbolic result"
    )

def test_kernel_gradient():
    """Test that kernel derivatives match symbolic differentiation results."""
    # Setup symbolic expressions
    t, theta, gamma = sp.symbols('t theta gamma', real=True)
    kernel = theta**2 * sp.exp(-gamma * sp.Abs(t))
    
    # Compute symbolic derivatives
    d_theta = sp.diff(kernel, theta)
    d_gamma = sp.diff(kernel, gamma)
    
    # Convert to numeric functions
    f_theta = sp.lambdify((t, theta, gamma), d_theta, 'numpy')
    f_gamma = sp.lambdify((t, theta, gamma), d_gamma, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    times = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    
    # Initialize embedder
    emb = PronyEmbedder(theta_val, gamma_val)
    
    # Get numerical derivatives
    derivatives = emb.kernel(times, nu=1)
    
    # Compare with symbolic results
    expected_theta = f_theta(times, theta_val, gamma_val)
    expected_gamma = f_gamma(times, theta_val, gamma_val)
    
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

def test_kernel_hessian():
    """Test that the second derivates of the kernel match symbolic differentiation."""
    # Setup symbolic expressions
    t, theta, gamma = sp.symbols('t theta gamma', real=True)
    kernel = theta**2 * sp.exp(-gamma * sp.Abs(t))
    
    # Compute symbolic derivatives
    d2_theta2 = sp.diff(kernel, theta, theta)
    d2_theta_gamma = sp.diff(kernel, theta, gamma)
    d2_gamma2 = sp.diff(kernel, gamma, gamma)
    
    # Convert to numeric functions
    f_theta2 = sp.lambdify((t, theta, gamma), d2_theta2, 'numpy')
    f_theta_gamma = sp.lambdify((t, theta, gamma), d2_theta_gamma, 'numpy')
    f_gamma2 = sp.lambdify((t, theta, gamma), d2_gamma2, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    times = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    
    # Initialize embedder
    emb = PronyEmbedder(theta_val, gamma_val)
    
    # Get numerical derivatives
    hessian = emb.kernel(times, nu=2)
    
    # Compare with symbolic results
    expected_theta2 = f_theta2(times, theta_val, gamma_val)
    expected_theta_gamma = f_theta_gamma(times, theta_val, gamma_val)
    expected_gamma2 = f_gamma2(times, theta_val, gamma_val)
    
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
        hessian[1,1], expected_gamma2,
        rtol=1e-15,
        err_msg="Second derivative with respect to γ does not match symbolic result"
    )
    

def test_spectrum_invalid_nu():
    """Test that kernel raises ValueError when nu is not 0 or 1."""
    # Setup
    theta = 1.0
    gamma = 0.5
    emb = PronyEmbedder(theta, gamma)
    freq = [-0.5, 0.0, 1.0]
    
    # Verify ValueError is raised with invalid nu
    nu = 3
    with pytest.raises(ValueError, match=f"Invalid value for nu = {nu}. Valid values are 0 and 1."):
        emb.spectrum(freq, nu=nu)


def test_spectrum_values():
    """Test that spectrum values match symbolic function."""
    # Setup symbolic expressions
    t, w, theta, gamma = sp.symbols('t omega theta gamma', real=True)
    kernel = theta**2 * sp.exp(-gamma * sp.Abs(t))

    # Compute cosine transform symbolically
    # sympy returns sqrt(2/π) ∫_{0}^{∞} K(t) cos(ωt) dt - NOTE as of 08/09/25 sympy's docs are wrong
    spectrum = sp.cosine_transform(kernel, t, w)
    # we use K(ω) = ∫_{0}^{∞} K(t) cos(ωt) dt
    spectrum = spectrum * sp.sqrt(sp.pi/2) 
    
    # Convert to numeric functions
    f_spectrum = sp.lambdify((w, theta, gamma), spectrum, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    freqs = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    
    # Initialize embedder
    emb = PronyEmbedder(theta_val, gamma_val)
    
    # Get numerical derivatives
    spectrum_vals = emb.spectrum(freqs)
    
    # Compare with symbolic results
    expected_kernel = f_spectrum(freqs, theta_val, gamma_val)
    
    assert_allclose(
        spectrum_vals, expected_kernel,
        rtol=1e-15, 
        err_msg="Friction spectrum does not match symbolic result"
    )


def test_spectrum_gradient():
    """Test that spectrum derivatives match symbolic Fourier transforms."""
    # Setup symbolic expressions
    t, w, theta, gamma = sp.symbols('t omega theta gamma ', real=True, positive=True)
    kernel = theta**2 * sp.exp(-gamma * t)
    
    # Compute cosine transform symbolically
    # sympy returns sqrt(2/π) ∫_{0}^{∞} K(t) cos(ωt) dt - NOTE as of 08/09/25 sympy's docs are wrong
    spectrum = sp.cosine_transform(kernel, t, w)
    # we use K(ω) = ∫_{0}^{∞} K(t) cos(ωt) dt
    spectrum = spectrum * sp.sqrt(sp.pi/2) 
    
    # Take derivatives of spectrum with respect to parameters
    d_theta_spectrum = sp.diff(spectrum, theta)
    d_gamma_spectrum = sp.diff(spectrum, gamma)
    
    # Convert to numeric functions
    f_theta = sp.lambdify((w, theta, gamma), d_theta_spectrum, 'numpy')
    f_gamma = sp.lambdify((w, theta, gamma), d_gamma_spectrum, 'numpy')
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    frequencies = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    
    # Initialize embedder
    emb = PronyEmbedder(theta_val, gamma_val)
    
    # Get numerical derivatives
    derivatives = emb.spectrum(frequencies, nu=1)
    
    # Compare with symbolic results
    expected_theta = f_theta(frequencies, theta_val, gamma_val)
    expected_gamma = f_gamma(frequencies, theta_val, gamma_val)
    
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

def test_spectrum_hessian():
    """Test that the second derivatives of the spectrum match symbolic Fourier transforms."""
    # Setup symbolic expressions
    t, w, theta, gamma = sp.symbols('t omega theta gamma ', real=True, positive=True)
    kernel = theta**2 * sp.exp(-gamma * t)
    
    # Compute cosine transform symbolically
    # sympy returns sqrt(2/π) ∫_{0}^{∞} K(t) cos(ωt) dt - NOTE as of 08/09/25 sympy's docs are wrong
    spectrum = sp.cosine_transform(kernel, t, w)
    # we use K(ω) = ∫_{0}^{∞} K(t) cos(ωt) dt
    spectrum = spectrum * sp.sqrt(sp.pi/2) 
    
    # Take derivatives of spectrum with respect to parameters
    d2_theta2_spectrum = sp.diff(spectrum, theta, theta)
    d2_theta_gamma_spectrum = sp.diff(spectrum, theta, gamma)
    d2_gamma2_spectrum = sp.diff(spectrum, gamma, gamma)
    
    # Convert to numeric functions
    f_theta2 = sp.lambdify((w, theta, gamma), d2_theta2_spectrum, 'numpy')
    f_theta_gamma = sp.lambdify((w, theta, gamma), d2_theta_gamma_spectrum, 'numpy')
    f_gamma2 = sp.lambdify((w, theta, gamma), d2_gamma2_spectrum, 'numpy')

    print(d2_gamma2_spectrum)
    
    # Setup numerical test
    theta_val = 1.0
    gamma_val = 0.5
    frequencies = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    
    # Initialize embedder
    emb = PronyEmbedder(theta_val, gamma_val)
    
    # Get numerical derivatives
    hessian = emb.spectrum(frequencies, nu=2)
    
    # Compare with symbolic results
    expected_theta2 = f_theta2(frequencies, theta_val, gamma_val)
    expected_theta_gamma = f_theta_gamma(frequencies, theta_val, gamma_val)
    expected_gamma2 = f_gamma2(frequencies, theta_val, gamma_val)
    
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
        hessian[1,1], expected_gamma2,
        rtol=1e-14,
        err_msg="Second spectrum derivative with respect to γ does not match symbolic result"
    )


if __name__ == "__main__":
    pytest.main([__file__])