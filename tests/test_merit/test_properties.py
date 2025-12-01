#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_properties.py
@Time    :   2025/09/11 14:01:34
@Author  :   George Trenins
@Desc    :   Test properties of the GLE embedding
'''


from __future__ import print_function, division, absolute_import
import pytest
import numpy as np
from scipy.linalg import expm
from glefit.merit import MemoryKernel, MemorySpectrum
from glefit.embedding import PronyEmbedder, MultiEmbedder, PronyCosineEmbedder, TwoAuxEmbedder
from glefit.utils.numderiv import jacobian


#---------- NUMERICAL DERIVATIVES FOR TESTING -----------#

def fd_grad_thetaT_expA_theta(A, tau, theta):
    """
    Compute gradient of f(A) = θᵀ exp(-τ A) θ w.r.t. A using glefit.utils.numderiv.jacobian.

    Parameters
    ----------
    A : (n, n) array_like
        Matrix argument.
    tau : float
        Scalar τ.
    theta : (n,) array_like
        Vector θ.

    Returns
    -------
    G : (n, n) ndarray
        Numerical gradient ∂/∂A of θᵀ exp(-τ A) θ.
    """
    A = np.asarray(A, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n = A.shape[0]
    if A.shape != (n, n) or theta.shape != (n,):
        raise ValueError("Shapes must be A:(n,n), theta:(n,)")

    def f(A_flat):
        """Function to differentiate: f(vec(A)) = θᵀ exp(-τ A) θ"""
        A_mat = A_flat.reshape((n, n))
        return np.array([np.linalg.multi_dot([theta, expm(-tau * A_mat), theta])])
    # Flatten A for derivative calculation
    A_flat = A.reshape(-1)
    J = jacobian(f, A_flat, order=4)
    # Reshape gradient back to matrix form
    G = J.reshape((n, n))
    return G

#--------------------------------------------------------#

class TestKernel:

    def test_memory_kernel_value(self):
        time = np.asarray([0.0, 0.5, 1.0, 4.0])
        prony_thetas = np.asarray([1.0, 2.0])
        prony_gammas = np.asarray([0.5, 0.25])
        prony_embs = [PronyEmbedder(theta, gamma) 
                    for theta, gamma in zip(prony_thetas, prony_gammas)]
        osc_thetas = np.asarray([0.9, 1.5])
        osc_gammas = np.asarray([0.75, 0.15])
        osc_omegas = np.asarray([1.0, 2.0])
        osc_embs = [PronyCosineEmbedder(theta, gamma, omega) 
                    for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)]
        multi_emb = MultiEmbedder(prony_embs + osc_embs)
        ref_kernel = multi_emb.kernel(time)
        kernel_object = MemoryKernel(time, ref_kernel, multi_emb, "squared")
        kernel_value = kernel_object.value
        np.testing.assert_allclose(
            ref_kernel, kernel_value,
            rtol=1e-10, atol=1e-12,
            err_msg="Kernel expressions via parameters and A matrix do not match"
        )

    def test_memory_kernel_A_gradient(self):
        """Test gradient of theta^T exp(-tau A) theta w.r.t. A elements."""
        time = np.asarray([0.5, 1.0, 4.0])
        
        # Setup embedders as before
        prony_thetas = np.asarray([1.0, 2.0])
        prony_gammas = np.asarray([0.5, 0.25])
        osc_thetas = np.asarray([3.0, 4.0])
        osc_gammas = np.asarray([0.1, 0.05])
        osc_omegas = np.asarray([1.0, 2.0])
        embs = ([PronyEmbedder(theta, gamma) 
                for theta, gamma in zip(prony_thetas, prony_gammas)] +
                [PronyCosineEmbedder(theta, gamma, omega) 
                for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)])
        multi_emb = MultiEmbedder(embs)
        Ap = multi_emb.drift_matrix
        A = Ap[1:,1:]
        theta = Ap[0,1:]
        ref_grad_A = []
        for tau in time:
            ref_grad_A.append(fd_grad_thetaT_expA_theta(A, tau, theta))
        ref_grad_A = np.stack(ref_grad_A, axis=0)
        kernel_object = MemoryKernel(time, np.ones_like(time), multi_emb, "squared")
        num_grad_A = kernel_object._grad_thetaT_expA_theta(A, theta, time)[1]
        np.testing.assert_allclose(
            ref_grad_A, num_grad_A,
            rtol=1e-8,
            err_msg="Kernel gradients w.r.t. drift matrix do not match"
        )

    def test_memory_kernel_param_gradient(self):
        """Test parameter gradients of the kernel."""
        time = np.asarray([0.0, 0.5, 1.0, 4.0, 10.0])
        prony_thetas = np.asarray([1.0, 2.0])
        prony_gammas = np.asarray([0.5, 0.25])
        osc_thetas = np.asarray([3.0, 4.0])
        osc_gammas = np.asarray([0.1, 0.05])
        osc_omegas = np.asarray([1.0, 2.0])
        
        embs = ([PronyEmbedder(theta, gamma) 
                for theta, gamma in zip(prony_thetas, prony_gammas)] +
                [PronyCosineEmbedder(theta, gamma, omega) 
                for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)])
        multi_emb = MultiEmbedder(embs)
        ref_kernel_gradient = multi_emb.kernel(time, nu=1, mapped=True)
        kernel_object = MemoryKernel(time, np.ones_like(time), multi_emb, "squared")
        num_kernel_gradient = kernel_object.grad_wrt_params()
        np.testing.assert_allclose(
            num_kernel_gradient, ref_kernel_gradient,
            rtol=1e-14,
            err_msg="Kernel parameter gradients do not match"
        )

    def test_memory_kernel_distance_gradient(self):
        """Test gradient of the distance metric."""
        time = np.asarray([0.0, 0.5, 1.0, 4.0, 10.0])
        prony_thetas = np.asarray([1.0, 2.0])
        prony_gammas = np.asarray([0.5, 0.25])
        osc_thetas = np.asarray([3.0, 4.0])
        osc_gammas = np.asarray([0.1, 0.05])
        osc_omegas = np.asarray([1.0, 2.0])
        embs = ([PronyEmbedder(theta, gamma) 
                for theta, gamma in zip(prony_thetas, prony_gammas)] +
                [PronyCosineEmbedder(theta, gamma, omega) 
                for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)])
        multi_emb = MultiEmbedder(embs)
        ref_kernel = multi_emb.kernel(time)
        rng = np.random.default_rng(seed=31415)
        target = ref_kernel * rng.normal(loc=1.0, scale=0.2, size=ref_kernel.shape)
        kernel_object = MemoryKernel(time, target, multi_emb, metric="squared")
        x = multi_emb.x
        
        def distance_func(y):
            return np.array([np.sum(kernel_object.distance_metric(
                kernel_object.function(x=y), target))])
        
        ref_distance_gradient = jacobian(distance_func, x, order=4).flatten()
        distance_gradient = kernel_object.gradient()
        
        np.testing.assert_allclose(
            distance_gradient, ref_distance_gradient,
            rtol=1.0e-6, atol=1.0e-8,
            err_msg="Distance metric gradients do not match"
        )

    def test_memory_kernel_param_hessian(self):
        time = np.asarray([0.0, 0.5, 1.0, 4.0, 10.0])
        prony_thetas = np.asarray([1.0, 2.0])
        prony_gammas = np.asarray([0.5, 0.25])
        osc_thetas = np.asarray([3.0, 4.0])
        osc_gammas = np.asarray([0.1, 0.05])
        osc_omegas = np.asarray([1.0, 2.0])
        
        embs = ([PronyEmbedder(theta, gamma) 
                for theta, gamma in zip(prony_thetas, prony_gammas)] +
                [PronyCosineEmbedder(theta, gamma, omega) 
                for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)])
        multi_emb = MultiEmbedder(embs)
        ref_kernel_hessian = multi_emb.kernel(time, nu=2, mapped=True)
        kernel_object = MemoryKernel(time, np.ones_like(time), multi_emb, "squared")
        _, num_kernel_hessian = kernel_object.gradhess_wrt_params()
        np.testing.assert_allclose(
            num_kernel_hessian, ref_kernel_hessian,
            rtol=1e-8, atol=1e-10,
            err_msg="Kernel parameter Hessians do not match"
        )
    
class TestSpectrum:

    def test_memory_spectrum_value(self):
        omega = np.asarray([0.0, 0.5, 1.0, 4.0])
        prony_thetas = np.asarray([1.0, 2.0])
        prony_gammas = np.asarray([0.5, 0.25])
        prony_embs = [PronyEmbedder(theta, gamma) 
                    for theta, gamma in zip(prony_thetas, prony_gammas)]
        osc_thetas = np.asarray([0.9, 1.5])
        osc_gammas = np.asarray([0.75, 0.15])
        osc_omegas = np.asarray([1.0, 2.0])
        osc_embs = [PronyCosineEmbedder(theta, gamma, omega) 
                    for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)]
        gen_emb = TwoAuxEmbedder([1.0,-0.5], 2.0, 1.2, 3.3, sigma=20.0, threshold=30.0)
        multi_emb = MultiEmbedder(prony_embs + osc_embs + [gen_emb])
        ref_specturm = multi_emb.spectrum(omega)
        spectrum_object = MemorySpectrum(omega, ref_specturm, multi_emb, "squared")
        spectrum_value = spectrum_object.value
        np.testing.assert_allclose(
            ref_specturm, spectrum_value,
            rtol=1e-10, atol=1e-12,
            err_msg="Kernel expressions via parameters and A matrix do not match"
        )

    def test_spectrum_generic_A(self):
        rng = np.random.default_rng(seed=31415)
        n = 10
        theta = rng.uniform(low=-10, high=10, size=n)
        A = rng.uniform(low=-10, high=10, size=(n,n))
        omega = np.linspace(0, 50, 100)
        reference = np.asarray([np.linalg.multi_dot(
            [theta, A, np.linalg.inv(A @ A + np.eye(n)*w**2), theta]
        ) for w in omega])
        Ap = np.zeros((n+1,n+1))
        Ap[0,1:] = theta
        Ap[1:,0] = -theta
        Ap[1:,1:] = A
        actual = MemorySpectrum._compute_spec_from_A(Ap, omega)
        np.testing.assert_allclose(
            actual, reference,
            rtol=1e-10, atol=1e-12,
            err_msg="Kernel expressions via parameters and A matrix do not match"
        )

    def test_memory_spectrum_param_gradient(self):
        """Test parameter gradients of the spectrum."""
        omega = np.asarray([0.0, 0.5, 1.0, 4.0, 10.0])
        prony_thetas = np.asarray([1.0, 2.0])
        prony_gammas = np.asarray([0.5, 0.25])
        prony_embs = [PronyEmbedder(theta, gamma) 
                    for theta, gamma in zip(prony_thetas, prony_gammas)]
        osc_thetas = np.asarray([0.9, 1.5])
        osc_gammas = np.asarray([0.75, 0.15])
        osc_omegas = np.asarray([1.0, 2.0])
        osc_embs = [PronyCosineEmbedder(theta, gamma, omega) 
                    for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)]
        gen_emb = TwoAuxEmbedder([1.0,-0.5], 2.0, 1.2, 3.3, sigma=20.0, threshold=30.0)
        multi_emb = MultiEmbedder(prony_embs + osc_embs + [gen_emb])
        ref_spectrum_gradient = multi_emb.spectrum(omega, nu=1, mapped=True)
        spectrum_object = MemorySpectrum(omega, np.ones_like(omega), multi_emb, "squared")
        num_spectrum_gradient = spectrum_object.grad_wrt_params()
        np.testing.assert_allclose(
            num_spectrum_gradient, ref_spectrum_gradient,
            rtol=1e-14,
            err_msg="Spectrum parameter gradients do not match"
        )

    def test_memory_spectrum_distance_gradient(self):
        """Test gradient of the spectrum distance metric."""
        omega = np.asarray([0.0, 0.5, 1.0, 4.0, 10.0])
        prony_thetas = np.asarray([1.0, 2.0])
        prony_gammas = np.asarray([0.5, 0.25])
        prony_embs = [PronyEmbedder(theta, gamma) 
                    for theta, gamma in zip(prony_thetas, prony_gammas)]
        osc_thetas = np.asarray([0.9, 1.5])
        osc_gammas = np.asarray([0.75, 0.15])
        osc_omegas = np.asarray([1.0, 2.0])
        osc_embs = [PronyCosineEmbedder(theta, gamma, omega) 
                    for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)]
        gen_emb = TwoAuxEmbedder([1.0,-0.5], 2.0, 1.2, 3.3, sigma=20.0, threshold=30.0)
        multi_emb = MultiEmbedder(prony_embs + osc_embs + [gen_emb])
        ref_spectrum = multi_emb.spectrum(omega)
        rng = np.random.default_rng(seed=31415)
        target = ref_spectrum * rng.normal(loc=1.0, scale=0.2, size=ref_spectrum.shape)
        spectrum_object = MemorySpectrum(omega, target, multi_emb, metric="squared")
        x = multi_emb.x
        

        def distance_func(y):
            return np.array([np.sum(spectrum_object.distance_metric(
                spectrum_object.function(x=y), target))])
        
        ref_distance_gradient = jacobian(distance_func, x, order=4).flatten()
        distance_gradient = spectrum_object.gradient()
        
        np.testing.assert_allclose(
            distance_gradient, ref_distance_gradient,
            rtol=1e-7,
            err_msg="Spectrum distance metric gradients do not match"
        )

    def test_memory_spectrum_param_hessian(self):
        """Test parameter Hessians of the spectrum."""
        omega = np.asarray([0.0, 0.5, 1.0, 4.0, 10.0])
        prony_thetas = np.asarray([1.0, 2.0])
        prony_gammas = np.asarray([0.5, 0.25])
        prony_embs = [PronyEmbedder(theta, gamma) 
                    for theta, gamma in zip(prony_thetas, prony_gammas)]
        osc_thetas = np.asarray([0.9, 1.5])
        osc_gammas = np.asarray([0.75, 0.15])
        osc_omegas = np.asarray([1.0, 2.0])
        osc_embs = [PronyCosineEmbedder(theta, gamma, omega) 
                    for theta, gamma, omega in zip(osc_thetas, osc_gammas, osc_omegas)]
        gen_emb = TwoAuxEmbedder([1.0,-0.5], 2.0, 1.2, 3.3, sigma=20.0, threshold=30.0)
        multi_emb = MultiEmbedder(prony_embs + osc_embs + [gen_emb])
        ref_spectrum_hessian = multi_emb.spectrum(omega, nu=2, mapped=True)
        spectrum_object = MemorySpectrum(omega, np.ones_like(omega), multi_emb, "squared")
        _, num_spectrum_hessian = spectrum_object.gradhess_wrt_params()
        np.testing.assert_allclose(
            num_spectrum_hessian, ref_spectrum_hessian,
            rtol=1e-8, atol=1e-10,
            err_msg="Spectrum parameter Hessians do not match"
        )

if __name__ == "__main__":
    pytest.main([__file__])
    # t = TestKernel()
    # t.test_memory_kernel_distance_gradient()
    # t = TestSpectrum()
    # t.test_memory_spectrum_distance_gradient()