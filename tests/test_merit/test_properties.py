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
from scipy.optimize._numdiff import approx_derivative
from glefit.merit import MemoryKernel
from glefit.embedding import PronyEmbedder, MultiEmbedder


#---------- NUMERICAL DERIVATIVES FOR TESTING -----------#

def fd_grad_thetaT_expA_theta(A, tau, theta, h=None, relative=True):
    """
    Central finite-difference gradient of f(A) = θᵀ exp(-τ A) θ w.r.t. A.

    Parameters
    ----------
    A : (n, n) array_like
        Matrix argument.
    tau : float
        Scalar τ.
    theta : (n,) array_like
        Vector θ.
    h : float or None, optional
        Base stepsize. If None, uses (eps)**(1/3) scaled per entry (good for central differences).
    relative : bool, optional
        If True, uses stepsize h * max(1, |A_ij|) per entry; else uses absolute stepsize h.

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

    # Default stepsize for central differences
    if h is None:
        h_base = (np.finfo(float).eps)**(1/3)  # ~6e-6
    else:
        h_base = float(h)

    def f(M):
        return float(theta @ (expm(-tau * M) @ theta))

    G = np.empty_like(A, dtype=float)

    for i in range(n):
        for j in range(n):
            hij = h_base * (max(1.0, abs(A[i, j])) if relative else 1.0)
            # Forward
            Aplus = A.copy()
            Aplus[i, j] += hij
            fplus = f(Aplus)
            # Backward
            Aminus = A.copy()
            Aminus[i, j] -= hij
            fminus = f(Aminus)
            # Central difference
            G[i, j] = (fplus - fminus) / (2.0 * hij)

    return G

#--------------------------------------------------------#

def test_memory_kernel_value():
    time = np.asarray([0.0, 0.5, 1.0, 4.0])
    thetas = np.asarray([1.0, 2.0, 3.0])
    gammas = np.asarray([0.5, 0.25, 0.1])
    # reference:
    embs = [PronyEmbedder(theta, gamma) for theta, gamma in zip(thetas, gammas)]
    multi_emb = MultiEmbedder(embs)
    ref_kernel = multi_emb.kernel(time)
    # numerical
    kernel_object = MemoryKernel(time, ref_kernel, multi_emb, "squared")
    kernel_value = kernel_object.value
    # Verify that the two kernels are identical
    np.testing.assert_allclose(
        ref_kernel, kernel_value,
        rtol=1e-14,
        err_msg="Expressions for the kernel in terms of optimization parameters and the A matrix do not match"
    )


def test_memory_kernel_A_gradient():
    """Test the implementation of the derivative of theta^T exp(-tau A) theta with respect to the elements of A
    """
    time = np.asarray([0.5, 1.0, 4.0])
    thetas = np.asarray([1.0, 2.0, 3.0])
    gammas = np.asarray([0.5, 0.25, 0.1])
    # reference:
    embs = [PronyEmbedder(theta, gamma) for theta, gamma in zip(thetas, gammas)]
    multi_emb = MultiEmbedder(embs)
    Ap = multi_emb.drift_matrix
    A = Ap[1:,1:]
    theta = Ap[0,1:]
    ref_grad_A = []
    for tau in time:
        ref_grad_A.append(fd_grad_thetaT_expA_theta(A, tau, theta))
    ref_grad_A = np.stack(ref_grad_A, axis=0)
    # numerical:
    kernel_object = MemoryKernel(
        time, np.ones_like(time), multi_emb, "squared")
    num_grad_A = kernel_object._grad_thetaT_expA_theta(A, theta, time)[1]
    np.testing.assert_allclose(
        ref_grad_A, num_grad_A,
        rtol=1e-8,
        err_msg="Expressions for the kernel gradient with respect to the drift matrix do not match"
    ) 
    

def test_memory_kernel_param_gradient():
    time = np.asarray([0.0, 0.5, 1.0, 4.0, 10.0])
    thetas = np.asarray([1.0, 2.0, 3.0])
    gammas = np.asarray([0.5, 0.25, 0.1])
    # reference:
    embs = [PronyEmbedder(theta, gamma) for theta, gamma in zip(thetas, gammas)]
    multi_emb = MultiEmbedder(embs)
    ref_kernel_gradient = multi_emb.kernel(time, nu=1)
    # numerical
    kernel_object = MemoryKernel(time, np.ones_like(time), multi_emb, "squared")
    num_kernel_gradient = kernel_object._grad_wrt_params()
    # Verify that the two kernels are identical
    np.testing.assert_allclose(
        num_kernel_gradient, ref_kernel_gradient,
        rtol=1e-14,
        err_msg="Expressions for the kernel gradient with respect to optimizable parameters do not match"
    )


def test_memory_kernel_distance_gradient():
    # System setup
    time = np.asarray([0.0, 0.5, 1.0, 4.0, 10.0])
    thetas = np.asarray([1.0, 2.0, 3.0])
    gammas = np.asarray([0.5, 0.25, 0.1])
    embs = [PronyEmbedder(theta, gamma) for theta, gamma in zip(thetas, gammas)]
    multi_emb = MultiEmbedder(embs)
    ref_kernel = multi_emb.kernel(time)
    rng = np.random.default_rng(seed=31415)
    # Introduce "noise"
    target = ref_kernel * rng.normal(
        loc=1.0, 
        scale=0.2, 
        size=ref_kernel.shape)
    kernel_object = MemoryKernel(time, target, multi_emb, "squared")
    # reference (finite difference)
    # define function for finite difference
    def fun(v):
        # get a reference to the embedder object
        emb = kernel_object.emb
        # store current value of parameters
        cache = emb.params
        # compute value of kernel for parameter vector v
        emb.params = v
        kernel = kernel_object.value
        # restore original params
        emb.params = cache
        # compute the distane
        return kernel_object.distance_metric(kernel, kernel_object.target)
    ref_distance_gradient = approx_derivative(fun, multi_emb.params, method='3-point')
    # test
    distance_gradient = kernel_object.gradient()
    # Verify that the two kernels are identical
    np.testing.assert_allclose(
        distance_gradient, ref_distance_gradient,
        rtol=1e-7,
        err_msg="Expressions for gradient of the distance betwixt the target and actual kernel w.r.t. optimizable parameters do not match."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
