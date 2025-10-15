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
            emb = TwoAuxEmbedder.from_matrix(theta, A)
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





if __name__ == "__main__":
    pytest.main([__file__])
