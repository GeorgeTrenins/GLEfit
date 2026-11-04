#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_properties.py
@Time    :   2026/11/04 18:30:00
@Author  :   George Trenins
@Desc    :   Test the gradients of K(t) and Λ(ω) with respect to the drift matrix
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
from numpy.testing import assert_allclose
from glefit.embedding import PronyEmbedder
from glefit.merit import MemoryKernel, MemorySpectrum


RTOL = 1.0e-12
# for entries whose exact derivative is zero, such as Ap[0, 0]
ATOL = 1.0e-13
TIME = np.array([0.0, 0.4, 1.3, 3.0])
FREQ = np.array([0.0, 0.7, 2.0, 5.0])

# Ap = [[0, θᵀ], [-θ, A]].
DRIFT_MATRICES = {
    "symmetric-A": [[0.0, 0.8, -0.5],
                    [-0.8, 1.3, 0.4],
                    [0.5, 0.4, 0.9]],
    "one-theta-entry": [[0.0, 1.1, 0.0],
                        [-1.1, 1.2, 0.7],
                        [0.0, -0.7, 1.2]],
    # γ = δ = 1.5, Ω = 2: s₂ is undamped
    "gen2x2-maximal-gauge": [[0.0, 1.0, 0.8],
                             [-1.0, 3.0, 2.0],
                             [-0.8, -2.0, 0.0]],
    "general-3-variables": [[0.0, 0.9, -0.4, 0.6],
                            [-0.9, 1.5, 0.3, -0.2],
                            [0.4, -0.6, 1.1, 0.4],
                            [-0.6, 0.5, -0.1, 0.8]],
}


@pytest.fixture
def placeholder_emb():
    return PronyEmbedder(1.0, 1.0)


def unit_directions(size):
    """Every unit matrix E_ij of shape (size, size), in row-major order of (i, j)."""
    return np.eye(size * size).reshape(-1, size, size)


@pytest.mark.parametrize("Ap", list(DRIFT_MATRICES.values()), ids=list(DRIFT_MATRICES))
@pytest.mark.parametrize("prop_class, method, grid",
                         [(MemoryKernel, "kernel", TIME), (MemorySpectrum, "spectrum", FREQ)],
                         ids=["kernel", "spectrum"])
def test_grad_wrt_A_matches_exact_derivative_for_every_entry_of_Ap(
        prop_class, method, grid, Ap, placeholder_emb, drift_matrix_reference):
    Ap = np.array(Ap)
    prop = prop_class(grid, np.zeros_like(grid), placeholder_emb, "squared")
    _, expected = getattr(drift_matrix_reference, method)(Ap, grid, unit_directions(len(Ap)))

    actual = prop.grad_wrt_A(A=Ap)

    assert_allclose(np.moveaxis(actual, 0, -1).reshape(expected.shape), expected,
                    rtol=RTOL, atol=ATOL)
