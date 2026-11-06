#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_properties.py
@Time    :   2026/11/04 18:30:00
@Author  :   George Trenins
@Desc    :   Test K(t), Λ(ω) and their gradients w.r.t. the drift matrix
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import pytest
from numpy.testing import assert_allclose
from glefit.embedding import PronyEmbedder, TwoAuxEmbedder
from glefit.mappers import IdentityMapper, LowerBoundMapper
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
    # γ = δ = 1.5, ζ = 2: s₂ is undamped
    "gen2x2-maximal-gauge": [[0.0, 1.0, 0.8],
                             [-1.0, 3.0, 2.0],
                             [-0.8, -2.0, 0.0]],
    # γ = δ = Ω = 1: A has the double eigenvalue 1 and no second
    # eigenvector, so it cannot be diagonalized
    "gen2x2-critical-damping": [[0.0, 1.0, 0.8],
                                [-1.0, 2.0, 1.0],
                                [-0.8, -1.0, 0.0]],
    "general-3-variables": [[0.0, 0.9, -0.4, 0.6],
                            [-0.9, 1.5, 0.3, -0.2],
                            [0.4, -0.6, 1.1, 0.4],
                            [-0.6, 0.5, -0.1, 0.8]],
    # The left column is not -θ, which no embedder builds. K and Λ then
    # take θᵀ from the top row and θ from the negated left column, as
    # the dynamics do.
    "row-differs-from-column": [[0.0, 0.9, -0.4],
                                [-0.5, 1.5, 0.3],
                                [0.7, -0.6, 1.1]],
}
PROPERTIES = [(MemoryKernel, "kernel", TIME),
              (MemorySpectrum, "spectrum", FREQ)]


@pytest.fixture
def placeholder_emb():
    return PronyEmbedder(1.0, 1.0)


def unit_directions(size):
    """Every unit matrix E_ij of shape (size, size), in row-major order of (i, j)."""
    return np.eye(size * size).reshape(-1, size, size)


class FixedDriftMatrix:
    """Stand-in for an embedder whose drift matrix no embedder builds.
    function() with x=None reads only the attribute A."""

    def __init__(self, Ap):
        self.A = np.array(Ap)


@pytest.mark.parametrize("Ap", list(DRIFT_MATRICES.values()), ids=list(DRIFT_MATRICES))
@pytest.mark.parametrize("prop_class, method, grid", PROPERTIES,
                         ids=["kernel", "spectrum"])
def test_grad_wrt_A_matches_exact_derivative_for_every_entry_of_Ap(
        prop_class, method, grid, Ap, placeholder_emb, drift_matrix_reference):
    Ap = np.array(Ap)
    prop = prop_class(grid, np.zeros_like(grid), placeholder_emb, "squared")
    _, expected = getattr(drift_matrix_reference, method)(Ap, grid, unit_directions(len(Ap)))

    actual = prop.grad_wrt_A(A=Ap)

    assert_allclose(np.moveaxis(actual, 0, -1).reshape(expected.shape), expected,
                    rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("prop_class, method, grid", PROPERTIES,
                         ids=["kernel", "spectrum"])
def test_function_reads_theta_from_the_left_column(
        prop_class, method, grid, drift_matrix_reference):
    Ap = np.array(DRIFT_MATRICES["row-differs-from-column"])
    emb = FixedDriftMatrix(Ap)
    prop = prop_class(grid, np.zeros_like(grid), emb, "squared")
    expected, _ = getattr(drift_matrix_reference, method)(Ap, grid)

    actual = prop.function()

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("prop_class, method, grid", PROPERTIES,
                         ids=["kernel", "spectrum"])
def test_function_at_critical_damping_matches_drift_matrix_reference(
        prop_class, method, grid, drift_matrix_reference):
    # δ = γ is already the maximal gauge, and ζ = γ gives Ω = γ, so the
    # drift matrix is "gen2x2-critical-damping" above
    emb = TwoAuxEmbedder([1.0, 0.8], 1.0, 1.0, 1.0)
    prop = prop_class(grid, np.zeros_like(grid), emb, "squared")
    expected, _ = getattr(drift_matrix_reference, method)(emb.A, grid)

    actual = prop.function()

    assert_allclose(actual, expected, rtol=RTOL)


@pytest.mark.parametrize("method", ["function", "grad_wrt_A"])
def test_spectrum_rejects_singular_A_at_zero_frequency(method):
    # δ = γ and ζ = 0 give Ω = 0 and A = [[2, 0], [0, 0]], so
    # Λ(0) = θᵀA⁻¹θ does not exist. The default mappers reject Ω = 0.
    emb = TwoAuxEmbedder([1.0, 0.8], 1.0, 1.0, 0.0, mappers=[
        IdentityMapper(), LowerBoundMapper(), IdentityMapper(),
        IdentityMapper()])
    prop = MemorySpectrum(FREQ, np.zeros_like(FREQ), emb, "squared")

    with pytest.raises(ValueError, match="singular"):
        getattr(prop, method)()
