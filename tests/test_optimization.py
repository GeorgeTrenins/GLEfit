#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_optimization.py
@Time    :   2026/11/10 12:00:00
@Author  :   George Trenins
@Desc    :   Test the optimizers
'''


from __future__ import print_function, division, absolute_import
import io
import numpy as np
import pytest
from numpy.testing import assert_allclose
from glefit.analysis.plot_objective import parse_trajectory_file
from glefit.embedding import (
    MultiEmbedder, PronyCosineEmbedder, PronyEmbedder)
from glefit.merit import MemorySpectrum
from glefit.optimization import EigenvectorFollowing, NewtonRaphson


RTOL = 1.0e-12
# Q is orthogonal but not symmetric, so a step that projects onto the
# eigenvectors of H the wrong way round differs from the right one
Q = np.array([[1, -4, 8], [8, 4, 1], [-4, 7, 4]]) / 9
# H has eigenvalues 2, -1 and 1/2 along the columns of Q, and g has the
# components -3/4, -2/3 and 1/3 along them. Along each column, NR steps by
# -gₖ/bₖ = (3/8, -2/3, -2/3), uphill along the negative mode, and EF by
# -2gₖ/(|bₖ| + √(bₖ² + 4gₖ²)) = (1/3, 1/2, -1/2), downhill along all.
# The largest component of each step is negative, which tests the
# magnitude in max_step.
HESS = Q @ np.diag([2.0, -1.0, 0.5]) @ Q.T
GRAD = Q @ np.array([-0.75, -2 / 3, 1 / 3])
NR_STEP = [-55 / 216, -1 / 27, -53 / 54]
EF_STEP = [-17 / 27, 25 / 54, 1 / 54]


class QuadraticObjective:
    """f = gᵀd + dᵀHd/2, with d = emb.x - x0, so that the optimizer sees
    the gradient g and the Hessian H at the start. Optimizers only call
    all().
    """

    def __init__(self, emb, grad, hess):
        self.emb = emb
        self.x0 = emb.x
        self.grad = grad
        self.hess = hess

    def all(self):
        d = self.emb.x - self.x0
        return (self.grad @ d + d @ self.hess @ d / 2,
                self.grad + self.hess @ d,
                self.hess)


def logged_eigenvalues(log):
    """The Hessian eigenvalues in each entry of an optimizer's log."""
    blocks = log.split("=== Hessian eigenvalues ===\n")[1:]
    return [np.array(b.split("]")[0].strip(" [\n").split(","), dtype=float)
            for b in blocks]


@pytest.fixture
def quadratic():
    # three log-mapped parameters, all 1, so the start is x0 = 0
    emb = PronyCosineEmbedder(1.0, 1.0, 1.0)
    return emb, QuadraticObjective(emb, GRAD, HESS)


@pytest.fixture
def prony_fit():
    """A fit of two Prony terms to the spectrum of θ = (1, 0.7) and
    γ = (1, 3), started at θ = (1.3, 0.4) and γ = (0.6, 4).
    """
    grid = np.linspace(0.0, 10.0, 26)
    target = MultiEmbedder(
        [PronyEmbedder(1.0, 1.0), PronyEmbedder(0.7, 3.0)]).spectrum(grid)
    emb = MultiEmbedder([PronyEmbedder(1.3, 0.6), PronyEmbedder(0.4, 4.0)])
    return emb, MemorySpectrum(grid, target, emb, "squared")


@pytest.mark.parametrize("cls, expected", [
    (NewtonRaphson, NR_STEP),
    (EigenvectorFollowing, EF_STEP),
], ids=["NR", "EF"])
def test_first_step_matches_hand_computed_value(quadratic, cls, expected):
    emb, merit = quadratic
    opt = cls(emb, merit, trajfile=io.StringIO())

    opt.run(1, {"gtol": 1e-12})

    assert_allclose(emb.x, expected, rtol=RTOL, atol=0)


@pytest.mark.parametrize("cls, max_step, expected", [
    (NewtonRaphson, 0.1, [-11 / 424, -1 / 265, -0.1]),
    (EigenvectorFollowing, 0.1, [-0.1, 5 / 68, 1 / 340]),
    (EigenvectorFollowing, 1.0, EF_STEP),
], ids=["NR-scaled", "EF-scaled", "EF-within-max-step"])
def test_max_step_scales_the_whole_step(
        quadratic, cls, max_step, expected):
    emb, merit = quadratic
    opt = cls(emb, merit, trajfile=io.StringIO())

    opt.run(1, {"gtol": 1e-12, "max_step": max_step})

    assert_allclose(emb.x, expected, rtol=RTOL, atol=0)


# max|g| = 25/27 ≈ 0.926, while the 2-norm of g is about 1.057
@pytest.mark.parametrize("gtol, expected", [(1.0, True), (25 / 27, False)],
                         ids=["above-max-norm", "equal-to-max-norm"])
def test_converged_compares_max_gradient_with_gtol(
        quadratic, gtol, expected):
    emb, merit = quadratic
    opt = NewtonRaphson(emb, merit, trajfile=io.StringIO())
    opt.run(0, {"gtol": gtol})

    converged = opt.converged()

    assert converged == expected


def test_run_takes_no_step_if_converged_at_start(quadratic):
    emb, merit = quadratic
    opt = EigenvectorFollowing(emb, merit, trajfile=io.StringIO())

    opt.run(5, {"gtol": 1.0})

    assert opt.nsteps == 0
    assert_allclose(emb.x, [0.0, 0.0, 0.0], rtol=0, atol=0)


# NR reaches the stationary point of the quadratic in one step, while EF
# goes downhill along the negative mode and never converges
@pytest.mark.parametrize("cls, gtol, steps, expected", [
    (NewtonRaphson, 1.0, 5, True),
    (NewtonRaphson, 1e-12, 1, True),
    (NewtonRaphson, 1e-12, 0, False),
    (EigenvectorFollowing, 1.0, 5, True),
    (EigenvectorFollowing, 1e-12, 1, False),
], ids=["NR-at-start", "NR-after-a-step", "NR-out-of-steps",
        "EF-at-start", "EF-out-of-steps"])
def test_run_returns_whether_the_fit_converged(
        quadratic, cls, gtol, steps, expected):
    emb, merit = quadratic
    opt = cls(emb, merit, trajfile=io.StringIO())

    converged = opt.run(steps, {"gtol": gtol})

    assert converged == expected


def test_ef_fit_recovers_its_target_parameters(prony_fit):
    emb, merit = prony_fit
    opt = EigenvectorFollowing(emb, merit, trajfile=io.StringIO())

    opt.run(50, {"gtol": 1e-9, "max_step": 0.5})

    assert opt.converged()
    assert_allclose(emb.conventional_params, [1.0, 1.0, 0.7, 3.0],
                    rtol=1e-6)


def test_trajectory_holds_parameters_at_each_step(prony_fit, tmp_path):
    emb, merit = prony_fit
    path = tmp_path / "traj.out"

    with EigenvectorFollowing(emb, merit, trajfile=path) as opt:
        opt.run(3, {"gtol": 1e-9})
    steps, params = parse_trajectory_file(path)

    # the file keeps 7 significant digits
    assert steps == [0, 1, 2, 3]
    assert_allclose(params[0], [1.3, 0.6, 0.4, 4.0], rtol=1e-6)
    assert_allclose(params[-1], emb.conventional_params, rtol=1e-6)


@pytest.mark.parametrize("cls", [NewtonRaphson, EigenvectorFollowing],
                         ids=["NR", "EF"])
def test_log_holds_current_hessian_eigenvalues(prony_fit, cls):
    emb, merit = prony_fit
    log = io.StringIO()
    opt = cls(emb, merit, logfile=log, trajfile=io.StringIO())

    opt.run(1, {"gtol": 1e-9})

    # the Hessian after the step; the log keeps 7 significant digits
    _, _, hess = merit.all()
    assert_allclose(logged_eigenvalues(log.getvalue())[-1],
                    np.linalg.eigvalsh(hess), rtol=1e-6)
