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


class CorruptedObjective(QuadraticObjective):
    """The quadratic objective with GRAD and HESS, whose distance,
    gradient or Hessian (`which`) has `bad` in its last entry from the
    n-th call of all() on, counting from 0.
    """

    def __init__(self, emb, which, bad, n):
        super().__init__(emb, GRAD, HESS)
        self.which = which
        self.bad = bad
        self.n = n
        self.calls = 0

    def all(self):
        f, g, h = super().all()
        out = {"distance": np.array(f), "gradient": g.copy(),
               "Hessian": h.copy()}
        if self.calls >= self.n:
            out[self.which].flat[-1] = self.bad
        self.calls += 1
        return float(out["distance"]), out["gradient"], out["Hessian"]


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


@pytest.mark.parametrize("cls", [NewtonRaphson, EigenvectorFollowing],
                         ids=["NR", "EF"])
def test_constructor_rejects_unknown_arguments(quadratic, cls):
    emb, merit = quadratic

    with pytest.raises(TypeError, match="trajfle"):
        cls(emb, merit, trajfile=io.StringIO(), trajfle="traj.out")


@pytest.mark.parametrize("cls", [NewtonRaphson, EigenvectorFollowing],
                         ids=["NR", "EF"])
def test_run_rejects_unknown_options(quadratic, cls):
    emb, merit = quadratic
    opt = cls(emb, merit, trajfile=io.StringIO())

    with pytest.raises(TypeError, match="gtoll"):
        opt.run(1, {"gtol": 1e-3, "gtoll": 1e-6})


def quadratic_with_max_grad(max_grad):
    """A quadratic objective whose gradient at the start has max|g| equal
    to max_grad.
    """
    emb = PronyCosineEmbedder(1.0, 1.0, 1.0)
    grad = np.array([0.0, -max_grad, 0.0])
    return emb, QuadraticObjective(emb, grad, HESS)


# the default gtol is 0.05
NEAR_DEFAULT_GTOL = pytest.mark.parametrize(
    "max_grad, expected", [(0.049, True), (0.051, False)],
    ids=["below-default-gtol", "above-default-gtol"])


@NEAR_DEFAULT_GTOL
@pytest.mark.parametrize("cls", [NewtonRaphson, EigenvectorFollowing],
                         ids=["NR", "EF"])
def test_converged_before_run_uses_default_gtol(cls, max_grad, expected):
    emb, merit = quadratic_with_max_grad(max_grad)
    opt = cls(emb, merit, trajfile=io.StringIO())

    converged = opt.converged()

    assert converged == expected


@NEAR_DEFAULT_GTOL
def test_run_without_gtol_resets_it_to_default(max_grad, expected):
    emb, merit = quadratic_with_max_grad(max_grad)
    opt = NewtonRaphson(emb, merit, trajfile=io.StringIO())
    opt.run(0, {"gtol": 1.0})

    converged = opt.run(0, {})

    assert converged == expected


@pytest.mark.parametrize("cls, expected", [
    (NewtonRaphson, NR_STEP),
    (EigenvectorFollowing, EF_STEP),
], ids=["NR", "EF"])
def test_step_before_run_has_no_max_step(quadratic, cls, expected):
    emb, merit = quadratic
    opt = cls(emb, merit, trajfile=io.StringIO())

    step = opt.step()

    assert_allclose(step, expected, rtol=RTOL, atol=0)


# The first two parameters have curvatures -2 and 1/2 along U and W, and
# gradient components -3/4 and 1/3 along them, so EF, which uses |b|, steps
# by 1/3 and -1/2 along them, which is (3/5, -1/30). The last parameter
# has its own curvature c and gradient g. With max|b| = 2 and p = 3, EF
# takes no step along it where c <= TAU·2 = 2e-10 and |g| < gtol/√3, which
# is 0.0289 for gtol = 0.05, and -2g/(c + √(c² + 4g²)) ≈ -sign(g)
# otherwise.
U = np.array([0.6, 0.8, 0.0])
W = np.array([-0.8, 0.6, 0.0])


@pytest.mark.parametrize("curvature, grad, gtol, expected", [
    (0.0, 1e-3, 0.05, 0.0),
    (1e-200, 1e-3, 0.05, 0.0),
    (1.9e-10, 1e-3, 0.05, 0.0),
    (2.1e-10, 1e-3, 0.05, -1.0),
    (0.0, 0.0285, 0.05, 0.0),
    (0.0, 0.0292, 0.05, -1.0),
    (0.0, 1e-3, 1e-3, -1.0),
], ids=["zero", "tiny", "below-tau", "above-tau", "gradient-below-bound",
        "gradient-above-bound", "bound-from-gtol"])
def test_ef_cuts_flat_modes_with_small_gradients(
        curvature, grad, gtol, expected):
    emb = PronyCosineEmbedder(1.0, 1.0, 1.0)
    hess = (-2 * np.outer(U, U) + 0.5 * np.outer(W, W)
            + np.diag([0.0, 0.0, curvature]))
    merit = QuadraticObjective(emb, -0.75 * U + W / 3 + [0, 0, grad], hess)
    opt = EigenvectorFollowing(emb, merit, trajfile=io.StringIO())
    opt.run(0, {"gtol": gtol})

    step = opt.step()

    # above TAU, the step differs from -1 by c/(2g) = 1.05e-7
    assert_allclose(step, [0.6, -1 / 30, expected], rtol=1e-6, atol=0)


def test_ef_with_a_dead_term_converges_under_max_step():
    # θ₂² underflows to 0, so the second term has no effect, and its
    # modes have no curvature. EF stepped along them at random, and
    # max_step scaled down every step to fit: the fit took 89 steps.
    grid = np.linspace(0.0, 20.0, 41)
    target = PronyEmbedder(1.0, 1.0).spectrum(grid)
    emb = MultiEmbedder([PronyEmbedder(0.7, 0.6), PronyEmbedder(1e-200, 3.0)])
    dead = emb.x[2:]
    merit = MemorySpectrum(grid, target, emb, "squared")
    opt = EigenvectorFollowing(emb, merit, trajfile=io.StringIO())

    converged = opt.run(10, {"gtol": 1e-10, "max_step": 0.2})

    assert converged
    assert_allclose(emb.x[2:], dead, rtol=0, atol=1e-12)


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


def test_log_is_overwritten_by_a_new_optimizer(quadratic, tmp_path):
    emb, merit = quadratic
    path = tmp_path / "log.out"
    with NewtonRaphson(emb, merit, logfile=path,
                       trajfile=io.StringIO()) as opt:
        opt.run(0, {"gtol": 1e-12})

    with NewtonRaphson(emb, merit, logfile=path,
                       trajfile=io.StringIO()) as opt:
        opt.run(0, {"gtol": 1e-12})

    # only the entry for the start of the second run
    assert len(logged_eigenvalues(path.read_text())) == 1


WHICH = pytest.mark.parametrize("which", ["distance", "gradient", "Hessian"])
BAD = pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf],
                              ids=["nan", "inf", "-inf"])
OPTIMIZERS = pytest.mark.parametrize(
    "cls", [NewtonRaphson, EigenvectorFollowing], ids=["NR", "EF"])


@WHICH
@BAD
@OPTIMIZERS
def test_constructor_rejects_non_finite_merit(cls, which, bad):
    emb = PronyCosineEmbedder(1.0, 1.0, 1.0)
    merit = CorruptedObjective(emb, which, bad, n=0)

    with pytest.raises(ValueError, match=f"non-finite {which} at step 0"):
        cls(emb, merit, trajfile=io.StringIO())


@WHICH
@BAD
@OPTIMIZERS
def test_run_stops_at_the_first_non_finite_merit(cls, which, bad):
    emb = PronyCosineEmbedder(1.0, 1.0, 1.0)
    merit = CorruptedObjective(emb, which, bad, n=1)
    opt = cls(emb, merit, trajfile=io.StringIO())

    with pytest.raises(ValueError, match=f"non-finite {which} at step 1"):
        opt.run(5, {"gtol": 1e-12})
