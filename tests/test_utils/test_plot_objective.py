#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_plot_objective.py
@Time    :   2026/11/16 18:00:00
@Author  :   George Trenins
@Desc    :   Test the script that plots each step of a fit
'''


from __future__ import print_function, division, absolute_import
import argparse
import json
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from glefit.embedding import EMBEDDER_MAP
from glefit.optimize import main as optimize
from glefit.utils.plot_objective import (
    main, parse_trajectory_file, plot_trajectory, setup_from_config)

# a prony and a pronycos term, fitted from START to TARGET_PARAMS, the
# conventional parameters of their sum
START = {"embedders": [
    {"type": "prony", "parameters": {"theta": 0.8, "gamma": 1.2}},
    {"type": "pronycos",
     "parameters": {"theta": 0.5, "gamma": 0.4, "omega": 1.5}}]}
TARGET_PARAMS = [1.0, 0.8, 0.7, 0.5, 2.0]
PROPERTIES = ["kernel", "spectrum"]
GRIDS = {"kernel": np.linspace(0.0, 6.0, 13),
         "spectrum": np.linspace(0.0, 5.0, 11)}
# the fit cannot converge in MAX_ITER steps
MAX_ITER = 2
GTOL = 1e-12
# the drift-matrix and closed forms agree to rounding
RTOL, ATOL = 1e-10, 1e-12
TITLE = re.compile(r"Step (\d+) / (\d+) \| Merit = (\S+)$")


def closed_forms(prop, params):
    """K or Λ on GRIDS[prop] of each term of START, at the conventional
    parameters params of their sum."""
    terms = [EMBEDDER_MAP[term["type"]].from_dict(term["parameters"])
             for term in START["embedders"]]
    ends = np.cumsum([term.nparam for term in terms])
    for term, p in zip(terms, np.split(np.asarray(params), ends[:-1])):
        term.conventional_params = p
    return [getattr(term, prop)(GRIDS[prop]) for term in terms]


def write_config(path, prop, trajfile="traj.out"):
    """Write the config of a fit of K or Λ from START to TARGET_PARAMS."""
    grid = GRIDS[prop]
    target = sum(closed_forms(prop, TARGET_PARAMS))
    config = {
        "data": {prop: {"source": "inline", "grid": grid.tolist(),
                        "target": target.tolist()}},
        "merit_function": {
            "type": prop,
            "parameters": {"data": prop, "metric": "squared"}},
        "embedder": {"type": "multi", "parameters": START},
        "optimization": {
            "type": "EF",
            "parameters": {"trajfile": trajfile},
            "options": {"max_iter": MAX_ITER, "gtol": GTOL}}}
    path.write_text(json.dumps(config))


@pytest.fixture(autouse=True)
def no_window(monkeypatch):
    """Let plt.show return at once, and close the figures afterwards."""
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


@pytest.fixture
def fit(tmp_path, monkeypatch):
    """A function that runs a fit of K or Λ in tmp_path, the working
    directory, and returns the path of its config."""
    monkeypatch.chdir(tmp_path)

    def run(prop, trajfile="traj.out"):
        path = tmp_path / "config.json"
        write_config(path, prop, trajfile)
        optimize(argparse.Namespace(config=str(path)))
        return path
    return run


def plot(config):
    """Plot traj.out of the fit with this config. Return the merit
    function, the parameters at each step and the axes of the curves."""
    embedder, merit = setup_from_config(config)
    steps, params = parse_trajectory_file("traj.out")
    plot_trajectory(merit, embedder, steps, params)
    return merit, params, plt.gcf().axes[0]


def curves(ax):
    """The points (x, y) of each line in ax, by label, shape (n, 2)."""
    return {line.get_label(): line.get_xydata() for line in ax.get_lines()}


def test_setup_multi_accepts_out_of_bounds_values(tmp_path):
    write_config(tmp_path / "config.json", "kernel")
    embedder, _ = setup_from_config(tmp_path / "config.json")
    # the default mappers reject negative rates and frequencies
    params = [1.0, -0.8, 0.7, -0.5, -2.0]

    embedder.conventional_params = params

    assert_allclose(embedder.conventional_params, params)


@pytest.mark.parametrize("prop", PROPERTIES)
def test_plot_target_is_data_on_grid(fit, prop):
    _, _, ax = plot(fit(prop))

    target = sum(closed_forms(prop, TARGET_PARAMS))

    assert_allclose(curves(ax)["Target"],
                    np.column_stack([GRIDS[prop], target]))


@pytest.mark.parametrize("prop", PROPERTIES)
def test_plot_total_is_closed_form_at_step_0(fit, prop):
    _, params, ax = plot(fit(prop))

    total = sum(closed_forms(prop, params[0]))

    assert_allclose(curves(ax)["Total"],
                    np.column_stack([GRIDS[prop], total]),
                    rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("prop", PROPERTIES)
def test_plot_components_are_closed_forms(fit, prop):
    _, params, ax = plot(fit(prop))

    lines = curves(ax)
    terms = closed_forms(prop, params[0])

    assert_allclose(lines["Component 1"][:, 1], terms[0],
                    rtol=RTOL, atol=ATOL)
    assert_allclose(lines["Component 2"][:, 1], terms[1],
                    rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("prop", PROPERTIES)
def test_plot_title_shows_steps_and_distance(fit, prop):
    merit, params, ax = plot(fit(prop))

    first, last, distance = TITLE.search(ax.get_title()).groups()
    deviation = sum(closed_forms(prop, params[0])) - merit.target

    assert (int(first), int(last)) == (0, MAX_ITER)
    # the title has 7 significant digits
    assert float(distance) == pytest.approx(
        np.sum(merit.weight * deviation**2), rel=1e-6)


def test_plot_y_range_is_target_with_margin(fit):
    merit, _, ax = plot(fit("spectrum"))

    top = merit.target.max()

    # the spectrum is positive, so the range starts from 0
    assert ax.get_ylim() == pytest.approx((-0.05 * top, 1.05 * top))


@pytest.mark.parametrize("args, trajfile", [
    ([], "traj.out"),
    (["--traj", "other.out"], "other.out"),
], ids=["default", "given"])
def test_main_reads_given_trajectory(fit, monkeypatch, args, trajfile):
    config = fit("kernel", trajfile)
    monkeypatch.setattr(sys, "argv", ["plot_objective.py", str(config)]
                        + args)

    main()

    _, params = parse_trajectory_file(trajfile)
    total = sum(closed_forms("kernel", params[0]))
    assert_allclose(curves(plt.gcf().axes[0])["Total"][:, 1], total,
                    rtol=RTOL, atol=ATOL)


def test_main_rejects_missing_trajectory(tmp_path, monkeypatch):
    write_config(tmp_path / "config.json", "kernel")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["plot_objective.py", str(tmp_path / "config.json")])

    with pytest.raises(FileNotFoundError, match="traj.out"):
        main()
