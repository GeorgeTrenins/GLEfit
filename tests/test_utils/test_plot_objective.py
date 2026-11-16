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
import pickle
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from glefit.embedding import EMBEDDER_MAP
from glefit.merit import PROPERTY_MAP
from glefit.optimize import EMBEDDER_FILE, main as optimize
from glefit.utils.plot_objective import (
    main, parse_log_file, parse_trajectory_file, plot_trajectory,
    setup_from_config)

# a prony and a pronycos term, fitted from START to TARGET_PARAMS, the
# conventional parameters of their sum
START = {"embedders": [
    {"type": "prony", "parameters": {"theta": 0.8, "gamma": 1.2}},
    {"type": "pronycos",
     "parameters": {"theta": 0.5, "gamma": 0.4, "omega": 1.5}}]}
TARGET_PARAMS = [1.0, 0.8, 0.7, 0.5, 2.0]
# embedder sections by type: START, and each type alone
EMBEDDERS = {
    "multi": {"type": "multi", "parameters": START},
    "prony": START["embedders"][0],
    "pronycos": START["embedders"][1],
    "gen2x2": {"type": "gen2x2",
               "parameters": {"theta": [1.0, 0.5], "gamma": 0.9,
                              "delta": 0.9, "zeta": 1.5}},
}
# admissible conventional parameters that the default mappers reject:
# negative rates and frequencies, and for gen2x2, ζ = 0 at δ = γ, so Ω = 0
OUT_OF_BOUNDS = {
    "multi": [1.0, -0.8, 0.7, -0.5, -2.0],
    "prony": [1.0, -0.8],
    "pronycos": [0.7, -0.5, -2.0],
    "gen2x2": [1.0, 0.5, 0.9, 0.9, 0.0],
}
# the terms of each embedder section, for closed_forms
TERMS = {"multi": START["embedders"], "prony": [EMBEDDERS["prony"]],
         "pronycos": [EMBEDDERS["pronycos"]], "gen2x2": [EMBEDDERS["gen2x2"]]}
PROPERTIES = ["kernel", "spectrum"]
GRIDS = {"kernel": np.linspace(0.0, 6.0, 13),
         "spectrum": np.linspace(0.0, 5.0, 11)}
# the fit cannot converge in MAX_ITER steps
MAX_ITER = 2
GTOL = 1e-12
# the drift-matrix and closed forms agree to rounding
RTOL, ATOL = 1e-10, 1e-12
TITLE = re.compile(r"Step (\d+) / (\d+) \| Merit = (\S+)$")
GMAX_LABEL = r"max $|g_i|$"


def closed_forms(prop, params, sections=START["embedders"]):
    """K or Λ on GRIDS[prop] of each embedder in sections, by default the
    terms of START, at the conventional parameters params of their sum."""
    terms = [EMBEDDER_MAP[term["type"]].from_dict(term["parameters"])
             for term in sections]
    ends = np.cumsum([term.nparam for term in terms])
    for term, p in zip(terms, np.split(np.asarray(params), ends[:-1])):
        term.conventional_params = p
    return [getattr(term, prop)(GRIDS[prop]) for term in terms]


def write_config(path, prop, trajfile="traj.out",
                 embedder=EMBEDDERS["multi"]):
    """Write the config of a fit of an embedder section, by default
    START, to K or Λ of START at TARGET_PARAMS, which logs to log.out."""
    grid = GRIDS[prop]
    target = sum(closed_forms(prop, TARGET_PARAMS))
    config = {
        "data": {prop: {"source": "inline", "grid": grid.tolist(),
                        "target": target.tolist()}},
        "merit_function": {
            "type": prop,
            "parameters": {"data": prop, "metric": "squared"}},
        "embedder": embedder,
        "optimization": {
            "type": "EF",
            "parameters": {"trajfile": trajfile, "logfile": "log.out"},
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

    def run(prop, trajfile="traj.out", embedder=EMBEDDERS["multi"]):
        path = tmp_path / "config.json"
        write_config(path, prop, trajfile, embedder)
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


def draw(config, logfile=None):
    """Plot traj.out of the fit with this config, and the log if given.
    Return the merit function, the parameters at each step, the figure
    and the slider."""
    embedder, merit = setup_from_config(config)
    steps, params = parse_trajectory_file("traj.out")
    log = None if logfile is None else parse_log_file(logfile)
    fig, slider = plot_trajectory(merit, embedder, steps, params, log=log)
    return merit, params, fig, slider


def curves(ax):
    """The points (x, y) of each line in ax, by label, shape (n, 2)."""
    return {line.get_label(): line.get_xydata() for line in ax.get_lines()}


def merit_at(prop, parameters):
    """The merit function of a fit of K or Λ for a multi embedder with
    these parameters and the default mappers, as in the fit."""
    embedder = EMBEDDER_MAP["multi"].from_dict(parameters)
    target = sum(closed_forms(prop, TARGET_PARAMS))
    return PROPERTY_MAP[prop](GRIDS[prop], target, embedder, "squared")


def final_parameters():
    """The parameters at the end of the fit, at full precision."""
    with open(EMBEDDER_FILE, "rb") as f:
        return pickle.load(f)["parameters"]


def test_parse_rejects_empty_file(tmp_path):
    # as left by a fit that stops before it writes step 0
    path = tmp_path / "traj.out"
    path.touch()

    with pytest.raises(ValueError, match=re.escape(str(path))):
        parse_trajectory_file(path)


def test_parse_rejects_log_file(fit):
    fit("kernel")

    with pytest.raises(ValueError, match="log.out"):
        parse_trajectory_file("log.out")


@pytest.mark.parametrize("prop", PROPERTIES)
def test_parse_log_reads_distance_and_gradient(fit, prop):
    fit(prop)
    first, last = merit_at(prop, START), merit_at(prop, final_parameters())

    steps, distance, gmax = parse_log_file("log.out")

    assert steps == list(range(MAX_ITER + 1))
    # the log has 8 significant digits
    assert distance[[0, -1]] == pytest.approx(
        [first.distance, last.distance], rel=1e-7)
    assert gmax[[0, -1]] == pytest.approx(
        [np.max(np.abs(first.gradient())), np.max(np.abs(last.gradient()))],
        rel=1e-7)


def test_parse_log_rejects_trajectory(fit):
    fit("kernel")

    with pytest.raises(ValueError, match="traj.out"):
        parse_log_file("traj.out")


@pytest.mark.parametrize("name", OUT_OF_BOUNDS)
def test_setup_accepts_out_of_bounds_values(tmp_path, name):
    write_config(tmp_path / "config.json", "kernel",
                 embedder=EMBEDDERS[name])
    embedder, _ = setup_from_config(tmp_path / "config.json")

    embedder.conventional_params = OUT_OF_BOUNDS[name]

    assert_allclose(embedder.conventional_params, OUT_OF_BOUNDS[name])


@pytest.mark.parametrize("name", ["prony", "pronycos", "gen2x2"])
def test_plot_single_total_is_closed_form(fit, name):
    _, params, ax = plot(fit("kernel", embedder=EMBEDDERS[name]))

    lines = curves(ax)
    total, = closed_forms("kernel", params[0], [EMBEDDERS[name]])

    # no components are drawn
    assert set(lines) == {"Target", "Total"}
    assert_allclose(lines["Total"][:, 1], total, rtol=RTOL, atol=ATOL)


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


def test_plot_log_panel_shows_log(fit):
    _, _, fig, _ = draw(fit("kernel"), "log.out")

    steps, distance, gmax = parse_log_file("log.out")
    lines = curves(fig.axes[1])

    assert fig.axes[1].get_yscale() == "log"
    assert_allclose(lines["Distance"], np.column_stack([steps, distance]))
    assert_allclose(lines[GMAX_LABEL], np.column_stack([steps, gmax]))


def test_plot_rejects_log_of_other_run(fit):
    embedder, merit = setup_from_config(fit("kernel"))
    steps, params = parse_trajectory_file("traj.out")
    # as from a fit that stopped a step earlier
    log = [values[:-1] for values in parse_log_file("log.out")]

    with pytest.raises(ValueError, match="steps"):
        plot_trajectory(merit, embedder, steps, params, log=log)


@pytest.mark.parametrize("name", TERMS)
def test_slider_draws_total_at_selected_step(fit, name):
    merit, params, fig, slider = draw(
        fit("kernel", embedder=EMBEDDERS[name]))

    slider.set_val(MAX_ITER)

    total = sum(closed_forms("kernel", params[MAX_ITER], TERMS[name]))
    step, _, distance = TITLE.search(fig.axes[0].get_title()).groups()
    assert_allclose(curves(fig.axes[0])["Total"][:, 1], total,
                    rtol=RTOL, atol=ATOL)
    assert int(step) == MAX_ITER
    assert float(distance) == pytest.approx(
        np.sum(merit.weight * (total - merit.target)**2), rel=1e-6)


def test_slider_draws_components_at_selected_step(fit):
    _, params, fig, slider = draw(fit("kernel"))

    slider.set_val(MAX_ITER)

    lines = curves(fig.axes[0])
    terms = closed_forms("kernel", params[MAX_ITER])
    assert_allclose(lines["Component 1"][:, 1], terms[0],
                    rtol=RTOL, atol=ATOL)
    assert_allclose(lines["Component 2"][:, 1], terms[1],
                    rtol=RTOL, atol=ATOL)


def test_slider_moves_step_marker(fit):
    _, _, fig, slider = draw(fit("kernel"), "log.out")

    slider.set_val(MAX_ITER)

    assert_allclose(curves(fig.axes[1])["Selected step"][:, 0],
                    [MAX_ITER, MAX_ITER])


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


@pytest.mark.parametrize("args, naxes", [
    ([], 2),
    (["--log", "log.out"], 3),
], ids=["without", "with"])
def test_main_draws_log_panel_if_given(fit, monkeypatch, args, naxes):
    config = fit("kernel")
    monkeypatch.setattr(sys, "argv", ["plot_objective.py", str(config)]
                        + args)

    main()

    # the curves, the log panel if any, and the slider
    assert len(plt.gcf().axes) == naxes


def test_main_rejects_missing_trajectory(tmp_path, monkeypatch):
    write_config(tmp_path / "config.json", "kernel")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["plot_objective.py", str(tmp_path / "config.json")])

    with pytest.raises(FileNotFoundError, match="traj.out"):
        main()
