#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_make_target.py
@Time    :   2026/11/16 15:00:00
@Author  :   George Trenins
@Desc    :   Test the script that writes the target data of a fit
'''


from __future__ import print_function, division, absolute_import
import argparse
import numpy as np
import pytest
import yaml
from numpy.testing import assert_array_equal
from glefit.config.config_handler import ConfigError
from glefit.config.data_io import load_data
from glefit.spectral import Debye, ExpOhmic
from glefit.utils.make_target import main

ETA = 0.7
OMEGA_CUT = 2.5
NM_OMEGA = 3.0
# not in signature order, so that passing them by position fails
PARAMETERS = {"omega_cut": OMEGA_CUT, "eta": ETA}
LINSPACE = [0.0, 6.0, 13]
GRID = np.linspace(0.0, 6.0, 13)
MODELS = {"debye": Debye, "expohmic": ExpOhmic}


def make_config(**changes):
    """The spectrum of a Debye model on LINSPACE, with keys changed."""
    config = {
        "model": {"type": "debye", "parameters": PARAMETERS},
        "property": "spectrum",
        "grid": {"linspace": LINSPACE},
    }
    config.update(changes)
    return config


def read_target(path):
    """The grid and target that a fit config reads from path."""
    return load_data(
        {"source": "external", "path": str(path), "columns": [0, 1]},
        "target")


@pytest.fixture
def run_main(tmp_path, monkeypatch):
    """Run main on a config in tmp_path/inputs, with tmp_path/run as the
    working directory, and return the working directory."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)

    def run(config):
        path = inputs / "target.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        main(argparse.Namespace(config=str(path)))
        return run_dir

    return run


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "prop, method", [("kernel", "nmK"), ("spectrum", "nmLambda")],
    ids=["kernel", "spectrum"])
@pytest.mark.parametrize(
    "nm_config, nm_omega", [({}, 0.0), ({"nm_omega": NM_OMEGA}, NM_OMEGA)],
    ids=["default-nm-omega", "nm-omega-3"])
def test_target_is_the_transformed_property(
        run_main, model, prop, method, nm_config, nm_omega):
    config = make_config(
        model={"type": model, "parameters": PARAMETERS}, property=prop,
        **nm_config)
    bath = MODELS[model](eta=ETA, omega_cut=OMEGA_CUT)
    expected = getattr(bath, method)(nm_omega, GRID)

    run_dir = run_main(config)

    assert_array_equal(read_target(run_dir / "target.csv")[1], expected)


def test_linspace_grid_is_written_in_full_precision(run_main):
    # two of these points need all 17 significant digits
    config = make_config(grid={"linspace": [0.0, 2.0, 7]})

    run_dir = run_main(config)

    assert_array_equal(read_target(run_dir / "target.csv")[0],
                       np.linspace(0.0, 2.0, 7))


def test_grid_file_is_read_relative_to_the_config(run_main, tmp_path):
    # the kernel, unlike the spectrum, fails for a grid of shape (n, 1)
    (tmp_path / "inputs" / "grid.csv").write_text("# time\n0.1\n0.25\n3.5\n")
    config = make_config(property="kernel", grid={"path": "grid.csv"})

    run_dir = run_main(config)

    assert_array_equal(read_target(run_dir / "target.csv")[0],
                       [0.1, 0.25, 3.5])


def test_two_column_grid_file_raises_config_error(run_main, tmp_path):
    (tmp_path / "inputs" / "grid.csv").write_text("0.1,1.0\n0.25,2.0\n")

    with pytest.raises(ConfigError, match="must have one column"):
        run_main(make_config(grid={"path": "grid.csv"}))


@pytest.mark.parametrize(
    "output, name",
    [({}, "target.csv"), ({"output": "spectrum.csv"}, "spectrum.csv")],
    ids=["default", "given"])
def test_output_is_written_to_the_working_directory(run_main, output, name):
    run_dir = run_main(make_config(**output))

    assert (run_dir / name).is_file()


def test_header_records_the_inputs(run_main):
    config = make_config(property="kernel", nm_omega=NM_OMEGA)

    run_dir = run_main(config)

    assert (run_dir / "target.csv").read_text().splitlines()[:4] == [
        '# model: debye {"omega_cut": 2.5, "eta": 0.7}',
        "# property: kernel",
        "# nm_omega: 3.0",
        "# columns: time, kernel",
    ]


BAD_CONFIGS = {
    "unknown-model": (
        {"model": {"type": "lorentz", "parameters": PARAMETERS}},
        "model.type 'lorentz' is unknown"),
    "no-parameters": (
        {"model": {"type": "debye"}},
        "model.parameters must be a dictionary"),
    "unknown-property": (
        {"property": "density"}, "property 'density' is unknown"),
    "two-grids": (
        {"grid": {"linspace": LINSPACE, "path": "grid.csv"}},
        "grid must have exactly one key"),
    "no-grid-key": ({"grid": {}}, "grid must have exactly one key"),
    "short-linspace": (
        {"grid": {"linspace": [0.0, 6.0]}}, "grid.linspace must be"),
    "float-num": (
        {"grid": {"linspace": [0.0, 6.0, 13.0]}}, "grid.linspace must be"),
    "misspelled-key": ({"nm_omgea": NM_OMEGA}, r"unknown keys \['nm_omgea'\]"),
}


@pytest.mark.parametrize(
    "changes, message", BAD_CONFIGS.values(), ids=BAD_CONFIGS.keys())
def test_malformed_config_raises_config_error(run_main, changes, message):
    with pytest.raises(ConfigError, match=message):
        run_main(make_config(**changes))


def test_extra_model_parameter_raises_type_error(run_main):
    config = make_config(
        model={"type": "debye", "parameters": {**PARAMETERS, "gamma": 1.0}})

    with pytest.raises(TypeError, match="gamma"):
        run_main(config)


def test_invalid_eta_raises_value_error(run_main):
    config = make_config(
        model={"type": "debye",
               "parameters": {"eta": -1.0, "omega_cut": OMEGA_CUT}})

    with pytest.raises(ValueError, match="eta must be non-negative"):
        run_main(config)
