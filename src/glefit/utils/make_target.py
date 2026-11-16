#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   make_target.py
@Time    :   2026/11/16 15:00:00
@Author  :   George Trenins
@Desc    :   Write the memory kernel or spectrum of a model spectral
             density to a CSV file that a fit config reads as its target.
'''


from __future__ import print_function, division, absolute_import
import argparse
import json
import numbers
from pathlib import Path
import numpy as np
import numpy.typing as npt
from glefit.config.config_handler import ConfigError, ConfigHandler
from glefit.spectral import SPECTRAL_MAP
from glefit.spectral._base import BaseSpectralDensity

KEYS = {"model", "property", "nm_omega", "grid", "output"}
# property: (method, grid)
PROPERTIES = {
    "kernel": ("nmK", "time"),
    "spectrum": ("nmLambda", "frequency"),
}


def read_model(section) -> BaseSpectralDensity:
    """The model that the `model` section of the config defines. Its
    constructor receives `parameters` as keyword arguments and checks them.

    Raises:
        ConfigError: if the section has no known `type`, or `parameters`
            is not a dictionary
    """
    if not isinstance(section, dict):
        raise ConfigError(f"model must be a dictionary, got {section!r}")
    name = section.get("type")
    if not isinstance(name, str) or name not in SPECTRAL_MAP:
        raise ConfigError(
            f"model.type {name!r} is unknown; "
            f"the known types are {sorted(SPECTRAL_MAP)}")
    parameters = section.get("parameters")
    if not isinstance(parameters, dict):
        raise ConfigError(
            f"model.parameters must be a dictionary, got {parameters!r}")
    return SPECTRAL_MAP[name](**parameters)


def read_grid(section, config_dir: Path) -> npt.NDArray[np.floating]:
    """The grid that the `grid` section of the config defines, either
    `linspace: [start, stop, num]` or the `path` of a one-column CSV file,
    relative to config_dir, in which `#` starts a comment.

    Raises:
        ConfigError: unless the section has exactly one of these keys, with
            a valid value
    """
    if not isinstance(section, dict) or \
            list(section) not in (["linspace"], ["path"]):
        raise ConfigError(
            "grid must have exactly one key, linspace or path, "
            f"got {section!r}")
    if "linspace" in section:
        values = section["linspace"]
        if (not isinstance(values, list) or len(values) != 3
                or not all(isinstance(v, numbers.Real)
                           and not isinstance(v, bool) for v in values)
                or not isinstance(values[2], int) or values[2] < 1):
            raise ConfigError(
                "grid.linspace must be [start, stop, num], with num a "
                f"positive integer, got {values!r}")
        return np.linspace(*values)
    if not isinstance(section["path"], str):
        raise ConfigError(
            f"grid.path must be a string, got {section['path']!r}")
    # an absolute path replaces config_dir
    path = config_dir / section["path"]
    try:
        grid = np.loadtxt(path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as e:
        raise ConfigError(f"grid.path: cannot read {path}: {e}") from None
    if grid.shape[1] != 1:
        raise ConfigError(
            f"grid.path: {path} must have one column, "
            f"it has {grid.shape[1]}")
    return grid[:, 0]


def main(args: argparse.Namespace) -> None:
    """Write the target that the YAML or JSON file args.config defines:

        model:                         # required
          type: debye                  # a key of SPECTRAL_MAP
          parameters: {eta: 0.7, omega_cut: 2.5}  # to the constructor
        property: spectrum             # required: kernel or spectrum
        nm_omega: 3.0                  # optional, default 0
        grid:                          # required, one of
          linspace: [0.0, 40.0, 4001]  #   passed to np.linspace
          path: grid.csv               #   one column, relative to the config
        output: target.csv             # optional, default target.csv,
                                       # relative to the working directory

    The kernel is Kₙ(t) and the spectrum Λₙ(ω), for the normal-mode
    frequency ωₙ = nm_omega, without rescaling. The output has the columns
    grid and target, separated by commas, with 17 significant digits, so a
    fit config reads it with `columns: [0, 1]`. A `#` header records the
    inputs.

    Raises:
        ConfigError: if the file has an unknown key, or a section is
            missing or malformed
    """
    handler = ConfigHandler(args.config)
    config = handler.config
    unknown = set(config) - KEYS
    if unknown:
        raise ConfigError(
            f"unknown keys {sorted(unknown)}; "
            f"the known keys are {sorted(KEYS)}")
    model = read_model(config.get("model"))
    prop = config.get("property")
    if not isinstance(prop, str) or prop not in PROPERTIES:
        raise ConfigError(
            f"property {prop!r} is unknown; "
            f"the known properties are {sorted(PROPERTIES)}")
    method, grid_name = PROPERTIES[prop]
    grid = read_grid(config.get("grid"), handler.config_path.parent)
    nm_omega = config.get("nm_omega", 0.0)

    target = getattr(model, method)(nm_omega, grid)

    section = config["model"]
    header = "\n".join([
        f"model: {section['type']} {json.dumps(section['parameters'])}",
        f"property: {prop}",
        f"nm_omega: {nm_omega}",
        f"columns: {grid_name}, {prop}",
    ])
    np.savetxt(config.get("output", "target.csv"),
               np.column_stack([grid, target]),
               fmt="%.17g", delimiter=",", header=header)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write the target data of a fit from a model.")
    parser.add_argument(
        'config', type=str, help="Path to config file (YAML/JSON)")
    main(parser.parse_args())
