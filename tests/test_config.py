#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_config.py
@Time    :   2026/11/10 18:00:00
@Author  :   George Trenins
@Desc    :   Test reading, checking and loading a configuration
'''


from __future__ import print_function, division, absolute_import
import copy
import json
import re
import pytest
import yaml
from numpy.testing import assert_array_equal
from glefit.config.config_handler import ConfigError, ConfigHandler
from glefit.config.data_io import DataLoadError, load_data


CONFIG = {
    "data": {"kernel": {"source": "inline", "grid": [0.0, 1.0, 2.0],
                        "target": [1.0, 0.5, 0.25]}},
    "merit_function": {"type": "kernel",
                       "parameters": {"data": "kernel", "metric": "squared"}},
    "embedder": {"type": "prony", "parameters": {"theta": 1.0, "gamma": 0.7}},
    "optimization": {"type": "EF", "parameters": {"trajfile": "traj.out"},
                     "options": {"max_iter": 5, "gtol": 1e-3}},
}


def without(path):
    """A handler for CONFIG without the key at path."""
    config = copy.deepcopy(CONFIG)
    *parents, key = path
    section = config
    for parent in parents:
        section = section[parent]
    del section[key]
    return ConfigHandler.from_dict(config)


def with_options(options):
    """A handler for CONFIG with the given optimization options."""
    config = copy.deepcopy(CONFIG)
    config["optimization"]["options"] = options
    return ConfigHandler.from_dict(config)


def replaced(section, value):
    """A copy of CONFIG with the given value for section."""
    config = copy.deepcopy(CONFIG)
    config[section] = value
    return config


# a CSV file, and the grid and target that columns [0, 1] hold
CSV_TEXT = "0.0,1.0\n0.5,0.25\n"
GRID = [0.0, 0.5]
TARGET = [1.0, 0.25]


def csv_dataset(file, text, **entries):
    """An external dataset with the given entries, after writing text to
    file."""
    file.write_text(text)
    return {"source": "external", "path": str(file), "columns": [0, 1],
            **entries}


def test_validate_accepts_a_complete_config():
    handler = ConfigHandler.from_dict(CONFIG)

    handler.validate()


# every key that optimize.py reads, and the text that the error names it by
REQUIRED = [
    (("data",), "data"),
    (("merit_function",), "merit_function"),
    (("merit_function", "type"), "merit_function.type"),
    (("merit_function", "parameters"), "merit_function.parameters"),
    (("embedder",), "embedder"),
    (("embedder", "type"), "embedder.type"),
    (("embedder", "parameters"), "embedder.parameters"),
    (("optimization",), "optimization"),
    # the message names optimization.optimizer, an old name of the key
    (("optimization", "type"), "optimization"),
    (("optimization", "parameters"), "optimization.parameters"),
    (("optimization", "options"), "optimization.options"),
    (("optimization", "options", "max_iter"),
     "optimization.options.max_iter"),
]


@pytest.mark.parametrize("path, name", REQUIRED,
                         ids=[".".join(path) for path, _ in REQUIRED])
def test_validate_requires_key(path, name):
    handler = without(path)

    with pytest.raises(ConfigError, match=re.escape(name)):
        handler.validate()


def test_validate_requires_options_to_be_a_dictionary():
    handler = with_options(None)

    with pytest.raises(ConfigError, match="optimization.options"):
        handler.validate()


@pytest.mark.parametrize("max_iter", ["5", 2.5, -1, True, None],
                         ids=["str", "float", "negative", "bool", "none"])
def test_validate_rejects_bad_max_iter(max_iter):
    handler = with_options({"max_iter": max_iter})

    with pytest.raises(ConfigError, match="max_iter"):
        handler.validate()


@pytest.mark.parametrize("max_iter", [0, 5])
def test_validate_accepts_a_non_negative_max_iter(max_iter):
    handler = with_options({"max_iter": max_iter})

    handler.validate()


@pytest.mark.parametrize(
    "section", ["data", "merit_function", "embedder", "optimization"])
def test_validate_requires_section_to_be_a_dict(section):
    handler = ConfigHandler.from_dict(replaced(section, None))

    with pytest.raises(ConfigError, match=section):
        handler.validate()


@pytest.mark.parametrize("data, name", [
    ({}, "data"),
    (["kernel"], "data"),
    ({"kernel": [1.0, 0.25]}, "kernel"),
], ids=["empty", "list", "list dataset"])
def test_validate_rejects_bad_data(data, name):
    handler = ConfigHandler.from_dict(replaced("data", data))

    with pytest.raises(ConfigError, match=name):
        handler.validate()


# CONFIG as text: YAML cannot read JSON indented with tabs, and JSON
# cannot read YAML in block style
YAML_TEXT = yaml.safe_dump(CONFIG)
JSON_TEXT = json.dumps(CONFIG, indent="\t")


@pytest.mark.parametrize("name, text", [
    ("config.yaml", YAML_TEXT),
    ("config.yml", YAML_TEXT),
    ("config.json", JSON_TEXT),
    ("config", YAML_TEXT),
    ("config.cfg", json.dumps(CONFIG)),
], ids=["yaml", "yml", "json", "no suffix", "json without suffix"])
def test_handler_reads_each_file_type(name, text, tmp_path):
    path = tmp_path / name
    path.write_text(text)

    handler = ConfigHandler(path)

    assert handler.config == CONFIG


@pytest.mark.parametrize("name, text, message", [
    ("empty.yaml", "", "empty"),
    ("bad.yaml", "data: [\n", "YAML"),
    ("bad", "data: [\n", "YAML"),
    ("bad.json", "{\n", "JSON"),
], ids=["empty", "bad yaml", "bad yaml without suffix", "bad json"])
def test_handler_rejects_an_unreadable_file(name, text, message, tmp_path):
    path = tmp_path / name
    path.write_text(text)

    with pytest.raises(ConfigError, match=message):
        ConfigHandler(path)


def test_handler_rejects_a_missing_file(tmp_path):
    path = tmp_path / "config.yaml"

    with pytest.raises(ConfigError, match="not found"):
        ConfigHandler(path)


def test_relative_path_is_read_from_config_dir(tmp_path, monkeypatch):
    folder = tmp_path / "fit"
    folder.mkdir()
    dataset = csv_dataset(folder / "target.csv", CSV_TEXT, path="target.csv")
    config = replaced("data", {"kernel": dataset})
    (folder / "config.json").write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    handler = ConfigHandler("fit/config.json")

    grid, target = handler.load_data()["kernel"]

    assert_array_equal(grid, GRID, strict=True)
    assert_array_equal(target, TARGET, strict=True)


def test_from_dict_reads_relative_path_from_cwd(tmp_path, monkeypatch):
    dataset = csv_dataset(tmp_path / "target.csv", CSV_TEXT,
                          path="target.csv")
    monkeypatch.chdir(tmp_path)
    handler = ConfigHandler.from_dict(replaced("data", {"kernel": dataset}))

    grid, target = handler.load_data()["kernel"]

    assert_array_equal(grid, GRID, strict=True)
    assert_array_equal(target, TARGET, strict=True)


def test_handler_loads_every_dataset():
    unused = {"source": "inline", "grid": [0.0], "target": [3.0]}
    handler = ConfigHandler.from_dict(
        replaced("data", {**CONFIG["data"], "unused": unused}))

    datasets = handler.load_data()

    assert sorted(datasets) == ["kernel", "unused"]


def test_handler_names_the_dataset_that_fails():
    broken = {"source": "inline", "grid": [0.0, 1.0], "target": [1.0]}
    handler = ConfigHandler.from_dict(
        replaced("data", {**CONFIG["data"], "broken": broken}))

    with pytest.raises(ConfigError, match="'broken'"):
        handler.load_data()


@pytest.mark.parametrize("getter, section", [
    ("get_merit_function_config", "merit_function"),
    ("get_embedder_config", "embedder"),
    ("get_optimization_config", "optimization"),
], ids=["merit_function", "embedder", "optimization"])
def test_get_config_returns_a_deep_copy(getter, section):
    handler = ConfigHandler.from_dict(CONFIG)
    first = getattr(handler, getter)()

    first["parameters"].clear()

    assert getattr(handler, getter)() == CONFIG[section]


@pytest.mark.parametrize("text, entries", [
    pytest.param(CSV_TEXT, {}, id="comma by default"),
    pytest.param("0.0;1.0\n0.5;0.25\n", {"delimiter": ";"}, id="semicolon"),
    pytest.param("0.0\t1.0\n0.5\t0.25\n", {"delimiter": "\t"}, id="tab"),
    pytest.param("0.0 1.0\n0.5\t 0.25\n", {"delimiter": None},
                 id="whitespace"),
    pytest.param(" 0.0 , 1.0\n0.5,0.25 \n", {}, id="padded values"),
    pytest.param("# t, K\n" + CSV_TEXT, {}, id="comment"),
    pytest.param("0.0,9.0,1.0\n0.5,9.0,0.25\n", {"columns": [0, 2]},
                 id="third column"),
    pytest.param("1.0,0.0\n0.25,0.5\n", {"columns": [1, 0]},
                 id="reversed columns"),
    pytest.param(CSV_TEXT, {"format": "CSV"}, id="format in capitals"),
])
def test_load_data_reads_csv_columns(text, entries, tmp_path):
    dataset = csv_dataset(tmp_path / "target.csv", text, **entries)

    grid, target = load_data(dataset, "target")

    assert_array_equal(grid, GRID, strict=True)
    assert_array_equal(target, TARGET, strict=True)


def test_load_data_reads_a_csv_with_one_row(tmp_path):
    dataset = csv_dataset(tmp_path / "target.csv", "0.5,0.25\n")

    grid, target = load_data(dataset, "target")

    assert_array_equal(grid, [0.5], strict=True)
    assert_array_equal(target, [0.25], strict=True)


@pytest.mark.parametrize("text, entries, message", [
    pytest.param(CSV_TEXT, {"path": None}, "'path' required", id="no path"),
    pytest.param(CSV_TEXT, {"path": "/nonexistent/target.csv"},
                 "not found", id="no file"),
    pytest.param(CSV_TEXT, {"columns": None}, "'columns' required",
                 id="no columns"),
    pytest.param(CSV_TEXT, {"columns": [0.5, 1]}, "integers",
                 id="float index"),
    pytest.param(CSV_TEXT, {"columns": [-1, 1]}, "non-negative",
                 id="negative index"),
    pytest.param(CSV_TEXT, {"columns": [1, 1]}, "different",
                 id="same index"),
    pytest.param(CSV_TEXT, {"columns": [0]}, "Invalid column",
                 id="one index"),
    pytest.param(CSV_TEXT, {"columns": [0, 2]}, "out of bounds",
                 id="index too large"),
    pytest.param("0.0,abc\n", {}, "non-numeric", id="text value"),
    pytest.param("t,K\n" + CSV_TEXT, {}, "non-numeric", id="header line"),
    pytest.param("0.0,1.0\n0.5\n", {}, "non-numeric", id="ragged rows"),
    pytest.param("", {}, None, id="empty file"),
    pytest.param("# t, K\n", {}, None, id="comments only"),
    pytest.param(CSV_TEXT, {"format": "hdf5"}, "Only CSV", id="hdf5"),
])
def test_load_data_rejects_a_bad_csv(text, entries, message, tmp_path):
    dataset = csv_dataset(tmp_path / "target.csv", text, **entries)

    with pytest.raises(DataLoadError, match=message):
        load_data(dataset, "target")


@pytest.mark.parametrize("grid, target, expected_grid, expected_target", [
    pytest.param(GRID, TARGET, GRID, TARGET, id="floats"),
    pytest.param([0, 1], [2, 1], [0.0, 1.0], [2.0, 1.0], id="integers"),
    pytest.param(0.5, 0.25, [0.5], [0.25], id="numbers"),
])
def test_load_data_reads_inline_arrays(
        grid, target, expected_grid, expected_target):
    dataset = {"source": "inline", "grid": grid, "target": target}

    loaded_grid, loaded_target = load_data(dataset, "target")

    assert_array_equal(loaded_grid, expected_grid, strict=True)
    assert_array_equal(loaded_target, expected_target, strict=True)


def test_load_data_reads_inline_scalar_without_grid():
    dataset = {"source": "inline", "target": 2}

    grid, target = load_data(dataset, "target")

    assert grid is None
    assert type(target) is float and target == 2.0


@pytest.mark.parametrize("entries, message", [
    pytest.param({"grid": GRID}, "'target' required", id="no target"),
    pytest.param({"target": TARGET}, "'grid' required",
                 id="list without grid"),
    pytest.param({"grid": GRID, "target": [1.0]}, "different lengths",
                 id="lengths differ"),
    pytest.param({"grid": [GRID], "target": TARGET}, "1D", id="2-D grid"),
    pytest.param({"grid": ["a", "b"], "target": TARGET}, "convert",
                 id="text"),
])
def test_load_data_rejects_bad_inline_data(entries, message):
    dataset = {"source": "inline", **entries}

    with pytest.raises(DataLoadError, match=message):
        load_data(dataset, "target")


def test_load_data_rejects_an_unknown_source():
    dataset = {"source": "file", "target": TARGET}

    with pytest.raises(DataLoadError, match="Unknown data source"):
        load_data(dataset, "target")
