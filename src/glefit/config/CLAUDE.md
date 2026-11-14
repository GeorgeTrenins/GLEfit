# CLAUDE.md: `glefit.config`

This module reads and checks the YAML or JSON file that defines a fit, and loads its target data. This file lists conventions and pitfalls. Open issues are in `ISSUES.md` at the repository root, as `CFG-n`.

## Model

`config_handler.py` holds `ConfigHandler` and `ConfigError`, and `data_io.py` holds `load_data` and `DataLoadError`. `__init__.py` is empty, so import from the submodules. `optimize.py` and `analysis/plot_objective.py` call `ConfigHandler(path)` and `validate()`, then `load_data()` and the `get_*_config` methods, and look up each section's `type` in its registry. Working configs are in `legacy_tests/test_optimize/fixtures/` and `tutorials/fixtures/`; those in `legacy_tests/test_config/fixtures/` follow an older schema. The `parameters` of each type are described in the `CLAUDE.md` of `embedding`, `merit` and `optimization`.

```yaml
data:                      # required: one or more named datasets
  spectrum:
    source: external       # defaults to inline (CFG-4)
    path: target.csv       # relative to the config file
    format: csv            # optional; csv is the only format
    columns: [0, 1]        # grid column, then target column, from 0
    delimiter: ","         # default ","; null for whitespace
  kernel:
    source: inline
    grid: [0.0, 0.5, 1.0]
    target: [1.0, 0.5, 0.1]
merit_function:            # required
  type: spectrum
  parameters: {data: spectrum, metric: squared}
embedder:                  # required
  type: prony
  parameters: {theta: 1.0, gamma: 0.5}
optimization:             # required
  type: EF
  parameters: {trajfile: traj.out, logfile: "-"}  # to the constructor
  options: {max_iter: 100, gtol: 0.01}  # to run(), after popping max_iter
restart: {enabled: false}  # optional, never read
project: {}                # optional, never read
```

## Checks

- **`__init__`:** the file exists, parses and is not empty. It makes `path` absolute for `source: external` datasets, before `validate()` has checked any types (CFG-3).
- **`validate()`:** the four required sections are dictionaries, `data` is a non-empty dictionary of dictionaries, and `merit_function`, `embedder` and `optimization` each have `type` and `parameters`. `optimization.options` is a dictionary with `max_iter`, a non-negative `int` that is not a `bool`. Other sections are never read.
- **`load_data()`:** every dataset loads, used or not. `DataLoadError` becomes `ConfigError`, naming the dataset.
- **Elsewhere:** `Optimization.from_config` checks each `type` against its registry (OPT-7). Each class's `from_dict` checks its `parameters`, including whether `data` names a dataset.

## Conventions

- **File type:** `.yaml` and `.yml` are read as YAML, `.json` as JSON, and any other suffix as YAML (CFG-6).
- **Paths:** input paths are relative to the config file, and output paths to the working directory. `from_dict` builds a handler without a file and leaves paths as given, so relative ones are then relative to the working directory.
- **Data:** `load_data` returns `(grid, target)` as 1-dimensional float arrays of equal length. CSV files are read with `np.loadtxt`, so `#` starts a comment and any other header line fails. An inline scalar `target` with no `grid` loads as `(None, float(target))`, which no merit function accepts.
- **Copies:** the `get_*_config` methods return deep copies, which `optimize.py` relies on when it pops `max_iter`.
- **Logging:** messages go to `logging` at level `INFO`, so they are hidden unless the caller sets up logging.
- **Unused code:** `to_dict` and `_json_encoder` have no callers, and the checkpoint methods are commented out. Only tests call `from_dict`.
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## New section or key

Check its structure in `validate()`, raising `ConfigError`, and add a `get_<section>_config` method that returns a deep copy. Make an input path absolute against the config file's directory. Read the section in `optimize.py`, and in `analysis/plot_objective.py` if plotting needs it.

## Testing

Run `PYTHONPATH=src conda run -n tools python -m pytest tests/test_config.py` from the repository root. It checks that `ConfigHandler` reads each file type and rejects files it cannot read. `validate()` must require every key that `optimize.py` reads, with each section a dictionary and `max_iter` a non-negative integer, and ignore `restart` and `project`, whatever they hold. Relative data paths must be read from the config file's directory, or from the working directory after `from_dict`, every dataset must load, and the `get_*_config` methods must return deep copies. For `load_data`, it checks each CSV layout and delimiter, inline arrays and scalars, and that bad input raises `DataLoadError`. The known defects CFG-1 to CFG-5 are not pinned. `legacy_tests/test_config/test_data_io.py` also covers `load_data`, and only `test_missing_source` fails (CFG-4). `legacy_tests/test_config/test_config_handler.py` follows the old schema, and 28 of its tests fail.
