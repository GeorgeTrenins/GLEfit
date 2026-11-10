# CLAUDE.md: `glefit.config`

This module reads and checks the YAML or JSON configuration file that defines a fit, and loads its target data.

## Background

A GLEFit run fits a Markovian embedding to target data, such as a memory kernel K(t) on a time grid or a spectrum Λ(ω) on a frequency grid. One config file names the data, the merit function, the embedder and the optimizer. `src/glefit/optimize.py` and `analysis/plot_objective.py` read it through `ConfigHandler`, then build each part by looking up its `type` in a registry: `EMBEDDER_MAP`, `PROPERTY_MAP` and `OPT_MAP`.

## Contents

- `config_handler.py`: `ConfigHandler` and `ConfigError`.
- `data_io.py`: `load_data` and `DataLoadError`.
- `__init__.py` is empty, so import from the submodules.

```python
from glefit.config.config_handler import ConfigHandler
handler = ConfigHandler("config.yaml")   # reads the file, makes data paths absolute
handler.validate()                       # checks the structure only
datasets = handler.load_data()           # {name: (grid, target)}
emb_cfg = handler.get_embedder_config()  # a deep copy
```

## The config file

`tests/test_optimize/fixtures/` and `tutorials/fixtures/` hold working examples. The YAML files in `tests/test_config/fixtures/` follow an older schema.

```yaml
data:                        # required: one or more named datasets
  spectrum:
    source: external         # default inline, so files need this line
    path: target.csv         # relative to the config file
    format: csv              # only csv
    columns: [0, 1]          # grid column, target column, from 0
    delimiter: ","           # default ","; null for whitespace
  kernel:
    source: inline
    grid: [0.0, 0.5, 1.0]
    target: [1.0, 0.5, 0.1]  # or a single number with no grid

merit_function:              # required
  type: spectrum             # a PROPERTY_MAP key
  parameters:                # required
    data: spectrum           # a dataset name
    metric: squared
    weight: 1.0              # optional

embedder:                    # required
  type: multi                # an EMBEDDER_MAP key
  parameters:                # required
    embedders:
      - {type: prony, parameters: {theta: 1.0, gamma: 0.5}}

optimization:                # required
  type: EF                   # an OPT_MAP key
  parameters:                # required, passed to the optimizer constructor
    trajfile: traj.out       # relative to the working directory
    logfile: "-"
  options:                   # required, passed to run()
    max_iter: 100            # required: an integer ≥ 0; 0 only evaluates the start
    gtol: 0.01
    max_step: 0.2

restart:                     # optional, checked, then ignored
  enabled: false
  restart_file: chk.json
project: {}                  # optional, never read
```

Keys not listed here are ignored. See `glefit/embedding/CLAUDE.md`, `glefit/merit/CLAUDE.md` and `glefit/optimization/CLAUDE.md` for the `parameters` of each type.

## What is checked where

| Stage | Checks |
| --- | --- |
| `ConfigHandler.__init__` | The file exists and parses. It makes `path` absolute for `source: external` datasets, and `restart_file` if restarts are enabled |
| `validate()` | The four required sections are dicts. `data` is not empty. `type` and `parameters` exist in `merit_function` and `embedder`, and `type`, `parameters` and `options` in `optimization`. `options` is a dict with `max_iter`, a non-negative `int` that is not a `bool`. If restarts are enabled, the file exists |
| `load_data()` | Each dataset loads: the file exists, the columns exist and differ, values are numbers, and inline grid and target have equal length |
| Each class's `from_dict` | Everything else, including whether `type` exists and whether the merit function's `data` names a dataset |

## Data loading

`load_data(config_dict, name)` in `data_io.py` returns `(grid, target)` as 1-dimensional float arrays. An inline target with no grid returns `(None, float(target))`. `ConfigHandler.load_data` loads every dataset, even unused ones, and turns `DataLoadError` into `ConfigError`.

External files are read with `np.loadtxt`, so `#` starts a comment and any other header line fails. `tests/test_optimize/fixtures/expohmic_nm_spectrum.csv` shows the expected layout.

## Conventions

- **File type:** `.yaml` and `.yml` are read as YAML and `.json` as JSON. Any other suffix is read as YAML, which also accepts JSON.
- **Paths:** input paths are relative to the config file. Output paths are relative to the working directory.
- **Copies:** the `get_*_config` methods and `to_dict` return deep copies. `optimize.py` relies on this when it pops `max_iter`.
- **`from_dict`:** builds a handler without reading a file. It does not make paths absolute, so relative data paths are then relative to the working directory.
- **Logging:** messages go to the `logging` module at level `INFO`, so they are hidden unless the caller sets up logging.
- **Unused code:** `get_restart_config`, `to_dict`, `from_dict` and `_json_encoder` have no callers. The checkpoint methods are commented out.
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Known gaps

Checked at commit `ffd03fd`. Everything under Wrong results and Surprising limits was reproduced by running the code.

### Wrong results

- **A single-column CSV file is read as one point.** `np.loadtxt` returns shape (n,), which `_load_external` reshapes to one row of n columns. A file with values 0, 1, 2, 3 and `columns: [0, 1]` gave grid [0.0] and target [1.0] with no error. Fix: pass `ndmin=2` to `np.loadtxt` and drop the reshape.

### Surprising limits

- **Malformed sections raise Python errors, not `ConfigError`.** Path resolution runs in `__init__`, before `validate()`. `data` given as a list raises `AttributeError`, as do `restart: true` and a top level that is a list.
- **Missing keys found late.** An unknown `type` raises `KeyError` from its registry (OPT-7). A missing key inside `embedder.parameters` raises `KeyError` from the embedder's `from_dict` (EMB-23).
- **`source` defaults to `inline`.** A dataset with `path` but no `source` fails with "'target' required for inline source".
- **The JSON fallback never runs.** For an unknown suffix, `_load_yaml` turns `yaml.YAMLError` into `ConfigError`, so the `except yaml.YAMLError` around it never catches. This does no harm, because YAML reads JSON.

### Missing features

- Restarts and checkpoints. `restart` is checked but ignored.
- Mappers, model spectral densities from `glefit.spectral`, and per-point weights from a file cannot be set in a config.
- Only one merit function per config.
- `validate()` does not check types against the registries or the contents of `parameters`.

### Test gaps

- `tests/test_config/test_config_handler.py`: 27 tests fail. They were written for the schema in `e1dc6d0` (`merit_function.property`, `data_ref`, `optimization.optimizer`, `max_steps`) and for the checkpoint methods, which `7da42a9` commented out.
- `tests/test_config/test_data_io.py::test_missing_source` fails, because it expects an error but `source` defaults to `inline`. The other `data_io` tests pass.
- Nothing tests single-column files, whitespace delimiters, or relative paths through `from_dict`.

### Outdated text in the code

- `validate()` reports a missing `optimization.type` as "'optimization.optimizer' is required".
- The module docstring of `config_handler.py` mentions checkpoints and "tracking optimization state", which are not implemented.
- `load_data` is annotated as always returning two arrays.
- The `data_io.py` header has `@Desc    :   None` before the real description.

## Adding features

### A new config section or key

1. Add structural checks to `validate()`, raising `ConfigError`.
2. If it holds an input path, make it absolute in `_resolve_paths`. Make that method tolerate wrong types, since it runs before `validate()`.
3. Add a `get_<section>_config` method that returns a deep copy.
4. Read it in `optimize.py`, and in `analysis/plot_objective.py` if plotting needs it.
5. Add a fixture to `tests/test_optimize/fixtures/`.

### Other changes

- **New data source or format:** add a branch to `load_data` or `_load_external`, returning float arrays of equal length.
- **Per-point weights from a file:** return a third column from `load_data`, and update every `from_dict` that unpacks `(grid, target)`.
- **Clearer errors:** check registry keys in `validate()` by importing the maps, and wrap `_resolve_paths` so that type errors become `ConfigError`.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_config
```

Expect the 28 known failures above. New tests should use the current schema, as in `tests/test_optimize/fixtures/`, and should check:

- that each malformed section raises `ConfigError`, not `AttributeError` or `KeyError`
- one-column, one-row and empty files, comment lines, and both delimiters
- paths relative to the config file when the working directory is elsewhere
