# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. More specific, frequently updated instructions for code development are found in @CODING_ASSIGNMENT.md

## Overview

GLEFit fits the parameters of an auxiliary-variable Markovian embedding so that the memory kernel or spectrum of a one-dimensional generalized Langevin equation matches target data. `tutorials/spectrum_optimization.ipynb` is the best end-to-end example.

## Commands

There is no build step, linter or CI. The empty `pytest.ini` makes the repository root pytest's rootdir, so that `tests/conftest.py` is loaded however the tests are run. `glefit` is not installed: it is imported through `PYTHONPATH`. For interactive use, `source env.sh` sets that up and adds `bin/` (symlinks into `src/glefit/`) to `PATH`.

Use the `tools` conda environment, not the `glefit` environment defined in `environment.yml`.

`optimize.py` writes these files to the working directory, and none of them are gitignored:

- `trajfile` (default `traj.out`): conventional parameters at each step.
- `logfile`: appended to. `-` means stdout, and leaving it unset discards the log.
- `optimized_Ap_matrix.json`: the full drift matrix from `propagation_matrix()`, which is the minimal gauge for `gen2x2`, written only on convergence.

A run that does not converge prints nothing and exits 0, so check for the JSON file.

`tutorials/spectrum_optimization.ipynb` creates its workspace under `tutorials/spectrum_optimization/`.

## Test status

Open issues are listed in `ISSUES.md`.

`tests/` is the maintained suite, written to `tests/CLAUDE.md`. So far it covers `glefit.embedding`, the merit functions of `glefit.merit` (values, `grad_wrt_A`, the distance and its derivatives, input checks, `from_dict` and the weights), `sinc` in `glefit.utils.special` and `jacobian` in `glefit.utils.numderiv`: `tests/test_embedding` has 386 passing tests, `tests/test_merit` has 161, `tests/test_utils` has 28, `tests/test_optimize.py` checks the file that `optimize.py` writes, and `tests/test_mappers.py` checks the bounded mappers and their derivatives against SymPy, and that they reject parameters on their bounds. Its references are SymPy and mpmath.

`legacy_tests/` is the original suite and must not be edited. Run it with `--continue-on-collection-errors`: `test_utils/test_special.py`, `test_utils/test_numderiv.py` and `test_mappers/test_intervals.py` cannot be imported. 73 tests pass, 51 fail and those three files fail to import. The failures are known, caused by tests lagging the code:

- `test_config/test_config_handler.py` (27): written for the config schema in `e1dc6d0` (`merit_function.property`, `data_ref`, `optimization.optimizer`, `max_steps`, `project.*`) and for the checkpoint methods that `7da42a9` commented out. For the current schema, read `legacy_tests/test_optimize/fixtures/` and `tutorials/fixtures/`.
- `test_embedding/test_multi.py` (3): expects the drift-matrix signs from before the fix in `e2299f6`. `docs/Auxiliary variables (SDE) - Kantorovich prescription.md` confirms the current signs.
- `test_config/test_data_io.py::test_missing_source` (1): expects an error, but `load_data` defaults `source` to `inline`.
- `test_embedding/test_gen2x2.py` (13), `test_merit/test_properties.py::TestSpectrum` (4) and `test_optimize/test_optimize.py` (2, the `gen2x2` and `multi` fixtures): they pass `sigma` to `gen2x2`, which raises `TypeError` since EMB-20.
- `test_embedding/test_gen2x2.py::test_matrix_rotation` (1): calls `TwoAuxEmbedder.rotate_matrix`, which was removed when `from_matrix` switched to rotation invariants (EMB-22).

`test_optimize.py` only checks that runs finish without raising. It does not assert convergence, and the `pronycos_embedder.yaml` fixture does not converge. It writes `log.out`, `trajectory.out` and `optimized_Ap_matrix.json` into the working directory, so run `legacy_tests` from a scratch directory: `PYTHONPATH=<repo>/src python -m pytest --continue-on-collection-errors <repo>/legacy_tests`.

## Working rules 

- Never commit. Propose one commit per concern: a pure refactor, or a fix that starts with a failing test. Messages are short: `type(module): subject`, at most 3 sentences of rationale, then `Parts of the code generated with <LLM>`.
- Before a new test file or a fix, send a short plan and wait for approval.
- In code, comments and test names, keep lines to 80 characters, and do not cite `docs/` or equation numbers.
- The maintainer edits files between steps. Re-read a file before editing it.
- Python runs in the `tools` conda env with `PYTHONPATH=src`. `conda run` does not forward stdin, so put scripts in the scratchpad instead of heredocs, and write input to a file instead of piping it: a pipe into `conda run python` reads nothing and fails silently.
- Run `legacy_tests` from the scratchpad directory, with `--continue-on-collection-errors`.
- Show that new tests can fail: copy `src/` to the scratchpad, apply one plausible bug at a time, run the suite against the copy (`PYTHONPATH=<copy>/src`) with a timeout per run, and report any bug that is missed and whether it is equivalent. Also run each new test alone, and the suite in random order with a few seeds (`--randomly-seed=N`; `-p no:randomly` turns `pytest-randomly` off). Nothing may be installed.
- A bug can vanish at a lucky test point. Scan a few values before choosing a test case.
- Leave `tutorials/` alone for now.

## Architecture

### Config-driven pipeline

`Optimization.from_config` in `src/glefit/optimize.py` builds everything from one YAML or JSON file, in this order: `ConfigHandler`, embedder, datasets, merit function, optimizer. Each `type` string is looked up in a registry: `EMBEDDER_MAP`, `PROPERTY_MAP` and `OPT_MAP` in the matching package `__init__.py`, and `METRICS` in `merit/_base.py`. A new component needs a class and a registry entry.

Config behavior that the validator does not make obvious:

- `optimization.parameters` goes to the optimizer constructor (`logfile`, `trajfile`). `optimization.options` goes to `run()` (`gtol`, `max_step`), after the driver pops `max_iter`.
- Input paths resolve against the config file's directory. Output paths resolve against the working directory.
- `project` is never read. `restart` is validated but ignored, because restarts are not implemented.
- Mappers cannot be set from a config, so each embedder's defaults apply. This is by design: custom mappers are for debugging only.

### The drift matrix

Every fitted quantity derives from the (n+1)×(n+1) matrix `Ap = [[0, θᵀ], [-θ, A]]`. Row and column 0 belong to the physical coordinate, and the coupling is skew-symmetric. `MemoryKernel` computes θᵀ exp(-|t|A) θ and `MemorySpectrum` computes θᵀ A (A² + ω²)⁻¹ θ, both with θᵀ from row 0 and θ from the negated column 0.

### Three parameter spaces

| Space | Size | Role |
| --- | --- | --- |
| conventional | `nparam` | User-facing: config files and trajectory output |
| primitive | `ndof` | Non-degenerate: input to `compute_drift_matrix` and `drift_matrix_param_grad` |
| mapped `x` | `ndof` | Unconstrained: the optimizer steps here, with one `BaseMapper` per primitive parameter |

Conventional and primitive parameters differ only for `TwoAuxEmbedder`, which maps `[θ₁, θ₂, γ, δ, ζ]` to `[r, γ, Ω, φ]` to remove the embedding degeneracy. Every real r, Ω and φ with γ > 0 gives a valid embedding, but Ω is log-mapped as well as γ, which keeps fits away from Ω = 0 and makes them independent of the time unit. It builds the drift matrix in the maximal gauge δ = γ, and `propagation_matrix()` returns the minimal gauge, which is better suited to simulations. The `conventional_params`, `primitive_params` and `x` setters keep the spaces in sync, so always go through them.

### Two independent ways of computing objective function derivatives

- **Derivatives via chain rule w.r.t. drift matrix:** `BaseProperty.grad_wrt_params` contracts the analytic `grad_wrt_A` with `emb.drift_matrix_x_grad`. The Hessian is a 4th-order finite difference of that gradient (`gradhess_wrt_params`), not an analytic expression.
- **Closed-form:** `kernel_*` and `spectrum_*` on each embedder, exposed through `kernel()`, `spectrum()` and `spectral_density()`. The optimizers and merit functions never call these.

Fits therefore depend only on `compute_drift_matrix` and `drift_matrix_param_grad`. A bug on one path does not show up on the other.

### Embedders

Subclass `BaseEmbedder` and implement `from_dict`, `__len__`, the `nparam` property, `compute_drift_matrix`, `drift_matrix_param_grad`, and the `func`, `grad` and `hess` variants of `kernel_*` and `spectrum_*`, and set `_primitive_names` for error messages. If primitive parameters differ from conventional ones, also override the `ndof` property, `to_primitive` and `to_conventional`.

`BaseEmbedder.__init__` reads `ndof`, so set anything it needs before calling `super().__init__`. Set default mappers with `kwargs.setdefault("mappers", [...])`.

`MultiEmbedder` is a direct sum whose components own the parameter state. Its setters and map methods delegate by slice, and the conventional and primitive slices differ when a component is `gen2x2`.

### Optimizers

`Optimizer` follows ASE's design. `__init__` calls `initialize()`, then `run()` repeats `step()` and `update()` until `converged()` or `max_iter`. `EigenvectorFollowing` subclasses `NewtonRaphson` and changes only `step()`, so it can move away from saddle points.

### Standalone code

The pipeline does not use `spectral/` (model spectral densities). `utils/bath.py::WAM_discretization` supplies the initial guesses in the tutorial notebook.

## Conventions

Source files start with the `@File`/`@Time`/`@Author`/`@Desc` header block, followed by `from __future__ import print_function, division, absolute_import`.
