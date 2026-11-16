# CLAUDE.md: `glefit.utils`

This module holds numerical helpers for the rest of GLEFit: finite differences, special functions and bath discretization, a script that writes target data, and one that plots a fit. Its open issues and accepted limits are in `ISSUES.md` at the repository root, as `UTL-n`.

## Background

GLEFit fits a Markovian embedding so that it reproduces the memory kernel K(t) of a generalized Langevin equation, or its cosine transform, the spectrum Λ(ω). The spectral density is J(ω) = ω Λ(ω). The files here have no common theme. Each one depends only on NumPy or SciPy, except `make_target.py`, which imports `glefit.config`, `glefit.embedding` and `glefit.spectral`, and `plot_objective.py`, which imports Matplotlib, `glefit.config`, `glefit.embedding` and `glefit.merit`.

## Contents

| File | Provides | Used by |
| --- | --- | --- |
| `numderiv.py` | `jacobian`: central finite differences | Fitting Hessian in `glefit.merit` |
| `special.py` | `sinc` | `gen2x2` embedder |
| `bath.py` | `WAM_discretization`: initial guesses from Λ | `make_target.py`, tutorial notebook |
| `make_target.py` | Script that writes the target and an initial guess of a fit | `bin/make_target.py` |
| `plot_objective.py` | Script that plots each step of a fit against the target | `bin/plot_objective.py`; its parser, tests of `glefit.optimization` |

`__init__.py` is empty, so import from the submodules.

## `numderiv.py`

```python
jacobian(f, x, *, order=2)  # f: Rⁿ → Rᵐ, returns (m, n)
```

- **Stencils:** `order` 2 or 4.
- **Step:** by default h = ε^(1/(1+order))·max(1, |x|), adjusted so that x + h is exactly representable. `glefit.merit` calls `jacobian(..., order=4)`, which gives h ≈ 7.4e-4·max(1, |x|).
- **Inputs:** `f` must return a 1-dimensional array, so callers flatten and reshape, as `gradhess_wrt_params` does.
- **Cost:** 2n calls at order 2 and 4n at order 4, plus one at x.


## `special.py`

`sinc(x, nu=0)` is sin(x)/x or its nu'th derivative, without a factor of π, for real or complex x. It is copied unchanged from `rpmdgle/utils/special.py` in rpgle-model-dynamics. It sums spherical Bessel functions, so it is accurate to about 4e-15 relative even at small |x|, where closed forms of the derivatives cancel. For complex x, SciPy overflows when |Im x| > 700.9.

The functions that built the `gen2x2` kernel before EMB-20 (`coscosh`, `sincsinhc`, `expcoscosh`, `expsincsinhc`, `Softplus`, `Softmax` and `Softabs`) were removed once nothing used them.

## `bath.py`

`WAM_discretization(omega, Lambda, nbath, wmax=inf)` follows Walters, Allen and Makri, J. Comput. Chem. 38, 110 (2017). `Lambda` is an array of Λ values on the grid `omega`. The function:

1. fits an interpolating cubic B-spline to the samples,
2. forms C(ω) = ∫ Λ from `omega[0]` to ω,
3. returns `weight = C(wmax)/nbath` and the frequencies that solve C(ωⱼ) = (j + ½)·weight, for j = 0, ..., nbath − 1, by bisection. `wmax` is capped at `omega[-1]`.

It does not check that the grid starts at 0 (UTL-3) or that Λ ≥ 0 (UTL-4). `root_scalar` returns rather than raises when a bisection fails, so the function checks `converged` itself; 100 halvings reach `xtol` = 2e-12 unless the bracket is wider than about 2e18.

The tutorial passes the result to `TwoAuxEmbedder.from_harmonic(weight, omega, gamma)`. For a Debye spectrum with 5 modes, the weight matched the exact value to 1e-13 and the frequencies to 4e-12.

## `make_target.py`

`make_target.py CONFIG` reads a YAML or JSON file, whose keys the docstring of `main` lists, and writes Kₙ or Λₙ of a model to a CSV file that a fit config reads with `columns: [0, 1]`. With an `initial_guess` section, it also writes an `embedder` section for a fit config to a YAML file.

- **Model:** `model.type` is a key of `glefit.spectral.SPECTRAL_MAP`, and `model.parameters` go to its constructor as keyword arguments, so models with different parameters need no change here. The constructor raises `TypeError` or `ValueError` for bad parameters.
- **Grid:** either `linspace: [start, stop, num]` or the `path` of a one-column CSV file, read relative to the config file.
- **Output:** relative to the working directory, with 17 significant digits, not divided by Kₙ(0). A `#` header records the inputs.
- **Initial guess:** `exact` is the exact embedding of the model, from `EXACT_EMBEDDINGS` in the script, which holds Debye as `prony` and the Brownian oscillator as `gen2x2`, and is refused for ωₙ ≠ 0. `wam` is a `multi` of `nbath` `gen2x2` embedders from `from_harmonic`, at the WAM frequencies of Λₙ on its own grid, which should start at 0 (UTL-3). The damping rate of each is the spacing to the frequency below, γ₁ = ω₂ − ω₁, or ω₁ for one mode.
- **Errors:** an unknown key, an unknown `type` or `property`, a malformed `grid` or `initial_guess`, or an exact guess that does not exist raises `ConfigError`, before anything is computed or written.
- **Accuracy:** `nmK` uses the default tolerances of `quad` (SPC-6), which the config cannot change.

## `plot_objective.py`

`plot_objective.py CONFIG [--traj TRAJ]` rebuilds the embedder and merit function from a fit config, and plots the fitted curve against the target at each step of the trajectory that the optimizer wrote, `traj.out` in the working directory by default. The window shows the target (red), the fitted total (black) and, for a `multi` embedder, each component (dashed), with a slider over the steps. The title shows the distance at the selected step.

| Function | Does |
| --- | --- |
| `parse_trajectory_file(path)` | Returns step numbers and parameter arrays from a trajectory |
| `setup_from_config(config_path)` | Builds the embedder and merit function, then turns off the embedder's mappers |
| `compute_contributions(embedder, merit, params)` | Sets `embedder.conventional_params`, then evaluates the property for each component and for the total |
| `plot_trajectory(merit, embedder, steps, params_list)` | Draws the figure and slider, then calls `plt.show()` |

- **Trajectory:** the parser reads the format that `Optimizer.log` writes, a line with `|` and `Step N,`, the array, then a blank line, so change both together. Values have 7 significant digits, so the curves use rounded parameters.
- **Config:** read with `ConfigHandler`, as in `optimize.py`. The `optimization` section is ignored.
- **Merit function:** uses `function()`, `distance`, `grid` and `target`, and sets `merit.emb` to each component in turn.
- **Embedder:** uses the private attributes `_embs` and `_mappers`. Setting `_mappers = None` turns off range checks on the values read back.
- **Display:** needs an interactive Matplotlib backend, and `plt.show()` blocks until the window closes. The y-range is fixed from min(0, target) to max(target), with a 5% margin, so curves outside it are cut off.

## Conventions

- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Adding features

- **Special functions:** follow the `nu` interface of `sinc`, and test real and complex arguments, small and large, against SymPy.
- **Finite differences:** add first-derivative stencils to `STENCILS`, keyed by their order of accuracy. The default step uses that order.
- **Discretization schemes:** add them to `bath.py`, taking Λ on a grid and returning values that `from_harmonic` accepts.
- **Scripts:** read the config with `ConfigHandler`, look up each `type` in its registry as `optimize.py` does, and add a symlink in `bin/`.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_utils
```

`test_special.py` tests `sinc` against SymPy. `test_numderiv.py` tests that `jacobian` is exact for polynomials, its default step through the known truncation error for sin(ωx), and its input checks. `test_bath.py` compares `WAM_discretization` with the closed form for a Debye spectrum, on a grid from 0 and on one from 0.5 (UTL-3), with `wmax` inside and beyond the grid. On 4001 points, the weight agrees to 1e-12 and the frequencies to 1e-11, which is the bisection's `xtol`. `test_make_target.py` runs the script on configs in a temporary directory, and reads the output back with `load_data`. It checks that the target equals `nmK` or `nmLambda` exactly, that both kinds of grid are read, where the output goes, the header, and the errors for malformed configs. For the initial guess, it rebuilds the embedder with `EMBEDDER_MAP`: the exact guess must have the Λ of Debye or of the Brownian oscillator, and the WAM guess the closed-form Debye frequencies, damping rates and K(0), and the frequencies of `WAM_discretization` of Λₙ at ωₙ > 0. `test_plot_objective.py` runs a short fit of a `multi` embedder, a `prony` and a `pronycos` term, with `glefit.optimize.main` in a temporary directory, and plots it with the `Agg` backend and `plt.show` patched out. At step 0, for K and Λ, the total and each component must equal the closed forms, and the title the steps and the distance. A fit of each other type alone must be drawn as its closed-form total only. It also checks the target line, the y-range, the trajectory that `main` reads, and that the plotted embedder of every type accepts values outside the mappers' bounds, such as Ω = 0 for `gen2x2`. The slider is not reachable, so its callback is untested. `tests/test_optimize.py` and `tests/test_optimization.py` test `parse_trajectory_file`.

New or changed code should have tests that:

- compare `special.py` functions and their derivatives against SymPy, at small and large arguments
- pass integer and scalar inputs
- check `jacobian` against exact derivatives, as `tests/test_utils/test_numderiv.py` does
- check `WAM_discretization` against the closed form for a Debye spectrum, where C(ω) = η ωc arctan(ω/ωc)
- check the curves of `plot_objective.py` against the closed-form `kernel` or `spectrum` of each embedder type
