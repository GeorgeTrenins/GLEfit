# CLAUDE.md: `glefit.utils`

This module holds numerical helpers for the rest of GLEFit: finite differences, special functions and bath discretization, and a script that writes target data. Its open issues and accepted limits are in `ISSUES.md` at the repository root, as `UTL-n`.

## Background

GLEFit fits a Markovian embedding so that it reproduces the memory kernel K(t) of a generalized Langevin equation, or its cosine transform, the spectrum Λ(ω). The spectral density is J(ω) = ω Λ(ω). The files here have no common theme. Each one depends only on NumPy or SciPy, except `make_target.py`, which imports `glefit.config`, `glefit.embedding` and `glefit.spectral`.

## Contents

| File | Provides | Used by |
| --- | --- | --- |
| `numderiv.py` | `jacobian`: central finite differences | Fitting Hessian in `glefit.merit` |
| `special.py` | `sinc` | `gen2x2` embedder |
| `bath.py` | `WAM_discretization`: initial guesses from Λ | `make_target.py`, tutorial notebook |
| `make_target.py` | Script that writes the target and an initial guess of a fit | `bin/make_target.py` |

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
- **Initial guess:** `exact` is the exact embedding of the model, from `EXACT_EMBEDDINGS` in the script, which so far holds only Debye as `prony`, and is refused for ωₙ ≠ 0. `wam` is a `multi` of `nbath` `gen2x2` embedders from `from_harmonic`, at the WAM frequencies of Λₙ on its own grid, which should start at 0 (UTL-3). The damping rate of each is the spacing to the frequency below, γ₁ = ω₂ − ω₁, or ω₁ for one mode.
- **Errors:** an unknown key, an unknown `type` or `property`, a malformed `grid` or `initial_guess`, or an exact guess that does not exist raises `ConfigError`, before anything is computed or written.
- **Accuracy:** `nmK` uses the default tolerances of `quad` (SPC-6), which the config cannot change.

## Conventions

- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Adding features

- **Special functions:** follow the `nu` interface of `sinc`, and test real and complex arguments, small and large, against SymPy.
- **Finite differences:** add first-derivative stencils to `STENCILS`, keyed by their order of accuracy. The default step uses that order.
- **Discretization schemes:** add them to `bath.py`, taking Λ on a grid and returning values that `from_harmonic` accepts.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_utils
```

`test_special.py` tests `sinc` against SymPy. `test_numderiv.py` tests that `jacobian` is exact for polynomials, its default step through the known truncation error for sin(ωx), and its input checks. `test_bath.py` compares `WAM_discretization` with the closed form for a Debye spectrum, on a grid from 0 and on one from 0.5 (UTL-3), with `wmax` inside and beyond the grid. On 4001 points, the weight agrees to 1e-12 and the frequencies to 1e-11, which is the bisection's `xtol`. `test_make_target.py` runs the script on configs in a temporary directory, and reads the output back with `load_data`. It checks that the target equals `nmK` or `nmLambda` exactly, that both kinds of grid are read, where the output goes, the header, and the errors for malformed configs. For the initial guess, it rebuilds the embedder with `EMBEDDER_MAP`: the exact guess must have the Debye Λ, and the WAM guess the closed-form Debye frequencies, damping rates and K(0), and the frequencies of `WAM_discretization` of Λₙ at ωₙ > 0.

New or changed code should have tests that:

- compare `special.py` functions and their derivatives against SymPy, at small and large arguments
- pass integer and scalar inputs
- check `jacobian` against exact derivatives, as `tests/test_utils/test_numderiv.py` does
- check `WAM_discretization` against the closed form for a Debye spectrum, where C(ω) = η ωc arctan(ω/ωc)
