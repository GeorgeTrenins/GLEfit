# CLAUDE.md: `glefit.utils`

This module holds numerical helpers for the rest of GLEFit: finite differences, special functions, linear solves, bath discretization, ring-polymer frequencies and unit conversion.

## Background

GLEFit fits a Markovian embedding so that it reproduces the memory kernel K(t) of a generalized Langevin equation, or its cosine transform, the spectrum Λ(ω). The spectral density is J(ω) = ω Λ(ω). The files here have no common theme, and each one depends only on NumPy, SciPy or SymPy.

## Contents

| File | Provides | Used by |
| --- | --- | --- |
| `numderiv.py` | `jacobian`: central finite differences | Fitting Hessian in `glefit.merit` |
| `special.py` | `sinc` | `gen2x2` embedder |
| `linalg.py` | `mat_inv_vec` (`np.linalg.solve`), `symmat_inv_vec` (Cholesky) | NR step. `symmat_inv_vec` is unused |
| `bath.py` | `WAM_discretization`: initial guesses from Λ | Tutorial notebook |
| `rp.py` | `nmfreq`: ring-polymer normal-mode frequencies | Unused. Gives ωₙ for the `nm` methods in `glefit.spectral` |
| `units.py` | Unit systems built on `sympy.physics.units` | Unused |

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

`WAM_discretization(omega, Lambda, nbath, wmax=inf, options=None)` follows Walters, Allen and Makri, J. Comput. Chem. 38, 110 (2017). `Lambda` is an array of Λ values on the grid `omega`, not a function. The function:

1. fits an interpolating B-spline to the samples, passing `options` to `scipy.interpolate.make_interp_spline`,
2. forms C(ω) = ∫ Λ from `omega[0]` to ω,
3. returns `weight = C(wmax)/nbath` and the frequencies that solve C(ωⱼ) = (j + ½)·weight. `wmax` is capped at `omega[-1]`.

The tutorial passes the result to `TwoAuxEmbedder.from_harmonic(weight, omega, gamma)`. For a Debye spectrum with 5 modes, the weight matched the exact value to 1e-13 and the frequencies to 4e-12.

## `rp.py` and `units.py`

- `nmfreq(beta, P, n, hbar=1.0)` returns 2(P/βħ)|sin(πn/P)|, the frequency of normal mode n of a free ring polymer with P beads.
- `units.py` defines unit systems as classes: `SI`, `atomic`, `hartAng`, `kcalAfs`, `kcalAamu`, `eVAamu` and `cmbohramu`. Each gives constants (`hbar`, `kb`, `e`, `amu`, `me`, `c`), the SI size of its units through attributes such as `.energy` and `.time`, string parsing such as `str2base("1 fs")`, and wavenumber conversions. For example, the `eVAamu` time unit is 10.18 fs, and `atomic().wn2energy(1.0)` gives 4.556e-6. Importing it registers new quantities, such as `angstrom` and `hartree`, in SymPy's global SI system.

## Conventions

- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Known gaps

Checked at commit `ffd03fd`. Everything under Surprising limits was reproduced by running the code.

### Surprising limits

- **`WAM_discretization` needs a grid that starts at 0 and a non-negative Λ.** It integrates from `omega[0]`, so a Debye grid starting at 0.5 gave a weight of 0.437 instead of 0.506. A region of negative Λ gave meaningless frequencies with no error: five between 0.05 and 0.42, then 11.4, 13.8 and 17.4.
- **`symmat_inv_vec` raises `LinAlgError`** for a matrix that is not positive definite, such as an indefinite Hessian.

### Missing features

- `units.py` is not connected to the config or the data, so all inputs must already share one unit system.

### Test gaps

- `tests/test_utils/test_special.py` tests only `sinc`, against SymPy.
- `tests/test_utils/test_numderiv.py` tests `jacobian`, but only the exactness for polynomials, the default step through the known truncation error for sin(ωx), and the input checks.
- `bath.py`, `linalg.py`, `rp.py` and `units.py` have no tests.

### Outdated text in the code

- `WAM_discretization` documents `Lambda` as a callable and does not document `omega`, `wmax` or `options`.
- The `units.py` docstring contains a stray `"""` and lists five of the seven unit systems.

## Adding features

- **Special functions:** follow the `nu` interface of `sinc`, and test real and complex arguments, small and large, against SymPy.
- **Finite differences:** add first-derivative stencils to `STENCILS`, keyed by their order of accuracy. The default step uses that order.
- **Discretization schemes:** add them to `bath.py`, taking Λ on a grid and returning values that `from_harmonic` accepts.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_utils
```

New or changed code should have tests that:

- compare `special.py` functions and their derivatives against SymPy, at small and large arguments
- pass integer and scalar inputs
- check `jacobian` against exact derivatives, as `tests/test_utils/test_numderiv.py` does
- check `WAM_discretization` against the closed form for a Debye spectrum, where C(ω) = η ωc arctan(ω/ωc)
