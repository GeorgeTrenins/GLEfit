# CLAUDE.md: `glefit.utils`

This module holds numerical helpers for the rest of GLEFit: finite differences, special functions, linear solves, bath discretization, ring-polymer frequencies and unit conversion.

## Background

GLEFit fits a Markovian embedding so that it reproduces the memory kernel K(t) of a generalized Langevin equation, or its cosine transform, the spectrum Λ(ω). The spectral density is J(ω) = ω Λ(ω). The files here have no common theme, and each one depends only on NumPy, SciPy or SymPy.

## Contents

| File | Provides | Used by |
| --- | --- | --- |
| `numderiv.py` | `diff`, `jacobian`: central finite differences | Fitting Hessian in `glefit.merit`, tests |
| `special.py` | `sinc`; `coscosh`, `sincsinhc`, `expcoscosh`, `expsincsinhc`, `Softplus`, `Softmax`, `Softabs` | `gen2x2` embedder. `sinc` is not used yet |
| `linalg.py` | `mat_inv_vec` (`np.linalg.solve`), `symmat_inv_vec` (Cholesky) | NR step, `MemorySpectrum`. `symmat_inv_vec` is unused |
| `bath.py` | `WAM_discretization`: initial guesses from Λ | Tutorial notebook |
| `rp.py` | `nmfreq`: ring-polymer normal-mode frequencies | Unused. Gives ωₙ for the `nm` methods in `glefit.spectral` |
| `units.py` | Unit systems built on `sympy.physics.units` | Unused |

`__init__.py` is empty, so import from the submodules.

## `numderiv.py`

```python
diff(f, x, *, h=None, rel_step=None, max_step=None, order=2, nu=1, args=None, kwargs=None)  # f: R → R
jacobian(f, x, *, fshape=None, h=None, rel_step=None, max_step=None, order=2, value=False, args=None, kwargs=None)  # f: Rⁿ → Rᵐ, returns (m, n)
```

- **Stencils:** `diff` supports `nu=1` with `order` 2 or 4, and `nu=2` with `order` 2, 4 or 6. `jacobian` supports `order` 2 or 4.
- **Step:** by default h = ε^(1/(nu+order))·max(1, |x|), adjusted so that x + h is exactly representable. `glefit.merit` calls `jacobian(..., order=4)`, which gives h ≈ 7.4e-4·max(1, |x|).
- **Inputs:** `diff` needs a Python `float` (`np.float64` passes). `jacobian` needs `f` to return a 1-dimensional array, so callers flatten and reshape, as `gradhess_wrt_params` does.
- **Cost:** `jacobian` makes 2n calls at order 2 and 4n at order 4, plus one if `fshape` is not given or `value=True`.

## `special.py`

`sinc(x, nu=0)` is sin(x)/x or its nu'th derivative, without a factor of π, for real or complex x. It is copied unchanged from `rpmdgle/utils/special.py` in rpgle-model-dynamics. It sums spherical Bessel functions, so it is accurate to about 4e-15 relative even at small |x|, where closed forms of the derivatives cancel. For complex x, SciPy overflows when |Im x| > 700.9.

The other functions are the building blocks of the `gen2x2` kernel, K(t) = r² [expcoscosh − α t·expsincsinhc]:

```text
coscosh(G, t)          = cos(G t) if G < 0,           cosh(G t) if G ≥ 0
sincsinhc(G, t)        = sin(G|t|)/(G|t|) if G < 0,   sinh(G|t|)/(G|t|) if G ≥ 0
expcoscosh(G, t, λ)    = exp(−λt) · coscosh(G, t)
expsincsinhc(G, t, λ)  = exp(−λt) · sincsinhc(G, t)
Softplus(x)            = ln(1 + exp(σx))/σ, replaced by x or 0 where |σx| > threshold
Softmax(x₁, x₂)        = x₁ + Softplus(x₂ − x₁)  ≥ max(x₁, x₂)
Softabs(x)             = −x + Softplus(2x)       ≥ |x|, and Softabs(0) = ln 2/σ
```

- **Derivatives:** all functions take `nu` = 0, 1 or 2. The first four also take `wrt="time"` or `wrt="Gamma"`. `Softmax` returns shape (2, M) for `nu=1` and (2, 2, M) for `nu=2`, without the last axis for scalar input.
- **Stability:** for G ≥ 0, the `exp*` functions use exp(−(λ ∓ G)t) instead of multiplying exp(−λt) by cosh or sinh, so they assume λ > max(G, 0) and t ≥ 0. `sincsinhc` uses a series for |Gt| < 0.01.
- **The kink at G = 0:** the switch from cos to cosh makes the second G-derivative change sign at G = 0. For `expcoscosh` at t = 2 and λ = 2, it is −0.073 just below 0 and +0.073 just above. Finite differences across G = 0 are therefore inaccurate.
- **Types:** `Softplus`, `Softmax` and `Softabs` require `sigma` and `threshold` to be Python numbers. `np.float32` raises `TypeError`.

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

Checked at commit `ffd03fd`. Everything under Wrong results and Surprising limits was reproduced by running the code. Apart from the items below, `diff`, `jacobian`, and all `special.py` derivatives with respect to t, and the first derivatives with respect to G, matched SymPy or finite differences.

### Wrong results

- **`expsincsinhc(G, t, λ, nu=2, wrt="Gamma")` is wrong for G·t ≥ 0.01.** At G = 1.1, t = 0.3 and λ = 2 it gave 0.569, while SymPy gives 0.0170. Two terms of the hyperbolic branch are wrong. With N = E₋ − E₊ and E∓ = exp(−(λ ∓ G)t), the correct result is t·N/(2G) − (E₋ + E₊)/G² + N/(tG³). The code has −t·N/(2G) and −N/G² for the first two terms. Nothing calls it today, and the fix agreed for EMB-2 in `ISSUES.md` removes the remaining uses of `expcoscosh` and `expsincsinhc` from `src/`.
- **`max_step` crashes `diff` and `jacobian`.** `np.clip(h, a_max=max_step)` raises `TypeError` for the missing `a_min` in NumPy 2.4.2. Fix: `np.minimum(h, max_step)`.
- **Error paths in `numderiv.py` raise the wrong error.** `raise e(...)` calls a caught exception object, so an unsupported `nu` or `order`, or a wrongly shaped `h` in `jacobian`, raises `TypeError: 'KeyError' object is not callable`. Fix: `raise KeyError(...) from e`, or `ValueError`.
- **Integer times break `expcoscosh` and `expsincsinhc`.** Both create their output with `np.empty_like(t)`. `expsincsinhc(1.0, np.array([0, 1, 2]), 2.0)` returned `[1, 0, 0]`, and `expcoscosh(..., nu=1)` raised `UFuncTypeError`. Fix: convert `t` to float at the top.

### Surprising limits

- **`Softplus` jumps at its threshold.** The value and slope jump by about e^(−threshold): 2e-9 at the default of 20, but 4.5e-5 at the threshold of 10 that the tutorial uses for `gen2x2`.
- **`WAM_discretization` needs a grid that starts at 0 and a non-negative Λ.** It integrates from `omega[0]`, so a Debye grid starting at 0.5 gave a weight of 0.437 instead of 0.506. A region of negative Λ gave meaningless frequencies with no error: five between 0.05 and 0.42, then 11.4, 13.8 and 17.4.
- **Harmless warnings.** `sincsinhc(G < 0, t=0, nu=2)` warns "divide by zero", because `np.where` evaluates every branch. The result is correct.
- **`symmat_inv_vec` raises `LinAlgError`** for a matrix that is not positive definite, such as an indefinite Hessian.

### Missing features

- `numderiv` has no mixed second derivatives and no `nu=1, order=6` stencil.
- `units.py` is not connected to the config or the data, so all inputs must already share one unit system.

### Test gaps

- `tests/test_utils/test_special.py` tests only `sinc`, against SymPy.
- `legacy_tests/test_utils/test_special.py` tests derivatives with respect to t only. Nothing tests `wrt="Gamma"`, which is how the `expsincsinhc` bug went unnoticed. `tests/test_embedding/test_gen2x2.py` covers the first G-derivatives indirectly.
- `legacy_tests/test_utils/test_numderiv.py` does not test `max_step`, the error paths or integer inputs.
- `bath.py`, `linalg.py`, `rp.py` and `units.py` have no tests.

### Outdated text in the code

- `coscosh` says "cos(F*t)", and says that `nu` > 0 differentiates with respect to t only.
- `expcoscosh` lists `wrt="lamda"`, which raises `ValueError`.
- `WAM_discretization` documents `Lambda` as a callable and does not document `omega`, `wmax` or `options`.
- The `jacobian` docstring gives the default step as ε^(1/3), which is true only for `order=2`.
- The `units.py` docstring contains a stray `"""` and lists five of the seven unit systems.

## Adding features

- **Special functions:** follow the `nu`/`wrt` interface, handle G < 0 and G ≥ 0 with masks, and test both branches and the G = 0 boundary against SymPy.
- **Finite differences:** add stencils to `STENCILS`. Keys are the derivative order, then the accuracy order.
- **Discretization schemes:** add them to `bath.py`, taking Λ on a grid and returning values that `from_harmonic` accepts.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_utils
```

New or changed code should have tests that:

- compare `special.py` derivatives, with respect to both t and G, against SymPy on both sides of G = 0 and on both sides of |Gt| = 0.01
- pass integer and scalar inputs
- check `numderiv` against known derivatives, including with `h`, `rel_step` and `max_step`
- check `WAM_discretization` against the closed form for a Debye spectrum, where C(ω) = η ωc arctan(ω/ωc)
