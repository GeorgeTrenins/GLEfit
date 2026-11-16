# CLAUDE.md: `glefit.spectral`

This module defines model spectral densities and their memory kernels. The fitting pipeline does not use it: `utils/make_target.py` writes target data from its models, and tests use them as reference values. Its open issues, accepted limits and missing features are in `ISSUES.md` at the repository root, as `SPC-n`.

## Background

GLEFit fits a Markovian embedding so that it reproduces the memory kernel K(t) of a generalized Langevin equation, or its cosine transform, the spectrum Λ(ω). This module uses the same conventions as `glefit.embedding` and `glefit.merit`:

```text
Λ(ω) = ∫₀^∞ K(t) cos(ωt) dt
K(t) = (2/π) ∫₀^∞ Λ(ω) cos(ωt) dω
```

The `nm` methods transform these for a normal mode of frequency ωₙ ≥ 0, with x = 1 − ωₙ²/ω² and θ the Heaviside step function:

```text
Λₙ(ω) = θ(x) √x Λ(ω√x)                 0 for |ω| ≤ ωₙ
Kₙ(t) = (2/π) ∫_ωₙ^∞ Λₙ(ω) cos(ωt) dω
      = (2/π) ∫₀^∞ Λ(u) u²/(u² + ωₙ²) cos(t√(u² + ωₙ²)) du
```

At ωₙ = 0 they reduce to Λ and K. For a free ring polymer with P beads at inverse temperature β, normal mode n has the frequency ωₙ = 2(P/βħ)|sin(πn/P)|.

## Contents

| Class | File | Λ(ω) | K(t) |
| --- | --- | --- | --- |
| `Debye(eta, omega_cut)` | `debye.py` | η ωc² / (ω² + ωc²) | η ωc exp(−ωc\|t\|) |
| `ExpOhmic(eta, omega_cut)` | `expohmic.py` | η exp(−\|ω\|/ωc) | (2/π) η ωc / (1 + ωc²t²) |

η is the static friction Λ(0), so K integrates to 2η over all t. ωc is the cut-off frequency.

`_base.py` holds the abstract `BaseSpectralDensity`, which requires `Lambda` and `K` and provides `nmLambda` and `nmK`. It also holds `check_parameters`, which the constructors call, and the decorator `check_nm_omega`, which checks ωₙ. `__init__.py` exports `Debye`, `ExpOhmic` and the registry `SPECTRAL_MAP`, whose keys are the `model.type` of `make_target.py`. Models are not a `type` in a fit config (CFG-7).

```python
from glefit.spectral import ExpOhmic
bath = ExpOhmic(eta=1.0, omega_cut=1.0)
bath.Lambda(omega)            # Λ(ω), closed form
bath.nmLambda(10.0, omega)    # Λₙ(ω) for ωₙ = 10, closed form
bath.nmK(10.0, time)          # Kₙ(t), numerical integral
```

## Relation to the rest of GLEFit

- **Reference values:** `Debye(η, ωc)` is exactly `PronyEmbedder(theta=√(η ωc), gamma=ωc)`. Their K and Λ agree to 2e-16. `ExpOhmic` has no exact finite embedding, so it is a natural fitting target.
- **Target data:** the config reads targets only from files or inline values, so evaluate a model on a grid and save it. `legacy_tests/test_optimize/fixtures/expohmic_nm_spectrum.csv` matches `ExpOhmic(1, 1).nmLambda(10, ω)` divided by `ExpOhmic(1, 1).nmK(10, 0)` = 1.1516e-02 to 5e-8. Its header records ωₙ = 10 and that K₀. The script that wrote it is not in the repository. `utils/make_target.py` writes such files, but does not divide by Kₙ(0).
- **Initial guesses:** `utils/bath.py::WAM_discretization(omega, Lambda, nbath)` takes Λ sampled on a grid, so pass `bath.Lambda(omega)` or `bath.nmLambda(wn, omega)`, not the model object.

## Conventions

- **Names:** `Lambda` is Λ, as in the embedders' `spectrum()`. `nm_omega` is ωₙ and always comes first.
- **Arguments:** `Lambda` and `K` are NumPy expressions and accept scalars or arrays. ωₙ must be a real number, and ωₙ ≥ 0: other types raise `TypeError`, and negative values and NaN raise `ValueError`. `check_nm_omega` passes ωₙ on as a Python float.
- **Closed form or numerical:** `nmLambda` is a closed form built from `Lambda`, with ω√x computed as √((|ω| − ωₙ)(|ω| + ωₙ)), which stays accurate near ωₙ. `nmK` integrates `Lambda` with `scipy.integrate.quad`, once per time point, using the Fourier weight for t ≠ 0. Extra keyword arguments go to `quad`. 201 time points take about 0.06 s.
- **Symmetry:** Λ, K and their `nm` versions are even in ω and t.
- **Return types:** for ωₙ > 0, `nmLambda` returns a 0-dimensional array for a scalar ω. `nmK` returns a float for a Python scalar t, and an array otherwise, including for a 0-dimensional array.
- **Constructors:** they take only `eta` and `omega_cut`, and `check_parameters` in `_base.py` raises `ValueError` unless η ≥ 0 and ωc > 0. They store the values as given.
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Adding features

### A new model

1. Add `<name>.py` and subclass `BaseSpectralDensity`.
2. Check and store the parameters in `__init__`, and implement `Lambda` and `K` in closed form. Use |ω| and |t| so that both are even.
3. The `nm` methods are inherited, and call `Lambda`.
4. Export the class from `__init__.py` and add it to `SPECTRAL_MAP`. `make_target.py` then builds it from `model.parameters`, passed to the constructor as keyword arguments, so its parameters may differ from η and ωc.

### Other changes

- **Closed-form `nmK`:** override it in the subclass. Keep the signature `(nm_omega, t, **kwargs)` and the `@check_nm_omega` decorator.
- **Models as config targets:** add a data `source` beside `external` and `inline` in `glefit/config/data_io.py` that builds a model and evaluates Λ or K on a grid.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_spectral.py
```

`tests/test_spectral.py` compares both models with values from mpmath at 30 digits: Λ with its closed form, K with the cosine transform of Λ, `nmLambda` with θ(x)√x Λ(ω√x), where x = 1 − ωₙ²/ω², and `nmK` with (2/π)∫₀^∞ Λ(u) u²/(u² + ωₙ²) cos(t√(u² + ωₙ²)) du, which shares no code with it. It also checks that Λ, K and their `nm` versions are even, that the `nm` versions reduce to Λ and K at ωₙ = 0, that `nmK` passes keyword arguments to `quad`, that `Debye` matches `PronyEmbedder`, that ωₙ may be any real number ≥ 0, and that the constructors and the `nm` methods reject bad input.

A new or changed model should have tests that:

- compare K(t) with the cosine transform of Λ, including t = 0
- check the symmetry of Λ and K at negative ω and t
- compare `nmLambda` and `nmK` at ωₙ = 0 with `Lambda` and `K`, and at ωₙ > 0 with the references above
- pass scalars and integer arrays
- for a model with an exact embedding, compare with it, as `Debye` does with `PronyEmbedder`
