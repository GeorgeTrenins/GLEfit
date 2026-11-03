# CLAUDE.md: `glefit.spectral`

This module defines model spectral densities and their memory kernels. Nothing in the repository imports it, and the fitting pipeline does not use it. It is for making target data and reference values by hand.

## Background

GLEFit fits a Markovian embedding so that it reproduces the memory kernel K(t) of a generalized Langevin equation, or its cosine transform, the spectrum Λ(ω). This module uses the same conventions as `glefit.embedding` and `glefit.merit`:

```text
Λ(ω) = ∫₀^∞ K(t) cos(ωt) dt
K(t) = (2/π) ∫₀^∞ Λ(ω) cos(ωt) dω
J(ω) = ω Λ(ω)                          the spectral density
```

The `nm` methods transform these for a normal mode of frequency ωₙ ≥ 0:

```text
Jₙ(ω) = J(√(ω² − ωₙ²)) for |ω| > ωₙ, and 0 otherwise
Λₙ(ω) = Jₙ(ω) / |ω|
Kₙ(t) = (2/π) ∫_ωₙ^∞ Λₙ(ω) cos(ωt) dω
```

At ωₙ = 0 they reduce to J, Λ and K. For a free ring polymer with P beads at inverse temperature β, `glefit.utils.rp.nmfreq(beta, P, n)` gives the frequency of normal mode n, ωₙ = 2(P/βħ)|sin(πn/P)|.

## Contents

| Class | File | J(ω) | Λ(ω) | K(t) |
| --- | --- | --- | --- | --- |
| `Debye(eta, omega_cut)` | `debye.py` | η ωc² ω / (ω² + ωc²) | η ωc² / (ω² + ωc²) | η ωc exp(−ωc\|t\|) |
| `ExpOhmic(eta, omega_cut)` | `expohmic.py` | η \|ω\| exp(−\|ω\|/ωc) | η exp(−\|ω\|/ωc) | (2/π) η ωc / (1 + ωc²t²) |

η is the static friction Λ(0), so K integrates to 2η over all t. ωc is the cut-off frequency.

`_base.py` holds the abstract `BaseSpectralDensity`, which requires `J`, `Lambda` and `K` and provides `nmJ`, `nmLambda` and `nmK`. It also holds the decorator `check_nm_omega`, which checks ωₙ. `__init__.py` exports `Debye` and `ExpOhmic`. There is no registry and no config `type`.

```python
from glefit.spectral import ExpOhmic
bath = ExpOhmic(eta=1.0, omega_cut=1.0)
bath.Lambda(omega)            # Λ(ω), closed form
bath.nmLambda(10.0, omega)    # Λₙ(ω) for ωₙ = 10, closed form
bath.nmK(10.0, time)          # Kₙ(t), numerical integral
```

## Relation to the rest of GLEFit

- **Reference values:** `Debye(η, ωc)` is exactly `PronyEmbedder(theta=√(η ωc), gamma=ωc)`. Their K and Λ agree to 2e-16. `ExpOhmic` has no exact finite embedding, so it is a natural fitting target.
- **Target data:** the config reads targets only from files or inline values, so evaluate a model on a grid and save it. `tests/test_optimize/fixtures/expohmic_nm_spectrum.csv` matches `ExpOhmic(1, 1).nmLambda(10, ω)` divided by `ExpOhmic(1, 1).nmK(10, 0)` = 1.1516e-02 to 5e-8. Its header records ωₙ = 10 and that K₀. The script that wrote it is not in the repository.
- **Initial guesses:** `utils/bath.py::WAM_discretization(omega, Lambda, nbath)` takes Λ sampled on a grid, so pass `bath.Lambda(omega)` or `bath.nmLambda(wn, omega)`, not the model object.

## Conventions

- **Names:** `Lambda` is Λ and `J` is ω·Λ, as in the embedders' `spectrum()` and `spectral_density()`. `nm_omega` is ωₙ and always comes first.
- **Arguments:** `J`, `Lambda` and `K` are NumPy expressions and accept scalars or arrays. ωₙ must be a Python `int` or `float` (`np.float64` passes) and must not be negative.
- **Closed form or numerical:** `nmJ` and `nmLambda` are closed forms built from `J`. `nmK` integrates `Lambda` with `scipy.integrate.quad`, once per time point, using the Fourier weight for t ≠ 0. Extra keyword arguments go to `quad`. 201 time points take about 0.06 s.
- **Symmetry:** Λ, K and their `nm` versions are even in ω and t. J is not consistent: see Wrong results.
- **Return types:** for ωₙ > 0, `nmJ` and `nmLambda` return a 0-dimensional array for a scalar ω. `nmK` returns a float for a Python scalar t, and an array otherwise, including for a 0-dimensional array.
- **Constructors:** they store `eta` and `omega_cut` without checks, and ignore extra arguments.
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Known gaps

Checked at commit `10dc1e7`. Everything under Wrong results and Surprising limits was reproduced by running the code. K and Λ of both models, and `nmK` against a direct integral of `nmLambda` at ωₙ = 3, were confirmed correct.

### Wrong results

- **The sign of J at negative ω differs between models.** `Debye.J` is odd and `ExpOhmic.J` is even, because it uses |ω|. So `ExpOhmic.Lambda(ω)` is not `J(ω)/ω` for ω < 0. `nmJ` is even for ωₙ > 0 but returns `J` at ωₙ = 0, so for `Debye(0.7, 2.5)`, `nmJ(0, -2)` gives -0.85 while `nmJ(1e-12, -2)` gives +0.85. Λ and K are not affected. Fix: choose one convention, usually an odd J = ωΛ, and use it in `nmJ` as well.
- **Integer frequency arrays give zeros.** For ωₙ > 0, `nmJ` and `nmLambda` create their output with `np.zeros_like(omega)`, so `ExpOhmic(0.7, 2.5).nmJ(1.0, np.array([0, 2, 3]))` returns `[0, 0, 0]`. Fix: `np.zeros(np.shape(omega))`.

### Surprising limits

- **Strict ωₙ type check.** `np.float32` and `np.int64` raise `TypeError`.
- **No parameter checks.** Negative η or ωc are accepted. `Debye(1.0, 0.0).Lambda(0.0)` raises `ZeroDivisionError`. A misspelled keyword argument is silently ignored.
- **`nmK` uses `quad`'s default tolerances.** At ωₙ = 0, `Debye(0.7, 2.5).nmK(0, 10)` gives 7.1e-11 where K(10) = 2.4e-11. Pass `epsabs` for small values at long times.

### Missing features

- Only two models exist.
- `nmK` has no closed form for either model.
- There are no derivatives with respect to model parameters.
- No code writes model spectra or kernels to data files for fitting.

### Test gaps

- There are no tests: `tests/` has no `test_spectral`.

### Outdated text in the code

- The `ExpOhmic` docstring gives K(t) = 2ηωc/π · (1 + ωc²t²). The code correctly divides by (1 + ωc²t²).
- The `BaseSpectralDensity` docstring says Λ = J/ω, which is false for `ExpOhmic` at ω < 0.
- The `check_nm_omega` error says ωₙ must be positive, but 0 is accepted.
- `nmJ`, `nmLambda` and `nmK` have no docstrings.

## Adding features

### A new model

1. Add `<name>.py` and subclass `BaseSpectralDensity`.
2. Store the parameters in `__init__`, and implement `J`, `Lambda` and `K` in closed form. Use |ω| and |t| so that Λ and K are even, and follow the J sign convention chosen in the fix above.
3. The `nm` methods are inherited. `nmJ` and `nmLambda` call `J`, and `nmK` calls `Lambda`.
4. Export the class from `__init__.py`. No registration is needed.

### Other changes

- **Closed-form `nmK`:** override it in the subclass. Keep the signature `(nm_omega, t, **kwargs)` and the `@check_nm_omega` decorator.
- **Models as config targets:** add a data `source` beside `external` and `inline` in `glefit/config/data_io.py` that builds a model and evaluates Λ or K on a grid.

## Testing

`glefit` is not installed. Put new tests in `tests/test_spectral/`, and from the repository root run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_spectral
```

A new or changed model should have tests that:

- compare `K(t)` with `(2/π)·quad(Lambda, 0, inf, weight='cos', wvar=t)`, including t = 0
- check J = ωΛ and the symmetry of J, Λ and K at negative ω and t
- compare `nmJ`, `nmLambda` and `nmK` at ωₙ = 0 with `J`, `Lambda` and `K`, and `nmK` at ωₙ > 0 with a direct integral of `nmLambda`
- pass scalars and integer arrays
- for a model with an exact embedding, compare with it, as `Debye` does with `PronyEmbedder`
