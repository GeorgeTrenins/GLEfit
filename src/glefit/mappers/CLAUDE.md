# CLAUDE.md: `glefit.mappers`

This module defines the mappers that keep embedding parameters within their allowed ranges during a fit.

## Background

GLEFit fits a Markovian embedding of a generalized Langevin equation by adjusting the parameters of an `embedder`. Each embedder has three sets of parameters:

| Set | Role |
| --- | --- |
| conventional | Config files and trajectory output |
| primitive (`p`) | Builds the drift matrix. One per degree of freedom (`ndof`) |
| mapped (`x`) | What the optimizer changes. Has no limits on its values |

Conventional and primitive parameters are the same except for the `gen2x2` embedder, whose primitive parameters are [r, γ, Ω, φ].

Some primitive parameters must satisfy inequalities. For example, decay rates must be positive for the embedding to be stable. A mapper links one primitive parameter p to one unlimited variable x by a smooth one-to-one function, so that every real x gives an allowed p. Each embedder holds one mapper per primitive parameter. Because each xᵢ depends only on pᵢ, the chain rule needs only the diagonal terms:

```text
∂f/∂xᵢ      = pᵢ' ∂f/∂pᵢ
∂²f/∂xᵢ∂xⱼ  = pᵢ' pⱼ' ∂²f/∂pᵢ∂pⱼ + δᵢⱼ pᵢ'' ∂f/∂pᵢ      where p' = dp/dx and p'' = d²p/dx²
```

## Contents

| Class | File | Allowed p | p = `inverse(x)` | x = `forward(p)` | dp/dx = `grad(x)` |
| --- | --- | --- | --- | --- | --- |
| `IdentityMapper()` | `_base.py` | any | x | p | 1 |
| `LowerBoundMapper(a=0.0)` | `intervals.py` | a < p | a + eˣ | ln(p − a) | eˣ |
| `UpperBoundMapper(b=0.0)` | `intervals.py` | p < b | b − eˣ | ln(b − p) | −eˣ |
| `IntervalMapper(a=0.0, b=1.0)` | `intervals.py` | a < p < b | a + (b − a)/(1 + e⁻ˣ) | ln(p − a) − ln(b − p) | (b − a)/(4 cosh²(x/2)) |

`_base.py` also holds the abstract `BaseMapper`, which requires `forward`, `inverse`, `grad` and `hess`. `hess(x)` returns d²p/dx². `__init__.py` exports all five classes. There is no registry.

Default mappers, set by each embedder's constructor:

| Embedder | Primitive parameters | Mappers |
| --- | --- | --- |
| `prony` | θ, γ | `LowerBoundMapper()` for both |
| `pronycos` | θ, γ, ω | `LowerBoundMapper()` for all three |
| `gen2x2` | r, γ, Ω, φ | `LowerBoundMapper()` for γ, `IdentityMapper()` for r, Ω and φ |
| `multi` | its components' | its components' |

No embedder uses `UpperBoundMapper` or `IntervalMapper` by default. To use them, pass `mappers` to the constructor in Python:

```python
from glefit.mappers import LowerBoundMapper, IntervalMapper
from glefit.embedding import PronyEmbedder
emb = PronyEmbedder(1.0, 2.0, mappers=[LowerBoundMapper(), IntervalMapper(0.1, 10.0)])
emb.x               # [0.0, -1.4376]
emb.x = [0.0, 50.0]
emb.primitive_params  # [1.0, 10.0]: γ is exactly at its bound, see Surprising limits
```

## How embedders use mappers

`BaseEmbedder` in `glefit/embedding/_base.py` applies the mappers one element at a time:

| Mapper method | Embedder method | Called by |
| --- | --- | --- |
| `forward` | `_forward_map` | The `primitive_params` and `conventional_params` setters |
| `inverse` | `_inverse_map` | The `x` setter, `drift_matrix_x_grad`, and merit functions evaluating trial `x` |
| `grad` | `jac_px`, `grad_param_to_x` | `drift_matrix_x_grad` (fits), `kernel()` and `spectrum()` with `mapped=True` |
| `hess` | `hess_px`, `hess_param_to_x` | Only `kernel()` and `spectrum()` with `nu=2, mapped=True` |

Fits use only `forward`, `inverse` and `grad`. The fitting Hessian is a finite difference of the gradient in x, so a wrong `hess` does not change a fit, but it makes the closed-form second derivatives disagree with it.

`MultiEmbedder` passes each slice of x to its components' mappers. Its own `mappers` argument is checked for length and then ignored.

## Conventions

- **Direction:** `forward` goes from p to x and removes the limit. `inverse` goes from x to p. Docstrings define each map by its inverse, for example "p = exp(x) + a".
- **Arguments:** `grad` and `hess` take x, not p, and return derivatives of `inverse`.
- **p is primitive:** for `gen2x2`, mappers act on r, γ, Ω and φ, not on θ, δ or ζ. Limits that couple parameters, such as the stability condition γ ≥ |δ|, must be built into the primitive parameters instead.
- **Errors:** `forward` raises `ValueError` when p is outside its range or on a bound, which would map to x = ±inf. It is the only place where limits are checked.
- **Scalars only:** each method takes and returns one number. `forward` raises on arrays. `IdentityMapper.grad` returns `1.0` whatever the input, and `IdentityMapper.forward` and `inverse` return 0-dimensional arrays.
- **No state:** a mapper stores only its bounds, so one instance can serve several parameters. `pronycos` uses `3*[LowerBoundMapper(),]`.
- **No mappers:** `_mappers = None` on an embedder means no limits. It is read at every call, so changing it after construction leaves the stored `x` out of date until a setter runs.
- **Units:** with the bounded mappers, changing the units of p and its bounds shifts x by at most a constant. With `IdentityMapper`, x has the units of p. This matters because the fitting Hessian's finite-difference step is about 7e-4·max(1, |x|).
- **Outside use:** `analysis/plot_objective.py` sets `_mappers = None` to plot parameters without limits.
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Known gaps

Checked at commit `10dc1e7`. Everything under Wrong results and Surprising limits was reproduced by running the code.

### Wrong results

- **p exactly at a bound is accepted.** `forward` checks `diff < 0`, so p = a or p = b returns x = ±inf with only a NumPy warning, and NaN passes as NaN. A starting value of 0 for a parameter with a default mapper therefore gives x = -inf, and the fitting Hessian is then not finite. Fix: `if not diff > 0: raise ValueError(...)`, which also catches NaN.
- **`IntervalMapper(a, a)` is accepted.** `forward(a)` returns NaN, and `inverse` returns a with a gradient of 0 for every x. Fix: raise in `__init__` when a == b.
- **`plot_objective.setup_from_config` raises `UnboundLocalError` for any embedder except `multi`.** Its `else` branch sets `emb._mappers` instead of `embedder._mappers`. Reproduced with `tests/test_optimize/fixtures/gen2x2_embedder.yaml`.

### Surprising limits

- **Bounds hold only up to rounding.** `IntervalMapper().inverse(40.0)` and `LowerBoundMapper(1.0).inverse(-40.0)` both return the bound exactly, and `LowerBoundMapper().inverse(-800.0)` returns 0. Passing such a p back through `forward`, for example through `conventional_params`, raises `ValueError` (EMB-10). `LowerBoundMapper.inverse` and `UpperBoundMapper.inverse` overflow to ±inf for x > 709.
- **Gradients vanish near bounds.** dp/dx falls like e^(−|x|): about 5e-5 at |x| = 10 and 2e-9 at |x| = 20. A parameter pushed against its bound gets almost no gradient.
- **Strict bound types.** Bounds must be Python `int` or `float` (`np.float64` and `bool` pass). `None`, `np.float32` and `np.int64` raise `ValueError` rather than `TypeError`.
- **`IntervalMapper` sorts its bounds.** `IntervalMapper(1, 0)` silently becomes `IntervalMapper(0, 1)`.

### Missing features

- Mappers cannot be set from a config file. There is no registry and no `from_dict`.
- Mappers act on one parameter each. There is no way to limit a combination of parameters.
- There is no vectorized interface: `BaseEmbedder` loops over parameters in Python.

### Test gaps

- `tests/test_mappers.py` compares `inverse`, `grad` and `hess` of the bounded mappers with SymPy, at bounds -2.5, 3 and (-1, 4) and x = -3, 0, 5, and checks that `forward` rejects p on a bound. Nothing tests round trips, reversed bounds, NaN, or `IdentityMapper`. A one-off check found round-trip errors below 3e-15 and correct derivatives with bounds (4, -1).
- The Hessian failure message in `legacy_tests/test_mappers/test_intervals.py` prints `grad` instead of `hess`.

### Outdated text in the code

- The `_base.py` header and the test docstring call p the conventional parameter. It is the primitive parameter.
- The bounds are typed `Optional[float]`, but `None` is rejected.
- `intervals.py` ends with a leftover `__main__` block that builds an unused `IntervalMapper`.

## Adding features

### A new mapper

1. Subclass `BaseMapper` in `intervals.py`, or in a new file for a different kind of limit.
2. Implement `forward(p)`, `inverse(x)`, `grad(x)` and `hess(x)` for single numbers. `grad` and `hess` are the first and second derivatives of `inverse`.
3. In `forward`, raise `ValueError` for p outside the open range, using `not diff > 0` so that the bound and NaN are rejected.
4. Store only constants on the instance, since one instance may be shared.
5. Export the class from `__init__.py` and add it, with its p(x) in SymPy, to `MAPS` in `tests/test_mappers.py`.
6. To make it an embedder's default, change that embedder's `kwargs.setdefault("mappers", [...])`. There must be exactly `ndof` mappers, in primitive-parameter order.

### Other changes

- **Mappers in config files:** add a name-to-class registry here, and build mapper objects in each embedder's `from_dict`. `MultiEmbedder` would need to pass them to its components, since it ignores its own.
- **Vectorized mappers:** change `_forward_map`, `_inverse_map`, `jac_px` and `hess_px` in `glefit/embedding/_base.py` together with the mappers, and make `forward`'s check work on arrays.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_mappers tests/test_embedding tests/test_merit
```

A new or changed mapper should have tests that:

- compare `inverse`, `grad` and `hess` with p(x) and its derivatives in SymPy, as `tests/test_mappers.py` does
- check that `inverse(forward(p))` returns p and `forward(inverse(x))` returns x
- use bounds other than the defaults, including negative ones
- check that `forward` raises at, beyond and on NaN, and that `inverse` stays within the bounds for large |x|
