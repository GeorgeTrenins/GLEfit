# CLAUDE.md: `glefit.mappers`

Mappers keep the primitive parameters p of an embedder within their allowed ranges during a fit. This file lists conventions and pitfalls. Open issues are in `ISSUES.md` at the repository root, as `MAP-n`.

## Model

Each embedder holds one mapper per primitive parameter, so `ndof` in all (for gen2x2, one each for r, γ, Ω and φ). A mapper links p to a variable x that has no limits, by a smooth one-to-one map, and the optimizer moves x. Each xᵢ depends only on pᵢ, so the chain rule needs only diagonal terms: ∂f/∂xᵢ = pᵢ′ ∂f/∂pᵢ and ∂²f/∂xᵢ∂xⱼ = pᵢ′pⱼ′ ∂²f/∂pᵢ∂pⱼ + δᵢⱼ pᵢ″ ∂f/∂pᵢ, with p′ = dp/dx = `grad(x)` and p″ = d²p/dx² = `hess(x)`.

| Class | Allowed p | p = `inverse(x)` | x = `forward(p)` |
| --- | --- | --- | --- |
| `IdentityMapper()` | any | x | p |
| `LowerBoundMapper(a=0.0)` | a < p | a + eˣ | ln(p − a) |
| `UpperBoundMapper(b=0.0)` | p < b | b − eˣ | ln(b − p) |
| `IntervalMapper(a=0.0, b=1.0)` | a < p < b | a + (b − a)/(1 + e⁻ˣ) | ln(p − a) − ln(b − p) |

`IdentityMapper` and the abstract `BaseMapper` are in `_base.py`, the others in `intervals.py`. Every embedder uses `LowerBoundMapper()` for each primitive parameter by default, except for r and φ of gen2x2, which use `IdentityMapper()`. Configs cannot set mappers, so the others are for debugging only, passed as `mappers=[...]` in Python.

`BaseEmbedder._apply_mappers` calls the mappers one entry at a time. `forward` runs in the `primitive_params` and `conventional_params` setters. `inverse` runs in the `x` setter, in `drift_matrix_x_grad`, and for trial `x` in `glefit.merit`. `grad` runs in `drift_matrix_x_grad` and in `kernel()` and `spectrum()` with `mapped=True`, and `hess` only there with `nu=2`. Fits never call `hess`, because their Hessian is a finite difference of the gradient in x. So a wrong `hess` only makes the closed-form second derivatives disagree with the fit.

## Conventions

- **Direction:** `forward` goes from p to x, and `inverse` from x to p. Docstrings define each map by its inverse, as in "p = exp(x) + a". `grad` and `hess` take x, not p.
- **Errors:** limits are checked only in `forward`, which raises `ValueError` for p outside the open range. A p on a bound would give x = ±∞. Embedders put the class and parameter name in front of the message, as in "TwoAuxEmbedder parameter Ω: …", and `MultiEmbedder` adds the component, as in "MultiEmbedder.embedders[1]: …".
- **One parameter each:** a limit that couples parameters, such as γ ≥ |δ| for gen2x2, must be built into the primitive parameters.
- **Scalars only:** each method takes and returns one number, and the bounded mappers' `forward` raises on arrays. `IdentityMapper.forward` and `inverse` return 0-dimensional arrays, and its `grad` returns `1.0` whatever x is.
- **No state:** a mapper stores only its bounds, so one instance can serve several parameters, as in `pronycos`.
- **No mappers:** `_mappers = None` on an embedder means no limits, as `analysis/plot_objective.py` sets it. It is read at every call, so changing it after construction leaves the stored `x` out of date until a setter runs.
- **Units:** with a bounded mapper, changing the units of p and its bounds shifts x by at most a constant. With `IdentityMapper`, x has the units of p, which matters for the fitting Hessian's finite-difference step (MER-8).
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## New mapper

Subclass `BaseMapper` and implement `forward`, `inverse`, `grad` and `hess` for single numbers, and store only constants on the instance. `forward` must raise `ValueError` for any p outside the open range, including NaN. Export the class from `__init__.py`. To make it a default, change the embedder's `kwargs.setdefault("mappers", [...])`, with exactly `ndof` mappers in primitive-parameter order.

## Testing

Run `PYTHONPATH=src conda run -n tools python -m pytest tests/test_mappers.py` from the repository root. It compares `inverse`, `grad` and `hess` of the bounded mappers with SymPy at bounds other than the defaults, and checks that `forward` rejects p on a bound. Add a new mapper to `MAPS`, with its p(x) in SymPy. Nothing yet tests round trips, NaN, reversed bounds or `IdentityMapper`. `legacy_tests/test_mappers/test_intervals.py` fails to import, and `tests/test_mappers.py` replaces it.
