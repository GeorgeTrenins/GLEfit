# CLAUDE.md: `glefit.embedding`

This module defines the embedders that GLEFit fits to data.

## Background

This defines Kantorovich-style Markovian embeddings for generalized Langevin dynamics (GLD) described by the memory kernel K(t), or by its cosine transform, the spectrum Λ(ω). The spectral density is ω·Λ(ω). An embedding maps the GLD onto Markovian stochastic dynamics in an extended space with n extra auxiliary variables. The embedding is defined by the (n+1)×(n+1) drift matrix:

```text
Ap = [[ 0, θᵀ],      K(t) = θᵀ exp(-tA) θ
      [-θ, A ]]      Λ(ω) = θᵀ A (A² + ω²)⁻¹ θ
```
The embedding is stable only if the symmetric part of A is positive definite, so that the auxiliary variables decay. An `embedder` builds `Ap` from a small set of parameters. The `optimizer` adjusts those parameters until K or Λ matches target data.

## Contents

| Config `type` | Class | n | θ | A | K(t) |
| --- | --- | --- | --- | --- | --- |
| `prony` | `PronyEmbedder` | 1 | [θ] | [γ] | θ² exp(-γt) |
| `pronycos` | `PronyCosineEmbedder` | 2 | [θ, 0] | [[γ, ω], [-ω, γ]] | θ² exp(-γt) cos(ωt) |
| `gen2x2` | `TwoAuxEmbedder` | 2 | [θ₁, θ₂] | [[γ+δ, Ω], [-Ω, γ-δ]] | see the `gen2x2` section |
| `multi` | `MultiEmbedder` | sum | joined end to end | component blocks on the diagonal | sum of components |

Each class has its own file: `prony.py`, `pronycos.py`, `gen2x2.py` and `multi.py`. `_base.py` holds the abstract `BaseEmbedder`. `__init__.py` holds `EMBEDDER_MAP`, which links each config `type` to its class.

```python
from glefit.embedding import EMBEDDER_MAP
emb = EMBEDDER_MAP["pronycos"].from_dict({"theta": 1.0, "gamma": 0.5, "omega": 2.0})
emb.kernel(time)                        # K(t)
emb.spectrum(freq, nu=1, mapped=True)   # derivative of Λ with respect to x
```

```yaml
embedder:
  type: multi
  parameters:
    embedders:
      - {type: prony, parameters: {theta: 1.0, gamma: 0.5}}
      - {type: gen2x2, parameters: {theta: [1.0, 2.0], gamma: 1.5, delta: 0.25, Omega: 3.0, sigma: 10.0}}
```

## Three sets of parameters

| Set | Length | Role | Convert with |
| --- | --- | --- | --- |
| conventional | `nparam` | Config files and trajectory output | `to_primitive`, `to_conventional` |
| primitive (`p`) | `ndof` | Input to `compute_drift_matrix` and `drift_matrix_param_grad` | One mapper per parameter |
| mapped (`x`) | `ndof` | What the optimizer changes. Has no limits on its values | One mapper per parameter |

Conventional and primitive parameters are identical except for `gen2x2`.

Mappers come from `glefit.mappers` and enforce limits on `p`. `forward` converts p to x, `inverse` converts x to p, and `grad` and `hess` are dp/dx and d²p/dx². `LowerBoundMapper` uses p = exp(x) to keep p positive. `IdentityMapper` applies no limit. By default every parameter is kept positive, except α and Γ in `gen2x2`, which are unlimited.

Always set values through the `conventional_params`, `primitive_params` and `x` properties, because each one updates the others. Their getters return copies.

## `gen2x2` parameters

Different values of (θ₁, θ₂, γ, δ, Ω) can give the same kernel, so fits use four primitive parameters, [r, α, λ, Γ], that remove this redundancy:

```text
r² = θ₁² + θ₂²      α = δ(θ₁² - θ₂²)/r²      Γ = sign(δ²-Ω²)·sqrt(|δ²-Ω²|)
γ = λ + Softmax(Γ, Softabs(α))
K(t) = r² exp(-γt) [C - αt·S]
C = cosh(Γt) and S = sinh(Γt)/(Γt) for Γ ≥ 0;  C = cos(Γt) and S = sin(Γt)/(Γt) for Γ < 0
```

`Softmax` and `Softabs` are smooth versions of `max` and `abs` from `glefit.utils.special`. They are never smaller than the exact functions, so λ > 0 guarantees a stable embedding. Two consequences:

- **Values do not round-trip.** `conventional_params` returns one standard form with the same kernel, not the input. For example, [1, 2, 1.5, 0.25, 3] reads back as [0, 2.236, 1.5, 0.15, 2.993]. Trajectory files show this form.
- **`sigma` depends on your units.** It sets how sharp the smooth functions are (default 5) and has units of 1/rate. It raises the smallest allowed γ by up to ln(3)/σ, which is 0.22 at the default, so a low `sigma` also prevents narrow spectral peaks. The tutorial uses 50 for energies in eV.

## Two separate calculation paths

| Path | Used by | Depends on |
| --- | --- | --- |
| Fitting | Optimizers and merit functions | `compute_drift_matrix`, `drift_matrix_param_grad` |
| Closed-form | `kernel()`, `spectrum()`, `spectral_density()` | `kernel_*`, `spectrum_*` |

On the fitting path, `glefit.merit` computes K, Λ and their gradients from `Ap`. A bug on one path does not show up on the other, so compare them when testing.

The two fitting methods must use only their argument, never stored state, because `glefit.merit` calls them with trial values.

## Conventions

- **Shapes:** the parameter axis comes first. `compute_drift_matrix` returns (n+1, n+1), `drift_matrix_param_grad` returns (ndof, n+1, n+1), `*_grad` returns (ndof, ngrid), and `*_hess` (second derivatives) returns (ndof, ndof, ngrid).
- **Derivatives:** `kernel()` and `spectrum()` take `nu` = 0, 1 or 2. They differentiate with respect to `p`, or `x` if `mapped=True`.
- **Names that mislead:** `emb.A` is the full matrix `Ap`, not the block A, but `from_matrix(theta, A)` takes only the 2×2 block. `emb.grad_A` is with respect to `x`, while `drift_matrix_param_grad` is with respect to `p`.
- **Even functions:** closed forms use |t| and |ω|.
- **Constructors:** `BaseEmbedder.__init__` calls `__len__` and `ndof`. Subclasses must set what those need, check argument types, call `kwargs.setdefault("mappers", [...])`, call `super().__init__`, then set the parameters.
- **`from_dict`:** copies its input, pops the required keys and passes the rest to the constructor as keyword arguments.
- **`from_harmonic`:** returns a parameter dict for `from_dict`, not an embedder. Available on `pronycos` and `gen2x2`.
- **`MultiEmbedder`:** components hold the parameters. Slice conventional arrays with `nparam` and primitive arrays with `ndof`, because they differ for `gen2x2`. Set values on the `MultiEmbedder`, not on a component.
- **Outside use:** `analysis/plot_objective.py` uses the private attributes `_embs` and `_mappers`.
- **Style:** use `lamda`, since `lambda` is a Python keyword. Start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Known gaps

Checked at commit `85e5b71`. Everything under Wrong results and Surprising limits was reproduced by running the code.

### Wrong results

- **Wrong fit gradient at α = 0 with Γ < 0.** `jac_conventional` sets `jac[3,1] = np.sign(alpha)`, which is 0 at α = 0, so the fit gradient for α is 0 when it should not be, and the Hessian entries linking α to the other parameters come out at half their true value. Every `from_harmonic` start is at this point, as is any input with θ₁ = θ₂ and |δ| < |Ω|. An `EF` fit still reached its target in one trial, so this can go unnoticed. Fix: `jac[3,1] = 1.0 if alpha > 0 else -1.0`.
- **`gen2x2` kernels with integer times.** `kernel(t, nu=1)` silently truncates to integers and can return all zeros, while `nu=0` raises `UFuncTypeError`. The cause is `np.empty_like(t)` in `expcoscosh` and `expsincsinhc`. Pass floats, or convert `t` at the top of `kernel_*`.
- **Spectrum fits at Γ = 0 with α ≠ 0 (|δ| = |Ω|).** A cannot be diagonalized here, and `MemorySpectrum` relies on that. Its value was wrong by 0.5 in one check, and its gradient was also wrong. The closed form and `MemoryKernel` are correct. The error fades quickly away from Γ = 0 (about 2e-9 at Γ = 1e-8). The fix belongs in `merit/properties.py`.
- **Infinite or NaN gradients at |α| = Γ exactly.** `drift_matrix_param_grad` divides by zero here, which includes δ = Ω = 0 and Ω = 0 with θ₁ or θ₂ = 0. `EF` then raises `LinAlgError`. Gradients are accurate just off this line, so avoid these exact values in starting guesses.
- **`MultiEmbedder.to_primitive` raises `AttributeError`.** It uses `emb.nparams` instead of `emb.nparam`. Nothing on the fitting path calls it.
- **`MultiEmbedder.grad_A` can be out of date.** After a component is edited directly, it returns old values until another getter runs.

### Surprising limits

- **Stable `gen2x2` inputs can be rejected.** The code requires γ > Softmax(Γ, Softabs(α)), which is stricter than stability (γ > |δ|). `TwoAuxEmbedder([1.0, 0.0], 0.1, 0.0, 2.0)` raises `ValueError`, but succeeds with `sigma=50`. `from_harmonic` input and `from_matrix` are affected too. `test_param_conversion` catches this error, which hides it.
- **Zero starting values crash fits.** `LowerBoundMapper` turns p = 0 into x = -inf, with only a NumPy warning. The finite-difference Hessian is then not finite, and `EF` raises `LinAlgError`. `gen2x2` with θ = [0, 0] gives NaN parameters.
- **Silent or unclear input handling.** `from_dict` ignores unknown keys, so a misspelled `sigma` has no effect. Mappers cannot be set from a config file, which can only supply plain values. `MultiEmbedder` ignores `mappers`, and it needs a list or tuple because a generator fails with an unclear `ValueError`. Constructors reject `np.float32` and `np.int64`.

### Missing features

- `TwoAuxEmbedder.kernel_hess` raises `NotImplementedError`, which also affects `kernel(t, nu=2)` on any `MultiEmbedder` containing `gen2x2`. Fits do not use it.
- There is no general n-variable embedder. Embeddings are built only from blocks of 1 or 2 variables.
- `from_matrix` exists only for `gen2x2`, so a full `Ap` cannot be loaded back into an embedder.

### Test gaps

- `tests/test_embedding/test_multi.py`: 3 of 4 tests fail because they expect the signs used before `e2299f6`. The file has no `gen2x2` case.
- `tests/test_merit/test_properties.py` compares the two paths, but checks `gen2x2` only for the spectrum and at one ordinary point. Random test points never reach α = 0, |α| = Γ or Γ = 0.

### Outdated text in the code

- `BaseEmbedder.conventional_params` lists the `gen2x2` parameters for every embedder.
- `drift_matrix_gradient` says it differentiates with respect to `p`, but uses `x`.
- `TwoAuxEmbedder.compute_drift_matrix` says it takes conventional parameters and returns a 2×2 array. It takes primitive parameters and returns 3×3.
- The matrix diagrams in `MultiEmbedder` omit the minus sign on the first column.
- Smaller slips: the `TwoAuxEmbedder.__init__` type-check message swaps δ and Ω, `PronyCosineEmbedder.__init__` lists two mappers instead of three, and the `spectrum()` error message says `time`.

## Adding features

### A new embedder

1. Add `<name>.py` and subclass `BaseEmbedder`, following the constructor order under Conventions.
2. Implement `from_dict`, `__len__`, `_get_nparam`, `compute_drift_matrix`, `drift_matrix_param_grad`, and `kernel_func`, `kernel_grad`, `kernel_hess`, `spectrum_func`, `spectrum_grad`, `spectrum_hess`.
3. If different parameter values can give the same kernel, also override `_get_ndof`, `to_primitive` and `to_conventional`. Then apply the chain rule in `drift_matrix_param_grad`, as `TwoAuxEmbedder` does.
4. Import the class in `__init__.py` and add it to `EMBEDDER_MAP`. No other registration is needed: the config validator checks only `embedder.type`, and `MultiEmbedder` accepts any `BaseEmbedder`.
5. If A is not symmetric and θ uses more than one entry, first fix `grad_wrt_A` in `glefit/merit/properties.py`, which gives wrong θ gradients for such matrices.
6. Add `tests/test_embedding/test_<name>.py`.

### Other changes

- **New parameter limit:** subclass `BaseMapper` in `glefit/mappers/` and pass one mapper per primitive parameter through `mappers`.
- **`gen2x2` `kernel_hess`:** follow `spectrum_hess`. `expcoscosh` and `expsincsinhc` accept `nu=2, wrt="Gamma"`, and `_gamma_lower_bound(..., nu=2)` gives the derivatives of γ. First fix `expsincsinhc(..., nu=2, wrt="Gamma")`, which is wrong for Γt ≥ 0.01: see `glefit/utils/CLAUDE.md`. The second Γ-derivative also changes sign at Γ = 0, so test on both sides of it.
- **New quantity to fit:** this belongs in `glefit/merit`, not here, because every fitted quantity is computed from `Ap`.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_embedding tests/test_merit
```

A new or changed embedder should have tests that:

- compare closed forms against SymPy or `glefit.utils.numderiv.jacobian`
- compare `drift_matrix_param_grad` against `jacobian` applied to `compute_drift_matrix`
- compare the two paths, for example `MemoryKernel(grid, target, emb, "squared").grad_wrt_params()` against `emb.kernel(grid, nu=1, mapped=True)`, and the same for `MemorySpectrum`
- include special values, such as zeros and points where the code switches branch, because random values miss them
