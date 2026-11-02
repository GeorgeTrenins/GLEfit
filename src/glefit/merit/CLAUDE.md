# CLAUDE.md: `glefit.merit`

This module defines the merit functions that GLEFit minimizes.

## Background

This measures how well a Kantorovich-style Markovian embedding reproduces generalized Langevin dynamics (GLD), given target values of the memory kernel K(t) or of its cosine transform, the spectrum Λ(ω). An `embedder` from `glefit.embedding` defines the embedding by the (n+1)×(n+1) drift matrix:

```text
Ap = [[ 0, θᵀ],      K(t) = θᵀ exp(-tA) θ
      [-θ, A ]]      Λ(ω) = θᵀ A (A² + ω²)⁻¹ θ
```
The `optimizer` changes a vector `x` of p parameters that have no limits on their values, and the `embedder` turns `x` into `Ap`. The merit function, called the `distance` in the code, is:

```text
distance = Σᵢ wᵢ (hᵢ - yᵢ)²
```
Here hᵢ is the property (K or Λ) at grid point i, yᵢ is the target, wᵢ is the weight, and (hᵢ - yᵢ)² is the deviation. Derivatives of h with respect to `x` come from the chain rule, ∂h/∂x = ∂h/∂Ap · ∂Ap/∂x. This module supplies ∂h/∂Ap and the `embedder` supplies ∂Ap/∂x. Second derivatives of h are finite differences of ∂h/∂x, not analytic expressions.

## Contents

| Config `type` | Class | Fits | Values use | Gradients use |
| --- | --- | --- | --- | --- |
| `kernel` | `MemoryKernel` | K on a time grid | `scipy.linalg.expm` | `scipy.linalg.expm_frechet` |
| `spectrum` | `MemorySpectrum` | Λ on a frequency grid | Eigenvectors of A | Eigenvectors of A |

- `properties.py` holds these two classes.
- `_base.py` holds the base classes: `BaseProperty`, with its children `BaseArrayProperty` and `BaseScalarProperty`, and `BaseDeviation`, with its child `SquaredDeviation`. It also holds `METRICS`, which links each metric name to its class.
- `__init__.py` holds `PROPERTY_MAP`, which links each config `type` to its class.

```python
from glefit.merit import MemorySpectrum
merit = MemorySpectrum(grid, target, emb, "squared", weight=1.0)
merit.value                                  # Λ at each grid point
distance, gradient, hessian = merit.all()    # what the optimizers call
```

```yaml
merit_function:
  type: spectrum
  parameters:
    data: spectrum    # name of a dataset under the top-level `data` key
    metric: squared
    weight: 1.0       # optional: a number, or a list with one number per grid point
```

## Conventions

- **Names:** `grad_wrt_A`, `grad_wrt_params` and `gradhess_wrt_params` differentiate the property. `gradient`, `hessian`, `both` and `all` differentiate the distance.
- **Shapes:** m is the number of grid points. `function` returns (m,), `grad_wrt_A` returns (m, n+1, n+1), `grad_wrt_params` returns (p, m), and `gradhess_wrt_params` adds a Hessian of shape (p, p, m). `gradient` returns (p,) and `hessian` returns (p, p).
- **The `x` argument:** `x=None` means the embedder's current parameters. Passing `x` evaluates at trial values and leaves the embedder unchanged. `distance` and `value` take no argument, so they always use current parameters.
- **The name `A`:** the `A` argument of `grad_wrt_A` is the full matrix `Ap`. In local variables, and in `_grad_thetaT_expA_theta`, `A` is the n×n block.
- **What this module needs from an embedder:** `A`, `x`, `compute_drift_matrix`, `_inverse_map` and `drift_matrix_x_grad`. It never calls the embedder's closed-form `kernel()` or `spectrum()`.
- **Constructors:** `BaseArrayProperty.__init__` sets `_grid` before calling `super().__init__`, because `_set_weight` needs the grid.
- **Deviations:** they work point by point and return arrays. `hessian` returns only the second derivative at each point. `__init_subclass__` adds input checks to subclass methods automatically.
- **Outside use:** the optimizers call only `all()`. `analysis/plot_objective.py` uses `distance`, `function`, `grid` and `target`, and it swaps `merit.emb` to plot each component.
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Known gaps

Checked at commit `85e5b71`. Everything under Wrong results and Surprising limits was reproduced by running the code.

### Wrong results

- **θ gradients are wrong for a general A.** Both `grad_wrt_A` methods set `ans[:,1:,0] = -ans[:,0,1:]`, which is only valid when A is symmetric or θ uses a single entry. With a general 3×3 A, errors reached 0.59 on gradients of size 2.9, while the entries for A itself were correct. Today's embedders hide this: `pronycos` uses one θ entry, and `gen2x2` always has either Ω = 0 or one θ entry equal to 0. **Any new embedder with a non-symmetric A and more than one θ entry gets wrong gradients with no error.** Fix: set `ans[:,1:,0]` to `-exp(-tAᵀ)θ` in `MemoryKernel` and to `-np.real(u)` in `MemorySpectrum`. Both matched numerical derivatives to about 1e-12.
- **`gen2x2` at α = 0 with Γ < 0.** A bug in the embedder's `jac_conventional` makes the gradient for α equal to 0, and the Hessian entries linking α to the other parameters come out at half their true value. Every `from_harmonic` start is at this point. The fix belongs in `glefit/embedding/gen2x2.py`.
- **`MemorySpectrum` at repeated eigenvalues that cannot be diagonalized.** This happens in `gen2x2` at Γ = 0 with α ≠ 0 (|δ| = |Ω|). The value was wrong by 0.5 and the gradient by 1.7 in one check, and the error is still about 2e-9 at Γ = 1e-8. Ordinary repeated eigenvalues, such as two identical components, are fine. `MemoryKernel` is not affected. Solving (A² + ω²)y = θ at each frequency instead gave the correct value.
- **`BaseScalarProperty.gradient(x)` uses the wrong point.** It calls `self.value`, which ignores `x`, so it returned 3.54 where the true value was 11.46. Use `self.function(x=x)`. `both(x)` and `all(x)` are correct.
- **`BaseScalarProperty.from_dict` always raises `TypeError`.** It calls `parameters.get("weight", default=1.0)`, and `dict.get` takes no keyword arguments.

No scalar property exists yet, so the last two cannot be reached today.

### Surprising limits

- **The distance is a plain sum, while `gtol` is absolute.** The distance grows with the size of the data and with the number of grid points. With targets 1000 times smaller and `gtol=1e-4`, a fit stopped at step 0 with 50% error, and it counts as converged. Doubling the grid doubles the distance and gradient. Dense parts of a non-uniform grid count for more unless `weight` corrects for it.
- **The Hessian's finite-difference step is about 7e-4·max(1, |x|).** Log-mapped parameters are almost unaffected by units, with errors below 3e-6. α and Γ in `gen2x2` have no mapper, so the step is absolute, and it must also be small compared with 1/σ. With rates of order 1, 0.1 and 0.01 (and σ scaled to match), the Hessian error was 5e-9, 5e-5 and 50%. Gradients are not affected, and `EF` still converged in one trial with rates of 0.001, taking 11 steps instead of 8. Choosing units that make the rates near 1 avoids this, and so would passing a step for each parameter to `jacobian`.
- **`all()` is expensive.** It needs 4p + 2 gradient evaluations. With 44 parameters and 400 grid points, one call took about 0.8 s for the spectrum and 13 s for the kernel.
- **Time grids must not be negative.** `MemoryKernel` uses t as given, while the embedders' closed forms use |t|. At t = -1 they gave -0.69 and -0.25.
- **Direct Python use is checked late.** Constructors do not compare `target` with `grid`. A wrong length fails at the first evaluation, and a target of length 1 is silently applied to every point. Integer `target` arrays fail at `distance`, and integer `grid` arrays fail at the `MemoryKernel` gradient. The config loaders convert to float and check lengths, so config runs are safe.
- **Weak config checks.** An unknown `metric` raises `KeyError` instead of `ConfigError`. A scalar dataset passed to `kernel` or `spectrum` fails with an unclear `IndexError`. Unknown keys are ignored, so a misspelled `weight` has no effect. Negative weights are accepted, which gives a negative distance.

### Missing features

- `squared` is the only metric.
- `BaseScalarProperty` has no concrete subclass.
- There are no combined objectives, such as fitting K and Λ together. Docstrings mention them, but the config accepts one `merit_function`.
- Per-point weights cannot be read from a data file, as `TODO.md` notes. They must be listed in the config.
- There is no analytic Hessian.

### Test gaps

- `tests/test_merit/test_properties.py` has no direct tests of `distance`, `both`, `all`, `hessian`, `weight`, `from_dict`, or anything scalar. `tests/test_optimize` reaches `all()`, but only checks that runs finish. A one-off check found the array versions correct and consistent, including with per-point weights.
- Gradient tests use only embedders that hide the θ bug. `test_spectrum_generic_A` uses a general matrix, but checks only the value.
- `gen2x2` appears only in spectrum tests, at one ordinary point.

### Outdated text in the code

- The `BaseDeviation` example says the metric returns one summed number. It returns one number per point.
- `check_value_target` says it checks that dimensions match. It checks each argument separately.
- `MemorySpectrum` and `_compute_spec_from_A` call Λ the spectral density. Elsewhere, the spectral density is ω·Λ(ω).
- `BaseProperty` says "automatic differentiation". The chain rule is written by hand.
- The module docstring and `BaseProperty` mention combined objectives, which do not exist.
- `BaseArrayProperty.from_dict` is annotated as returning `BaseScalarProperty`.

## Adding features

### A new property on a grid

1. Subclass `BaseArrayProperty` in `properties.py`.
2. Implement `function(x=None)`. For a trial `x`, build the matrix with `emb.compute_drift_matrix(emb._inverse_map(x))`.
3. Implement `grad_wrt_A(A=None)`. Row 0 and column 0 of `Ap` need separate derivatives, so do not copy the current θ code.
4. Import the class in `__init__.py` and add it to `PROPERTY_MAP`. No other registration is needed: the config validator checks only `type` and `parameters`.

### Other changes

- **New metric:** subclass `BaseDeviation`, implement `__call__`, `gradient` and `hessian` point by point, and add it to `METRICS`.
- **New scalar property:** fix the two `BaseScalarProperty` bugs first, then subclass it. `function` must return a float, because 0-dimensional arrays fail the input checks.
- **Combined objective:** add a `BaseProperty` that sums weighted parts. Optimizers need only `all()`, but `ConfigHandler` and `optimize.py` assume a single `merit_function`.
- **Weights from a data file:** change `config/data_io.py`, and also `from_dict`, which unpacks each dataset as `(grid, target)`.
- **Analytic Hessian:** override `gradhess_wrt_params`.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_merit
```

A new or changed property or metric should have tests that:

- compare `grad_wrt_params` against `glefit.utils.numderiv.jacobian` applied to `function`, using a general matrix (non-symmetric A, several θ entries) as well as the shipped embedders
- compare `gradient` and `hessian` against numerical derivatives of the distance, with non-uniform weights
- compare against closed forms where they exist, for example `emb.spectrum(grid, nu=1, mapped=True)`
- include special values, such as repeated eigenvalues and points where the embedder switches branch
