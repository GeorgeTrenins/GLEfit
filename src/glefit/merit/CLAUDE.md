# CLAUDE.md: `glefit.merit`

This module defines the merit functions that GLEFit minimizes. Its open issues, accepted limits and missing features are in `ISSUES.md` at the repository root, as `MER-n`.

## Background

This measures how well a Kantorovich-style Markovian embedding reproduces generalized Langevin dynamics (GLD), given target values of the memory kernel K(t) or of its cosine transform, the spectrum Λ(ω). An `embedder` from `glefit.embedding` defines the embedding by the (n+1)×(n+1) drift matrix:

```text
Ap = [[ 0, θᵀ],      K(t) = θᵀ exp(-|t|A) θ
      [-θ, A ]]      Λ(ω) = θᵀ A (A² + ω²)⁻¹ θ
```
θᵀ is read from row 0 of `Ap` and θ from the negated column 0, as in the dynamics. Every embedder makes the two equal, but `function` and `grad_wrt_A` treat them as independent, so both describe the same function of `Ap`.

The `optimizer` changes a vector `x` of p parameters that have no limits on their values, and the `embedder` turns `x` into `Ap`. The merit function, called the `distance` in the code, is:

```text
distance = Σᵢ wᵢ (hᵢ - yᵢ)²      default wᵢ = Δᵢ / (L·max|y|²)
```
Here hᵢ is the property (K or Λ) at grid point i, yᵢ is the target, wᵢ is the weight, and (hᵢ - yᵢ)² is the deviation. By default, L is the length of the grid, and Δᵢ = (tᵢ₊₁ - tᵢ₋₁)/2 at interior points and the spacing to the neighbour at each end, so a uniform grid of m points gives every wᵢ = 1/((m - 1)·max|y|²). The distance is then about the mean squared relative deviation over the range, and `gtol` does not depend on m or on the scale of the data. An explicit `weight` replaces the default. Derivatives of h with respect to `x` come from the chain rule, ∂h/∂x = ∂h/∂Ap · ∂Ap/∂x. This module supplies ∂h/∂Ap and the `embedder` supplies ∂Ap/∂x. Second derivatives of h are finite differences of ∂h/∂x, not analytic expressions.

## Contents

| Config `type` | Class | Fits | Values use | Gradients use |
| --- | --- | --- | --- | --- |
| `kernel` | `MemoryKernel` | K on a time grid | `scipy.linalg.expm` | `scipy.linalg.expm_frechet` |
| `spectrum` | `MemorySpectrum` | Λ on a frequency grid | Λ(ω) = Re θᵀ(A + iω)⁻¹θ, one complex solve per ω | The same solves, with A and Aᵀ |

- `properties.py` holds these two classes.
- `_base.py` holds the base classes: `BaseProperty`, which both properties subclass, and `BaseDeviation`, with its child `SquaredDeviation`. It also holds `METRICS`, which links each metric name to its class.
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
    weight: 1.0       # optional: a number, or a list with one number per grid point; omitted or null gives the default
```

## Conventions

- **Names:** `grad_wrt_A`, `grad_wrt_params` and `gradhess_wrt_params` differentiate the property. `gradient`, `hessian`, `both` and `all` differentiate the distance.
- **Shapes:** m is the number of grid points. `function` returns (m,), `grad_wrt_A` returns (m, n+1, n+1), `grad_wrt_params` returns (p, m), and `gradhess_wrt_params` adds a Hessian of shape (p, p, m). `gradient` returns (p,) and `hessian` returns (p, p).
- **The `x` argument:** `x=None` means the embedder's current parameters. Passing `x` evaluates at trial values and leaves the embedder unchanged. `distance` and `value` take no argument, so they always use current parameters.
- **The name `A`:** the `A` argument of `grad_wrt_A` is the full matrix `Ap`. In local variables, and in `_grad_thetaT_expA_theta`, `A` is the n×n block.
- **What this module needs from an embedder:** `A`, `x`, `compute_drift_matrix`, `_inverse_map` and `drift_matrix_x_grad`. It never calls the embedder's closed-form `kernel()` or `spectrum()`.
- **Constructors:** `BaseProperty.__init__` stores float copies of `grid` and `target`, and raises `ValueError` unless the grid is a non-empty, strictly increasing 1-D array and `target` has its shape. It sets `_grid` before calling `_set_weight`, which needs the grid. `_set_weight` stores a float copy of the weight, and raises `ValueError` unless every entry is finite and non-negative and one at least is positive. `weight=None` gives the default above, which needs `_target` as well as `_grid`, and raises `ValueError` for a target that is zero everywhere. An unknown `metric` also raises `ValueError`.
- **`from_dict`:** takes only `data`, `metric` and `weight`. It raises `ConfigError` for a missing or unknown entry and for a dataset without a grid, and turns the constructor's `ValueError` into `ConfigError`.
- **Deviations:** they work point by point and return arrays. `hessian` returns only the second derivative at each point. `__init_subclass__` adds input checks to subclass methods automatically.
- **Outside use:** the optimizers call only `all()`. `analysis/plot_objective.py` uses `distance`, `function`, `grid` and `target`, and it swaps `merit.emb` to plot each component.
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Adding features

### A new property on a grid

1. Subclass `BaseProperty` in `properties.py`.
2. Implement `function(x=None)`. For a trial `x`, build the matrix with `emb.compute_drift_matrix(emb._inverse_map(x))`. Read θᵀ from row 0 and θ from the negated column 0.
3. Implement `grad_wrt_A(A=None)`. Differentiate θᵀ in row 0 and θ in the negated column 0 of `Ap` separately, as the existing properties do. The two derivatives differ unless A is symmetric.
4. Import the class in `__init__.py` and add it to `PROPERTY_MAP`. No other registration is needed: the config validator checks only `type` and `parameters`.

### Other changes

- **New metric:** subclass `BaseDeviation`, implement `__call__`, `gradient` and `hessian` point by point, and add it to `METRICS`.
- **Combined objective:** add a class that sums the distances of weighted `BaseProperty` objects. Optimizers need only `all()`, but `ConfigHandler` and `optimize.py` assume a single `merit_function`.
- **Weights from a data file:** change `config/data_io.py`, and also `from_dict`, which unpacks each dataset as `(grid, target)`.
- **Analytic Hessian:** override `gradhess_wrt_params`.

## Testing

`glefit` is not installed. From the repository root, run:

```bash
PYTHONPATH=src conda run -n tools python -m pytest tests/test_merit
```

`test_properties.py` tests K, Λ and `grad_wrt_A` for single drift matrices, including non-symmetric A with several θ entries, A that cannot be diagonalized, a left column other than −θ, and K at negative times. `test_distance.py` tests the distance and its derivatives for each shipped embedder, with non-uniform weights, at the embedder's parameters and at a trial `x`. `test_inputs.py` tests the constructor's checks of `grid`, `target` and `weight`, that `from_dict` passes on the data and the weights, and the errors for bad entries.

A new or changed property or metric should have tests that:

- compare `grad_wrt_A` entry by entry against `DriftMatrixReference` in `tests/conftest.py`, using a general matrix (non-symmetric A, several θ entries, a left column other than −θ) as well as the shipped embedders. Do not use `glefit.utils.numderiv` as a reference: finite differences are not exact
- compare `gradient` and `hessian` against the embedders' closed-form derivatives, with non-uniform weights, as `tests/test_merit/test_distance.py` does
- compare against closed forms where they exist, for example `emb.spectrum(grid, nu=1, mapped=True)`
- include special drift matrices, such as one with a repeated eigenvalue. Points where an embedder's closed form switches branch need no tests here: the derivatives via chain rule w.r.t. drift matrix have no branches, and `tests/test_embedding` tests the closed forms at those points
