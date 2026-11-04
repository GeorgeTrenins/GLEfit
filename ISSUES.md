# ISSUES.md

Known defects and limitations of GLEFit, one ID per issue. IDs are never reused. When a commit resolves an issue, change its status from `open` to `resolved in <commit>` and keep the entry. `accepted` marks behaviour that was examined and kept.

Only `glefit.embedding` keeps its issues here so far. The other modules still list theirs under "Known gaps" in their own `CLAUDE.md`, and move them here when they are revised.

## `glefit.embedding`

Every defect was reproduced by running the code (`src/` is unchanged from `85e5b71` to `67441e9`). Items under "Needs a decision" stay as they are until the maintainer has agreed on the intended behaviour.

### Agreed redesign

**EMB-20 · open · Reparametrise gen2x2 as described in the docs.**
Agreed with the maintainer on 2026-11-04, following `docs/Generalised Brownian oscillator auxiliary variables.md`. The primitive parameters become [r, α, g, y]. g = δ² − Ω² replaces Γ, and γ(α, g, y) from the docs replaces λ + Softmax(Γ, Softabs(α)). Its inverse is y = ½ ln[(γ² − g)(γ² − α²)]. The drift matrix uses the maximal gauge δ = γ, where θ₁, θ₂ and Ω are positive and smooth in p, and `conventional_params` and the trajectory file report this gauge. Only `optimized_Ap_matrix.json` uses the minimal gauge, with θ = (r, 0) at α = 0 as in the docs. Consequences:

- It relies on the θ-column fix of `grad_wrt_A` in `glefit.merit`. Before that fix, the α and γ gradients in the maximal gauge were wrong by up to 0.59 on values of size 0.2–2.7.
- y is finite only strictly inside the admissible region. Inputs with γ² = max(g, α²), such as an exact Brownian-oscillator kernel, cannot be represented and should raise `ValueError`, as out-of-range inputs do today.
- The closed forms are rewritten in g as EMB-2 describes, which fixes EMB-2, and `kernel_hess` is added (EMB-16).
- `sigma` and `threshold` are removed, and passing either raises an error (decided). `tutorials/` must be updated. The `legacy_tests` fixtures pass `sigma`, so their gen2x2 runs will fail.
- A new embedder method returns the Ap used for propagation (decided). It returns `A` by default, gen2x2 overrides it with the minimal gauge, and `MultiEmbedder` assembles it block by block. `optimize.py` writes it to `optimized_Ap_matrix.json` in place of `emb.A`.
- g has units of rate², and the fit Hessian takes absolute finite-difference steps in unmapped parameters (`glefit/merit/CLAUDE.md`). Expect less accurate Hessians than with Γ when rates are far from 1.
- Fit paths, trajectory files and `test_gen2x2.py` in both test suites change.

### Resolved by EMB-20

The maximal gauge has no branches, which removes EMB-1, EMB-5 and EMB-7. K and Λ are smooth in g, which removes EMB-8. γ(y) covers every strictly admissible γ, which removes EMB-9. The new closed forms fix EMB-2. Do not fix these separately. Mark them resolved in the EMB-20 commit, and EMB-16 in the commit that adds `kernel_hess`.

**EMB-1 · open · The gen2x2 gradient for α is zero at α = 0 with Γ < 0.**
`TwoAuxEmbedder.jac_conventional` sets `jac[3, 1] = np.sign(alpha)`, which is 0 at α = 0, so dAp/dα misses dδ/dα. The Hessian of K or Λ has its α off-diagonal entries at half their true value, and the Hessian of the distance loses the Gauss–Newton term in the whole α row and column. Every `from_harmonic` start lands here, as does any input with θ₁ = θ₂ and |δ| < |Ω|. An `EF` fit still reached its target in one trial, so this is easy to miss. Also listed in `glefit/merit/CLAUDE.md`.

**EMB-5 · open · The gen2x2 standard form at α = 0 disagrees with the docs.**
On the branch |α| ≥ Γ, the docs set θ = (r, 0) for α ≥ 0, but `to_conventional` and `jac_conventional` test `alpha > 0` and give θ = (0, r) at α = 0. Both give the same kernel. Decided: follow the docs. After EMB-20 this concerns only `optimized_Ap_matrix.json`.

**EMB-7 · open · gen2x2 gradients are infinite or NaN at |α| = Γ exactly.**
`drift_matrix_param_grad` divides by zero on this line, which includes δ = Ω = 0, and Ω = 0 with θ₁ = 0 or θ₂ = 0. `EF` then raises `LinAlgError`. Gradients are accurate just off the line, so avoid these exact values in starting guesses.

**EMB-8 · open · The gen2x2 spectrum has no second Γ-derivative at Γ = 0.**
The one-sided limits differ, and `spectrum_hess` returns their mean (`d2y_dGamma2 = 0`). Untested.

**EMB-9 · open · Stable gen2x2 inputs can be rejected.**
The code requires γ > Softmax(Γ, Softabs(α)), which is stricter than the condition γ² ≥ max(g, α²), γ ≥ 0 in the docs. `TwoAuxEmbedder([1.0, 0.0], 0.1, 0.0, 2.0)` raises `ValueError` but succeeds with `sigma=50`. Input to `from_harmonic` and `from_matrix` is affected too. In `legacy_tests`, `test_param_conversion` catches this error, which hides it.

**EMB-2 · open · gen2x2 kernels are wrong for integer times.**
`kernel(t, nu=1)` silently truncates to integers and can return all zeros, and `nu=0` raises `UFuncTypeError`. The cause is `np.empty_like(t)` in `expcoscosh` and `expsincsinhc` (`glefit/utils/special.py`, also listed in `glefit/utils/CLAUDE.md`). Fix: evaluate `kernel_func` and `kernel_grad` as `Density.K` in `rpmdgle/sysbath/spectral/brownian.py` (rpgle-model-dynamics) does. Take |t|, branch once on the sign of the scalar g instead of masking each element, and build the result from whole-array NumPy expressions, which return floats for integer input. The hyperbolic branch becomes a sum of decaying exponentials, and `expm1` replaces the special case for small κt. With ω̄ = √−g and κ = √g:

```text
g < 0:  K = r² e^{−γt} [cos(ω̄t) − (α/ω̄) sin(ω̄t)]
g = 0:  K = r² e^{−γt} (1 − αt)
g > 0:  K = r² e^{−(γ−κ)t} [(1 + e^{−2κt})/2 + (α/2κ) expm1(−2κt)]
```

These match the current `kernel_func` to 1e-16 for float times. `expcoscosh` and `expsincsinhc` are then unused in `src/`.

**EMB-16 · open · `TwoAuxEmbedder.kernel_hess` is not implemented.**
It raises `NotImplementedError`, which also affects `kernel(t, nu=2)` on any `MultiEmbedder` with a gen2x2 component. `test_gen2x2.py` marks it as a strict xfail. Fits do not use it. Agreed: add it right after EMB-20, in its own commit, by differentiating the closed forms of EMB-2 once more. K is then analytic in g, so there is no kink at g = 0.

### Wrong results

**EMB-3 · open · Integer arrays given to the parameter setters are truncated in the mapped space.**
`_forward_map`, `_inverse_map`, `jac_px` and `hess_px` in `_base.py` allocate their output with `np.zeros_like(input)`. After `primitive_params = [3, 1]` on a `prony` embedder, x is [1, 0] instead of [ln 3, 0], so x no longer matches p. Constructors convert to float, so configs are not affected.

**EMB-4 · open · `MultiEmbedder.to_primitive` raises `AttributeError`.**
It uses `emb.nparams` instead of `emb.nparam`. Fits never call it.

### Needs a decision

**EMB-6 · open · `MultiEmbedder.grad_A` can be out of date.**
After a component is edited directly, it returns old values until another getter runs. Either fix it, or document direct edits of components as unsupported (the conventions already say to set values on the `MultiEmbedder`).

**EMB-10 · open · Zero starting values crash fits.**
`LowerBoundMapper` turns p = 0 into x = −inf, with only a NumPy warning. The finite-difference Hessian is then not finite, and `EF` raises `LinAlgError`. gen2x2 with θ = [0, 0] gives NaN parameters.

**EMB-11 · open · `from_dict` ignores unknown keys.**
A misspelled key, such as `sigmaa`, is accepted silently and the default applies.

### Limitations

**EMB-12 · open · Fits never read the left column of Ap.**
`MemoryKernel` and `MemorySpectrum` take θ from the top row of Ap only. A wrong left column from `compute_drift_matrix` therefore leaves fits unchanged, but it is written to `optimized_Ap_matrix.json`, which simulations use. The left column of `drift_matrix_param_grad` does enter the fit gradient. `tests/test_embedding` checks the left column directly, and `DriftMatrixReference` reads θ from it.

**EMB-13 · accepted · `to_primitive` loses relative accuracy in Γ when |Γ| ≪ δ.**
g = δ² − Ω² cancels, so the relative error in Γ is of order ε·δ²/Γ² (1–4e-11 at Γ = ±1e-3). This is conditioning, not a bug. After EMB-20 the same holds for g, with relative error of order ε·δ²/|g|.

**EMB-14 · open · `MultiEmbedder` needs a list or tuple of components.**
A generator fails with the unclear `ValueError: need at least one array to concatenate`.

**EMB-15 · open · Constructors reject NumPy scalars.**
`np.float32` and `np.int64` parameters raise `TypeError`, for example `θ must be a float, got float32`.

### Missing features

**EMB-17 · open · `MultiEmbedder` should reject a `mappers` argument.**
Custom mappers are for debugging only, and the components do the mapping. `MultiEmbedder` currently stores a `mappers` argument but never uses it. It should raise an error instead.

**EMB-18 · open · `from_matrix` exists only for gen2x2.**
A full Ap cannot be loaded back into an embedder.

### Outdated text in the code

**EMB-19 · open · Docstrings, annotations and messages disagree with the code.**
The gen2x2 items are rewritten with EMB-20.

- `BaseEmbedder.conventional_params` lists the gen2x2 parameters for every embedder.
- `drift_matrix_gradient` says it differentiates with respect to p, but uses x.
- `TwoAuxEmbedder.compute_drift_matrix` says it takes conventional parameters and returns a 2×2 array. It takes primitive parameters and returns 3×3.
- The matrix diagrams in `MultiEmbedder` omit the minus sign on the first column.
- The `kernel()` and `spectrum()` docstrings should give the shapes (ngrid,), (ndof, ngrid) and (ndof, ndof, ngrid), with the grid axis kept for scalar input.
- Annotations say `NDArray` where lists are accepted.
- The `TwoAuxEmbedder.__init__` type-check message swaps δ and Ω, `PronyCosineEmbedder.__init__` lists two mappers instead of three, and the `spectrum()` error message says `time`.

### Tracked in other modules

These affect embedders, but the fix belongs elsewhere:

- `MemorySpectrum` is wrong where A cannot be diagonalized. Today that is gen2x2 at Γ = 0 with α ≠ 0. After EMB-20 it is g = 0 for every α, where A = γ[[2, 1], [−1, 0]]: the value was off by 0.28 at g = 0, but only by 2e-11 at g = 1e-12. The closed form and `MemoryKernel` are correct. See `glefit/merit/CLAUDE.md`.
- `glefit.utils.numderiv` builds the Hessian used in fits, crashes with `max_step` on NumPy 2, and has not been reviewed. See `glefit/utils/CLAUDE.md`.
