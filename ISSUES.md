# ISSUES.md

Known defects and limitations of GLEFit, one ID per issue. IDs are never reused. When a commit resolves an issue, change its status from `open` to `resolved in <commit>` and keep the entry. `accepted` marks behaviour that was examined and kept.

Only `glefit.embedding` keeps its issues here so far. The other modules still list theirs under "Known gaps" in their own `CLAUDE.md`, and move them here when they are revised.

## `glefit.embedding`

Every defect was reproduced by running the code (`src/` is unchanged from `85e5b71` to `67441e9`). Items under "Needs a decision" stay as they are until the maintainer has agreed on the intended behaviour.

### Agreed redesign

**EMB-20 · open · Reparametrise gen2x2 as described in the docs.**
Agreed with the maintainer on 2026-11-04/05, following `docs/Generalised Brownian oscillator auxiliary variables.md`. `290c7d3` replaced λ + Softmax(Γ, Softabs(α)) with the primitive parameters [r, α, g, y], where g = δ² − ζ² and γ(α, g, y) covers every strictly admissible γ. `9d29a67` replaced these with the note's (r, γ, Ω, φ). `to_primitive` is eq. (3) of the note, only γ is constrained, and the drift matrix uses the maximal gauge δ = γ, where θ = r(cos φ, sin φ) and ζ = Ω. `conventional_params` and the trajectory file report this gauge. Only `optimized_Ap_matrix.json` should use the minimal gauge, with θ = (r, 0) at cos 2φ = 0 as in the docs.

`from_matrix` computes (r, γ, Ω, φ) from rotation invariants of θ and A since `9291892` (EMB-22), `kernel_hess` exists since `4a5b5f3` (EMB-16), and `propagation_matrix()` returns the drift matrix for propagation since the `propagation_matrix` commit: the drift matrix by default, the minimal gauge for gen2x2, and the components' matrices assembled block by block for `MultiEmbedder`. One part remains:

- `optimize.py` writes `propagation_matrix()` to `optimized_Ap_matrix.json` in place of `emb.A`. Until then, that file holds the maximal gauge, which is a valid embedding in which s₂ receives no noise.

Consequences:

- It relies on the θ-column fix of `grad_wrt_A` in `glefit.merit` (`c1c86e3`). Before that fix, the α and γ gradients in the maximal gauge of `290c7d3` were wrong by up to 0.59 on values of size 0.2–2.7.
- K and Λ depend only on r², Ω² and cos 2φ, so every real r, Ω and φ with γ > 0 is a valid embedding. `to_primitive` accepts γ = |δ|, so an exact Brownian-oscillator kernel can be represented, which [r, α, g, y] could not. It rejects θ = 0, because the φ-derivative vanishes at r = 0. Ω = 0 is allowed, but there Λ(0) is 0/0.
- `sigma` and `threshold` raise `TypeError`, and so does `Omega`, which is now `zeta`. `tutorials/` still pass them and are left as they are for now. The `legacy_tests` that pass `sigma` fail (19 tests).
- r and Ω carry units, have no mapper, and the fit Hessian takes absolute finite-difference steps in unmapped parameters (`glefit/merit/CLAUDE.md`). Expect less accurate Hessians when they are far from 1.
- Spectrum fits can stall near Ω ≈ 0, where Λ(0) = 2γr² sin²φ/Ω² is singular. This belongs to `glefit.merit` (see `src/glefit/embedding/TODO.md`).

### Resolved by EMB-20

`290c7d3` resolved these. Its maximal gauge has no branches, which removed EMB-1, EMB-5 and EMB-7. K and Λ are smooth in g, which removed EMB-8. γ(y) covered every strictly admissible γ, which removed EMB-9. The new closed forms fixed EMB-2. `9d29a67` kept these fixes and resolved EMB-13. `4a5b5f3` resolved EMB-16.

**EMB-1 · resolved in `290c7d3` · The gen2x2 gradient for α is zero at α = 0 with Γ < 0.**
`TwoAuxEmbedder.jac_conventional` sets `jac[3, 1] = np.sign(alpha)`, which is 0 at α = 0, so dAp/dα misses dδ/dα. The Hessian of K or Λ has its α off-diagonal entries at half their true value, and the Hessian of the distance loses the Gauss–Newton term in the whole α row and column. Every `from_harmonic` start lands here, as does any input with θ₁ = θ₂ and |δ| < |ζ|. An `EF` fit still reached its target in one trial, so this is easy to miss. Also listed in `glefit/merit/CLAUDE.md`.

**EMB-5 · resolved in `290c7d3` · The gen2x2 standard form at α = 0 disagrees with the docs.**
On the branch |α| ≥ Γ, the docs set θ = (r, 0) for α ≥ 0, but `to_conventional` and `jac_conventional` test `alpha > 0` and give θ = (0, r) at α = 0. Both give the same kernel. Decided: follow the docs. Since EMB-20, `to_conventional` returns the maximal gauge, so the decision applies only to the minimal gauge of the propagation Ap, which must give θ = (r, 0) at α = 0, that is, at cos 2φ = 0. `propagation_matrix()` does so since the `propagation_matrix` commit.

**EMB-7 · resolved in `290c7d3` · gen2x2 gradients are infinite or NaN at |α| = Γ exactly.**
`drift_matrix_param_grad` divides by zero on this line, which includes δ = ζ = 0, and ζ = 0 with θ₁ = 0 or θ₂ = 0. `EF` then raises `LinAlgError`. Gradients are accurate just off the line, so avoid these exact values in starting guesses.

**EMB-8 · resolved in `290c7d3` · The gen2x2 spectrum has no second Γ-derivative at Γ = 0.**
The one-sided limits differ, and `spectrum_hess` returns their mean (`d2y_dGamma2 = 0`). Untested.

**EMB-9 · resolved in `290c7d3` · Stable gen2x2 inputs can be rejected.**
The code requires γ > Softmax(Γ, Softabs(α)), which is stricter than the condition γ² ≥ max(g, α²), γ ≥ 0 in the docs. `TwoAuxEmbedder([1.0, 0.0], 0.1, 0.0, 2.0)` raises `ValueError` but succeeds with `sigma=50`. Input to `from_harmonic` and `from_matrix` is affected too. In `legacy_tests`, `test_param_conversion` catches this error, which hides it.

**EMB-2 · resolved in `290c7d3` · gen2x2 kernels are wrong for integer times.**
`kernel(t, nu=1)` silently truncates to integers and can return all zeros, and `nu=0` raises `UFuncTypeError`. The cause is `np.empty_like(t)` in `expcoscosh` and `expsincsinhc` (`glefit/utils/special.py`, also listed in `glefit/utils/CLAUDE.md`). Fix: evaluate `kernel_func` and `kernel_grad` as `Density.K` in `rpmdgle/sysbath/spectral/brownian.py` (rpgle-model-dynamics) does. Take |t|, branch once on the sign of the scalar g instead of masking each element, and build the result from whole-array NumPy expressions, which return floats for integer input. The hyperbolic branch becomes a sum of decaying exponentials, and `expm1` replaces the special case for small κt. With ω̄ = √−g and κ = √g:

```text
g < 0:  K = r² e^{−γt} [cos(ω̄t) − (α/ω̄) sin(ω̄t)]
g = 0:  K = r² e^{−γt} (1 − αt)
g > 0:  K = r² e^{−(γ−κ)t} [(1 + e^{−2κt})/2 + (α/2κ) expm1(−2κt)]
```

These matched the old `kernel_func` to 1e-16 for float times. The g-derivative uses `sinc` from `glefit.utils.special` for √g·t ≤ 1 and decaying exponentials above. `expcoscosh` and `expsincsinhc` are now unused in `src/`.

**EMB-16 · resolved in `4a5b5f3` · `TwoAuxEmbedder.kernel_hess` is not implemented.**
It raised `NotImplementedError`, which also affected `kernel(t, nu=2)` on any `MultiEmbedder` with a gen2x2 component. `test_gen2x2.py` marked it as a strict xfail. Fits do not use it. It was added by differentiating both forms of the kernel once more: the trigonometric form in G = γ² − Ω² and the two decaying exponentials. K is analytic in G, so there is no kink at G = 0. With z = t√(−G) and s = sinc, ∂²sinhc(t√G)/∂G² = (t⁴/32)[s + 2s″ + s⁗](z).

### Wrong results

**EMB-3 · open · Integer arrays given to the parameter setters are truncated in the mapped space.**
`_forward_map`, `_inverse_map`, `jac_px` and `hess_px` in `_base.py` allocate their output with `np.zeros_like(input)`. After `primitive_params = [3, 1]` on a `prony` embedder, x is [1, 0] instead of [ln 3, 0], so x no longer matches p. Constructors convert to float, so configs are not affected.

**EMB-22 · resolved in `9291892` · gen2x2 `from_matrix` rejects some semi-definite A through rounding.**
`rotate_matrix` rejected A when `eigh` returned a negative eigenvalue for its symmetric part S. For a zero eigenvalue, rounding gives about −1e-17, so a rotated Brownian oscillator (θ along the undamped variable) was rejected at 29 of 62 rotation angles. `from_matrix` now checks tr S ≥ 0 and det S ≥ −4ε(tr S)², and sets invariants that rounding makes negative to zero.

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

**EMB-13 · resolved in `9d29a67` · `to_primitive` loses relative accuracy in g when |g| ≪ δ².**
g = δ² − ζ² cancels, so its relative error is of order ε·δ²/|g|. This is conditioning, not a bug. Before EMB-20 the same held for Γ = sign(g)·√|g| (1–4e-11 at Γ = ±1e-3). The primitive parameters no longer contain g, and Ω² = (γ − δ)(γ + δ) + ζ² is a sum of non-negative terms.

**EMB-14 · open · `MultiEmbedder` needs a list or tuple of components.**
A generator fails with the unclear `ValueError: need at least one array to concatenate`.

**EMB-15 · open · Constructors reject NumPy scalars.**
`np.float32` and `np.int64` parameters raise `TypeError`, for example `θ must be a float, got float32`.

### Missing features

**EMB-17 · open · `MultiEmbedder` should reject a `mappers` argument.**
Custom mappers are for debugging only, and the components do the mapping. `MultiEmbedder` currently stores a `mappers` argument but never uses it. It should raise an error instead.

**EMB-18 · open · `from_matrix` exists only for gen2x2.**
A full Ap cannot be loaded back into an embedder.

**EMB-21 · open · gen2x2 `from_matrix` rejects some admissible kernels.**
It rejects any 2×2 block A whose symmetric part is indefinite, even when the kernel is admissible (bounded, with J ≥ 0) and the maximal gauge would represent it. The generality proof in the gen2x2 note shows that such kernels exist, for example θ = (1, 1)/√2 with A = [[2, 3], [−3, −0.1]]. On 2026-11-05 the maintainer decided to keep rejecting such input. Low priority.

### Outdated text in the code

**EMB-19 · open · Docstrings, annotations and messages disagree with the code.**
`290c7d3` fixed the gen2x2 items: the `compute_drift_matrix` docstring and the `__init__` type-check message.

- `BaseEmbedder.conventional_params` lists the gen2x2 parameters for every embedder.
- `drift_matrix_gradient` says it differentiates with respect to p, but uses x.
- The matrix diagrams in `MultiEmbedder` omit the minus sign on the first column.
- The `kernel()` and `spectrum()` docstrings should give the shapes (ngrid,), (ndof, ngrid) and (ndof, ndof, ngrid), with the grid axis kept for scalar input.
- Annotations say `NDArray` where lists are accepted.
- `PronyCosineEmbedder.__init__` lists two mappers instead of three, and the `spectrum()` error message says `time`.

### Tracked in other modules

These affect embedders, but the fix belongs elsewhere:

- `MemorySpectrum` is wrong where A cannot be diagonalized. Since EMB-20 that is gen2x2 at critical damping Ω = γ, for every φ, where A = γ[[2, 1], [−1, 0]]: the value was off by 0.28 at γ² − Ω² = 0, but only by 2e-11 at γ² − Ω² = 1e-12. Before EMB-20 it was Γ = 0 with α ≠ 0. The closed form and `MemoryKernel` are correct. See `glefit/merit/CLAUDE.md`.
- `glefit.utils.numderiv` builds the Hessian used in fits, crashes with `max_step` on NumPy 2, and has not been reviewed. See `glefit/utils/CLAUDE.md`.
