# ISSUES.md

Known defects and limitations of GLEFit, one ID per issue. IDs are never reused. When a commit resolves an issue, move it to "Resolved" as a one-line entry that names the commit. `accepted` marks behaviour that was examined and kept.

Only `glefit.embedding` keeps its issues here so far. The other modules still list theirs under "Known gaps" in their own `CLAUDE.md`, and move them here when they are revised.

## `glefit.embedding`

### Open

**EMB-12 · open · Fits never read the left column of Ap.**
`MemoryKernel` and `MemorySpectrum` take θ from the top row of Ap only. A wrong left column from `compute_drift_matrix` therefore leaves fits unchanged, but it is written to `optimized_Ap_matrix.json`, which simulations use. The left column of `drift_matrix_param_grad` does enter the fit gradient. `tests/test_embedding` checks the left column directly, and `DriftMatrixReference` reads θ from it.

**EMB-14 · open · `MultiEmbedder` needs a list or tuple of components.**
A generator fails with the unclear `ValueError: need at least one array to concatenate`.

**EMB-15 · open · Constructors reject NumPy scalars.**
`np.float32` and `np.int64` parameters raise `TypeError`, for example `θ must be a float, got float32`.

**EMB-18 · open · `from_matrix` exists only for gen2x2.**
A full Ap cannot be loaded back into an embedder.

**EMB-19 · open · Annotations say `NDArray` where lists are accepted.**
The inputs themselves are inconsistent: `_forward_map`, `grad_param_to_x`, `hess_param_to_x`, `drift_matrix_param_grad` and `drift_matrix_x_grad` accept lists for every embedder, and so does every method of gen2x2, but `to_primitive` and `to_conventional` of `prony`, `pronycos` and `multi`, and `prony.compute_drift_matrix`, raise `AttributeError`. Left for the structural simplification of `_base.py` and `multi.py` (maintainer, 2026-11-05). The other docstring and message corrections are in `557a9f3`.

**EMB-21 · open · gen2x2 `from_matrix` rejects some admissible kernels.**
It rejects any 2×2 block A whose symmetric part is indefinite, even when the kernel is admissible (bounded, with J ≥ 0) and the maximal gauge would represent it, for example θ = (1, 1)/√2 with A = [[2, 3], [−3, −0.1]]. On 2026-11-05 the maintainer decided to keep rejecting such input. Low priority.

### Tracked in other modules

These affect embedders, but the fix belongs elsewhere:

- `MemorySpectrum` is wrong where A cannot be diagonalized: gen2x2 at critical damping Ω = γ, for every φ, where A = γ[[2, 1], [−1, 0]]. The value was off by 0.28 at γ² − Ω² = 0, but only by 2e-11 at γ² − Ω² = 1e-12. The closed form and `MemoryKernel` are correct. See `glefit/merit/CLAUDE.md`.
- Spectrum fits with gen2x2 can stall near Ω ≈ 0, where Λ(0) = 2γr² sin²φ/Ω² is singular. To fix in `glefit.merit`; see `src/glefit/embedding/TODO.md`.
- The fit Hessian takes absolute finite-difference steps in unmapped parameters, such as r and Ω of gen2x2, so it is less accurate when they are far from 1. See `glefit/merit/CLAUDE.md`.
- `glefit.utils.numderiv.jacobian`, which builds the Hessian used in fits, raises `TypeError` for an unsupported `order`. See `glefit/utils/CLAUDE.md`.
- `tutorials/` still pass `sigma`, `threshold` and `Omega` to gen2x2, which raise `TypeError`.

### Resolved

- **EMB-1** · `290c7d3` · The gen2x2 gradient for α was zero at α = 0.
- **EMB-2** · `290c7d3` · gen2x2 kernels were wrong for integer times.
- **EMB-3** · `66bd58d` · Integer input to the parameter setters was truncated in x.
- **EMB-4** · `19d41c5` · `MultiEmbedder.to_primitive` raised `AttributeError`.
- **EMB-5** · `290c7d3`, `4755501` · The gen2x2 minimal gauge takes θ = (r, 0) at cos 2φ = 0.
- **EMB-6** · `6932b74` · `MultiEmbedder.grad_A` could be out of date after a component was edited directly.
- **EMB-7** · `290c7d3` · gen2x2 gradients were infinite or NaN at |α| = Γ.
- **EMB-8** · `290c7d3` · The gen2x2 spectrum had no second Γ-derivative at Γ = 0.
- **EMB-9** · `290c7d3` · Stable gen2x2 inputs could be rejected.
- **EMB-10** · `7d4463c` · A starting value on a mapper bound crashed fits; the bounded mappers now reject it.
- **EMB-11** · `ae075a0` · `from_dict` ignored unknown keys; it now raises `TypeError`, but allows `mappers`.
- **EMB-13** · `9d29a67` · `to_primitive` lost relative accuracy in g.
- **EMB-16** · `4a5b5f3` · `TwoAuxEmbedder.kernel_hess` was not implemented.
- **EMB-17** · `baba304` · `MultiEmbedder` now rejects a `mappers` argument.
- **EMB-20** · `3985b1b`..`4f7c107` · gen2x2 fits (r, γ, Ω, φ) in the maximal gauge, and `optimized_Ap_matrix.json` holds the minimal gauge.
- **EMB-22** · `9291892` · gen2x2 `from_matrix` rejected some semi-definite A through rounding.
