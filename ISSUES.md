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

- Spectrum fits with gen2x2 can stop at two bad points near Ω = 0, which the default log mapper of Ω only keeps fits away from. With ω = 0 in the grid, Ω and φ → 0 at fixed sin φ/Ω turn the second mode into a spike at ω = 0 that fits only that point; the valley is so narrow (Hessian eigenvalues 1e-2 to 1e7) that fits stop there. This can in principle be averted by adding a penalty for the long-time limit of the memory kernel. At Ω = 0, the second mode drops out of Λ at ω > 0, and a single Lorentzian can be a true local minimum, as for the `expohmic_nm_spectrum.csv` fixture (residual 0.98 against 0.31). With Ω unmapped, 6 of 60 spectrum fits stopped at one of them, and none with ln Ω; started next to them, fits with ln Ω still stop there.
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
