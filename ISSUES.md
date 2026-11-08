# ISSUES.md

Known defects and limitations of GLEFit, one ID per issue. IDs are never reused. When a commit resolves an issue, move it to "Resolved" as a one-line entry that names the commit. `accepted` marks behaviour that was examined and kept. `deferred` marks a missing feature: adding it is outside the scope of the ongoing refactoring, which only fixes bugs and simplifies code, so it waits for an explicit request.

Only `glefit.embedding` and `glefit.merit` keep their issues here so far. The other modules still list theirs under "Known gaps" in their own `CLAUDE.md`, and move them here when they are revised.

## `glefit.embedding`

### Open

None.

### Deferred features

**EMB-18 · deferred · `from_matrix` exists only for gen2x2.**
A full Ap cannot be loaded back into an embedder. This is a new feature, not a defect, and is outside the scope of the ongoing refactoring.


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
- **EMB-12** · `fix(merit): read θ from the left column of Ap` · Fits read θ only from the top row of Ap, so their values ignored the left column and disagreed with their gradient.
- **EMB-13** · `9d29a67` · `to_primitive` lost relative accuracy in g.
- **EMB-14** · `fix(embedding): keep MultiEmbedder's own list of components` · `MultiEmbedder` failed with an unclear error for a generator or an empty list of components, and shared the caller's list.
- **EMB-15** · `fix(embedding): accept NumPy scalars in the constructors` · Constructors rejected NumPy scalars other than `np.float64`; they now accept any real number and store it at double precision.
- **EMB-16** · `4a5b5f3` · `TwoAuxEmbedder.kernel_hess` was not implemented.
- **EMB-17** · `baba304` · `MultiEmbedder` now rejects a `mappers` argument.
- **EMB-19** · `c05dee4` and the next commit · Every embedder method that takes parameters accepts lists, and is annotated `npt.ArrayLike`.
- **EMB-20** · `3985b1b`..`4f7c107` · gen2x2 fits (r, γ, Ω, φ) in the maximal gauge, and `optimized_Ap_matrix.json` holds the minimal gauge.
- **EMB-22** · `9291892` · gen2x2 `from_matrix` rejected some semi-definite A through rounding.

## `glefit.merit`

Reproduced at `368b598` by running the code.

### Open

**MER-1 · open · Constructors keep the caller's `target` and check `grid` and `target` late.**
The grid is copied but the target is not, so editing the caller's array afterwards changes the distance (24.3 became 26.6). A target of length 1 is applied to every point, and other wrong lengths fail at the first evaluation. A 2-D frequency grid gives values of the wrong shape without an error. An integer `target` fails at `distance`, and an integer time grid at the `MemoryKernel` gradient. The grid need not be increasing, which the default weights of MER-6 will require. Config runs are safe from the length and dtype problems, because the loaders convert to float and check lengths. Agreed fix: store float copies, and raise `ValueError` unless `grid` and `target` are non-empty, 1-D and of equal length, and the grid is strictly increasing.

**MER-2 · open · Constructors ignore extra positional and keyword arguments.**
`BaseProperty.__init__` takes `*args, **kwargs` and never reads them, so a misspelled `wieght=` has no effect.

**MER-3 · open · Weights are not checked.**
Negative weights give a negative distance, NaN weights a NaN distance, and all-zero weights make the distance and its gradient vanish everywhere. A weight array of length 1 is applied to every point. `weight=None`, which `weight: null` in a config gives, raises an unclear `TypeError`, although the annotation is `Optional`. Agreed fix: raise `ValueError` unless the weight is a number or has one entry per grid point, every entry is finite and non-negative, and at least one is positive.

**MER-4 · open · `MemoryKernel` evaluates negative times as given.**
K is even in t, and the embedders' closed forms use |t|. At t = −1, a Prony embedder with θ = 0.8 and γ = 1.3 gave 2.35, against 0.174 from the closed form. `MemorySpectrum` is already even in ω. Agreed fix: evaluate K at |t|.

**MER-5 · open · `from_dict` reports bad entries late or with the wrong error.**
An unknown `metric` raises `KeyError`. A dataset without a grid fails at the first evaluation with an unclear `IndexError`. Unknown keys are ignored, so a misspelled `weight` has no effect. Weights that the constructor rejects raise `ValueError`. Agreed fix: raise `ConfigError` for each.

**MER-6 · open · Whether a fit converges depends on the grid and the scale of the data.**
The distance is a plain sum, Σᵢ wᵢ(hᵢ − yᵢ)², and `gtol` is absolute. With the default weight of 1, doubling the number of grid points doubles the distance and its gradient, and targets 1000 times smaller make both 10⁶ times smaller: with `gtol=1e-4`, such a fit stopped at step 0 with 50% error. Dense parts of a non-uniform grid count for more. Agreed fix: by default, wᵢ = Δᵢ/(L·max|y|²), where L is the range of the grid, Δᵢ = (tᵢ₊₁ − tᵢ₋₁)/2 at interior points, and Δᵢ is the full spacing at each endpoint. A uniform grid of m points then gives 1/((m − 1)·max|y|²) at every point. A single point gets 1/max|y|², and a target that is zero everywhere raises `ValueError`. An explicit `weight` replaces the default.

### Accepted

**MER-7 · accepted · `MemorySpectrum` raises `ValueError` where A + iω is singular.**
Λ is not defined there. This happens with ω = 0 in the grid when A is singular, as for gen2x2 at Ω = 0, which its default mappers exclude.

**MER-8 · accepted · The Hessian's finite-difference step is absolute in unmapped parameters.**
The step is about 7e-4·max(1, |x|), so the units of unmapped parameters matter. All rates are log-mapped: changing the time unit by 1e-3 to 1e2 left the Hessian error of a gen2x2 fit below 4e-9. r and φ of gen2x2 have no mapper, but φ is dimensionless, and K and Λ are quadratic in r, where the stencil is exact.

**MER-9 · accepted · `all()` is expensive.**
It needs 4p + 2 gradient evaluations for p parameters. With 44 parameters and 400 grid points, one call took about 1 s for the spectrum and 13 s for the kernel. Speeding it up is outside the scope of the refactoring.

### Deferred features

**MER-10 · deferred · `squared` is the only metric.**

**MER-11 · deferred · There are no combined objectives.**
K and Λ cannot be fitted together, for example: the config accepts one `merit_function`.

**MER-12 · deferred · Per-point weights cannot be read from a data file.**
They must be listed in the config, as `TODO.md` notes.

**MER-13 · deferred · There is no analytic Hessian.**
