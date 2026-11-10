# ISSUES.md

Known defects and limitations of GLEFit, one ID per issue. IDs are never reused. When a commit resolves an issue, move it to "Resolved" as a one-line entry that names the commit. `accepted` marks behaviour that was examined and kept. `deferred` marks a missing feature: adding it is outside the scope of the ongoing refactoring, which only fixes bugs and simplifies code.

Only `glefit.embedding`, `glefit.mappers`, `glefit.merit` and `glefit.optimization` keep their issues here so far. The other modules still list theirs under "Known gaps" in their own `CLAUDE.md`, and move them here when they are revised.

## `glefit.embedding`

### Open

None.

### Deferred features

**EMB-18 · deferred · `from_matrix` exists only for gen2x2.**
A full Ap cannot be loaded back into an embedder.

**EMB-23 · deferred · `from_dict` does not check that required keys are present.**
`_check_config_keys` rejects unknown keys (EMB-11), but uses the required keys only to decide which keys are allowed. A missing one fails at the first `pop` in `from_dict` with a bare `KeyError`, such as `KeyError: 'theta'`, which names neither the class nor the config section. `MultiEmbedder.from_dict` behaves the same for a missing `embedders`, and for a component without `type` or `parameters`. A fix would raise `TypeError` for missing required keys in `_check_config_keys`, as for unexpected ones, and check each component in `MultiEmbedder.from_dict`. Deferred indefinitely.


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
- **EMB-12** · `e879390` · Fits read θ only from the top row of Ap, so their values ignored the left column and disagreed with their gradient.
- **EMB-13** · `9d29a67` · `to_primitive` lost relative accuracy in g.
- **EMB-14** · `b9d3ce0` · `MultiEmbedder` failed with an unclear error for a generator or an empty list of components, and shared the caller's list.
- **EMB-15** · `2d4ef21` · Constructors rejected NumPy scalars other than `np.float64`; they now accept any real number and store it at double precision.
- **EMB-16** · `4a5b5f3` · `TwoAuxEmbedder.kernel_hess` was not implemented.
- **EMB-17** · `baba304` · `MultiEmbedder` now rejects a `mappers` argument.
- **EMB-19** · `c05dee4` and `77263e5` · Every embedder method that takes parameters accepts lists, and is annotated `npt.ArrayLike`.
- **EMB-20** · `3985b1b`..`4f7c107` · gen2x2 fits (r, γ, Ω, φ) in the maximal gauge, and `optimized_Ap_matrix.json` holds the minimal gauge.
- **EMB-22** · `9291892` · gen2x2 `from_matrix` rejected some semi-definite A through rounding.

## `glefit.mappers`

Reproduced at `19ee5e5` by running the code.

### Open

None.

### Accepted

**MAP-5 · accepted · Bounds hold only up to rounding.**
`inverse` returns p exactly on a nonzero bound once the bound absorbs the increment: `LowerBoundMapper(a).inverse(x)` returns a once eˣ < |a|·ε/2, that is, for x < ln|a| − 36.7, and `IntervalMapper().inverse(x)` returns 1 for x > 36.7. With a = 0, as in every default mapper, this needs x < −745. `LowerBoundMapper` and `UpperBoundMapper` overflow to ±∞ for x > 709, whatever the bound. No double lies between a bound and the next double, so no rewrite of `inverse` avoids this.

**MAP-6 · accepted · Gradients vanish near bounds.**
dp/dx falls like e^(−|x|), to about 5e-5 at |x| = 10 and 2e-9 at |x| = 20, so a parameter pushed against its bound barely moves. This is how the exponential maps behave, and changing them would change every fit.

**MAP-7 · accepted · Mappers cannot be set from a config.**
This is by design: custom mappers are for debugging only, so fits always use each embedder's defaults. There is no registry and no `from_dict`.


### Resolved

- **MAP-1** · `c9bbf2d` · `forward` accepted NaN and infinite p; it now raises `ValueError`.
- **MAP-2** · `53e3bd8` · Bounds had to be Python `int` or `float`; they now accept any real number and raise `TypeError` otherwise.
- **MAP-3** · `53e3bd8` · Bounds must be finite, and `IntervalMapper` requires a < b instead of sorting its bounds.
- **MAP-4** · `068110c` · The `_base.py` header called p conventional, and `intervals.py` had an unused `__main__` block.

## `glefit.merit`

Reproduced at `368b598` by running the code.

### Open

None.

### Accepted

**MER-7 · accepted · `MemorySpectrum` raises `ValueError` where A + iω is singular.**
Λ is not defined there. This happens with ω = 0 in the grid when A is singular, as for gen2x2 at Ω = 0, which its default mappers exclude.

**MER-8 · accepted · The Hessian's finite-difference step is absolute in unmapped parameters.**
The step is about 7e-4·max(1, |x|), so the units of unmapped parameters matter. All decay rates are log-mapped, so these are not an issue. r and φ of gen2x2 have no mapper, but φ is dimensionless, and K and Λ are quadratic in r, where finite difference is exact.

**MER-9 · accepted · `all()` is expensive.**
It needs 4p + 2 gradient evaluations for p parameters. With 44 parameters and 400 grid points, one call took about 1 s for the spectrum and 13 s for the kernel.

### Deferred features

**MER-10 · deferred · `squared` is the only metric.**

**MER-11 · deferred · There are no combined objectives.**
K and Λ cannot be fitted together, for example: the config accepts one `merit_function`.

**MER-12 · deferred · Per-point weights cannot be read from a data file.**
They must be listed in the config, as `TODO.md` notes.

**MER-13 · deferred · There is no analytic Hessian.**

### Tracked in other modules

- `tutorials/fixtures/spectrum_optimization_config_template.json` sets `gtol: 0.01`, chosen when the default weight was 1. It needs retuning when `tutorials/` is revised.

### Resolved

- **MER-1** · `88ae373` · Constructors store float copies of grid and target, and check their shapes and the grid's order.
- **MER-2** · `5ae173d` · `BaseProperty.__init__` dropped `*args` and `**kwargs`, so unknown arguments raise `TypeError`.
- **MER-3** · `0bacc2a` · Weights must be finite, non-negative, not all zero, and scalar or per point; `None` means default.
- **MER-4** · `996c53d` · `MemoryKernel` evaluates K and its gradient at |t|, like the closed forms.
- **MER-5** · `3fcdcbf` · `from_dict` raises `ConfigError` for bad entries; the constructor, `ValueError` for unknown metrics.
- **MER-6** · `bf8ace7` · Default weights are Δᵢ/(L·max|y|²), with full spacing at the ends; an explicit `weight` overrides.

## `glefit.optimization`

Covers the optimizers and the driver `optimize.py`. Reproduced at `a6bcabb` by running the code.

### Open

**OPT-8 · open · The log is appended to.**
A second run in the same directory appends to the log, but overwrites the trajectory and `optimized_Ap_matrix.json`. Agreed fix: overwrite the log as well.

**OPT-9 · open · Outdated text in the code.**
`NewtonRaphson` is called "quasi-Newton", although it uses the full finite-difference Hessian. The `run` docstrings refer to `DEFAULT_MAX_STEPS`, but the constant is `MAX_ITERATIONS_DEFAULT`. `optimize.py` imports `load_data` without using it.

**OPT-10 · open · Final parameters are kept only to 7 significant digits.**
They are written only to the trajectory, and cannot be loaded back into an embedder. Agreed fix: embedders get a `to_dict()` that returns `{"type", "parameters"}`, like a config's `embedder` section, and `optimize.py` always pickles it at the end of a run, converged or not.

**OPT-15 · open · EF steps at random along a mode whose curvature is a rounding error.**
A parameter that has no effect gives H a zero row and column. Unless it is the first or last parameter, `eigh` usually returns its eigenvalue as about ±1e-16 rather than 0: in 200 random 4×4 Hessians with the second parameter flat, never exactly 0. EF's step along a mode depends only on r = gₖ/bₖ, as |hₖ| = 2|r|/(1 + √(1 + 4r²)), so two rounding errors give a step anywhere between 0 and 1 in x, with a random sign, at every iteration: a median of 0.66 in those 200 Hessians, with the gradient along the parameter exactly 0. The distance and the convergence test are not affected, but the parameter ends at an arbitrary value. A threshold on gₖ cannot catch this: `eigh` computes each eigenvector only to an angle of about ε·max|b| divided by the gap to the nearest eigenvalue, so next to a mode of curvature 1e-8, gₖ reached 2e7·ε‖g‖. Eigenvalues are accurate to ε·max|b| whatever the gap, so a possible fix is hₖ = 0 where |bₖ| ≤ τ·max|b|. For least squares, a direction with a real gradient has positive curvature near a fit, so this should not block progress. τ must match the noise of the finite-difference Hessian, which is far above ε and not yet measured. Plan: investigate thoroughly after the other optimization fixes, and fix before the final refactor of the module.

### Deferred features

**OPT-11 · deferred · Restarts are not implemented.**
`run` keeps a step count for them, and `restart` in the config is validated but ignored.

**OPT-12 · deferred · There is no line search or trust region.**
Only `max_step` limits a step.

**OPT-13 · deferred · Convergence is tested only on the gradient.**
It cannot be tested on the step size or the change in distance.

**OPT-14 · deferred · There is no quasi-Newton optimizer.**
Every step needs the finite-difference Hessian, at 4p + 2 gradient evaluations (MER-9).

### Resolved

- **OPT-1** · `fix(optimization): log the current Hessian eigenvalues in EF` · EF logged the eigenvalues of the previous step's Hessian.
- **OPT-2** · `fix(optimization): return the convergence flag from run` · `NR.run` and `EF.run` returned `None`; they now return whether the fit converged.
- **OPT-3** · `fix(optimization): reject unknown arguments and options` · Unknown constructor arguments and `run` options were ignored, so a misspelled `gtoll` left the default; both now raise `TypeError`.
- **OPT-4** · `fix(optimization): set default options at construction` · `converged()` and `step()` raised `AttributeError` before `run`; they now use the defaults `gtol = 0.05` and no `max_step`.
- **OPT-5** · `fix(optimization): EF step at a zero Hessian eigenvalue` · The EF step was NaN where `eigh` returned an eigenvalue of exactly 0, as for a first or last parameter without effect, and 0 instead of ±1 where |gₖ/bₖ| exceeded about 1e154.
- **OPT-16** · `fix(optimization): stop at a non-finite merit` · A NaN or ∞ from `merit.all()` turned every parameter into NaN, and the fit failed a step or two later with an unrelated message; it now raises `ValueError` at once, naming the quantity and the step.
- **OPT-6** · `fix(config): check required keys in validate` · A missing `embedder.parameters` or `optimization.options.max_iter` raised `KeyError` in `optimize.py`, and a `max_iter` of 2.5, −1 or `true` was used as given; `validate()` now requires both, with `max_iter` a non-negative integer, and `options` a dictionary.
- **OPT-7** · `fix(optimize): reject unknown types with ConfigError` · An unknown embedder, merit function or optimizer `type` raised `KeyError`, or `TypeError` if it was not a string; `Optimization.from_config` now raises `ConfigError` that lists the known types.
