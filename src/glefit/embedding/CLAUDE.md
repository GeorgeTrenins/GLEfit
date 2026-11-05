# CLAUDE.md: `glefit.embedding`

Embedders build the drift matrix of a Markovian embedding of a generalized Langevin equation from a few parameters. This file lists conventions and pitfalls. Open issues are in `ISSUES.md` at the repository root, as `EMB-n`.

## Model

An embedding with n auxiliary variables is defined by `Ap = [[0, θᵀ], [-θ, A]]`, of size (n+1)×(n+1). It gives K(t) = θᵀ exp(-tA) θ and Λ(ω) = θᵀ A (A² + ω²)⁻¹ θ, and the spectral density is ω·Λ(ω). It is stable when the symmetric part of A is positive definite.

| Config `type` | Class | n | θ | A |
| --- | --- | --- | --- | --- |
| `prony` | `PronyEmbedder` | 1 | [θ] | [γ] |
| `pronycos` | `PronyCosineEmbedder` | 2 | [θ, 0] | [[γ, ω], [-ω, γ]] |
| `gen2x2` | `TwoAuxEmbedder` | 2 | [θ₁, θ₂] | [[γ+δ, ζ], [-ζ, γ-δ]] |
| `multi` | `MultiEmbedder` | sum | concatenated | component blocks on the diagonal |

Config `parameters`: `theta` and `gamma`, plus `omega` for `pronycos`, and `delta` and `zeta` for `gen2x2`, where `theta` is [θ₁, θ₂]. `gen2x2` raises `TypeError` for the removed `sigma` and `threshold`, and for `Omega`, the old name of `zeta`. `multi` takes `embedders`, a list of `{type, parameters}` entries.

## Parameters

- Three sets: conventional (length `nparam`; configs and trajectory files), primitive `p` (length `ndof`; input to `compute_drift_matrix` and `drift_matrix_param_grad`) and mapped `x` (length `ndof`; what the optimizer moves). One `glefit.mappers` mapper per entry converts between p and x. By default p = exp(x), except for r, Ω and φ in gen2x2, which have no bounds. Custom mappers are for debugging only: configs cannot set them, so production runs always use the defaults.
- Set values only through the `conventional_params`, `primitive_params` and `x` properties, which keep all three in step. Their getters return copies.
- Conventional and primitive parameters differ only for gen2x2, which fits [r, γ, Ω, φ] because different (θ₁, θ₂, γ, δ, ζ) can give the same kernel. `to_primitive` is eq. (3) of `docs/Generalised Brownian oscillator auxiliary variables.md`: r² = θ₁² + θ₂², Ω² = γ² − δ² + ζ² and cos 2φ = δ(θ₁² − θ₂²)/(γr²). It raises `ValueError` for θ = 0, γ = 0, γ < |δ| or non-finite input, and accepts γ = |δ|.
- K and Λ depend only on r², Ω² and cos 2φ, so any real r, Ω and φ with γ > 0 are valid. `to_primitive` returns r, Ω ≥ 0 and 0 ≤ φ ≤ π/2.
- `from_matrix(theta, A)` computes [r, γ, Ω, φ] from |θ|², tr A, det A, θᵀAθ and θᵀ adj(A) θ, which do not change when s₁ and s₂ are rotated. It requires the symmetric part S of A to be positive semi-definite up to rounding, det S ≥ −4ε(tr S)², and rejects an indefinite S even when the kernel is admissible (EMB-21).
- gen2x2 values do not round-trip: `conventional_params` returns the maximal gauge δ = γ, θ = r(cos φ, sin φ), ζ = Ω, which has the same kernel and leaves s₂ undamped. [1, 2, 1.5, 0.25, 3] reads back as [1.5, 1.658, 1.5, 1.5, 3.345].
- The gen2x2 kernel has two forms. For G = γ² − Ω² ≤ 0 or √G·t ≤ 1, it is written with trigonometric functions of z = t√(−G), which is imaginary for G > 0, and differentiated through G with `sinc`. Otherwise it is two decaying exponentials, differentiated directly in (γ, Ω, φ). For Ω ≪ γ, the slow rate written as γ − √G, and derivatives taken through G, would both cancel: the code uses Ω²/(γ + √G). In the Debye limit Ω = γ|sin 2φ|, one amplitude vanishes; it is computed in factored form. At Ω = ω = 0, Λ is 0/0.

## Two independent derivative routes

- Closed forms: `kernel_*` and `spectrum_*`, called by `kernel()`, `spectrum()` and `spectral_density()`. Fits never use them.
- Derivatives via chain rule w.r.t. drift matrix: `glefit.merit` computes K, Λ and their gradients from `compute_drift_matrix` and `drift_matrix_param_grad`. Both must use only their argument, never stored state, because `merit` passes trial values.
- A bug in one route does not show in the other, so tests compare them. `merit` reads θ only from the top row of Ap, so test the left column directly (EMB-12).

## Conventions

- Shapes put the parameter axis first: `drift_matrix_param_grad` gives (ndof, n+1, n+1), `*_grad` (ndof, ngrid) and `*_hess` (ndof, ndof, ngrid). `kernel()` and `spectrum()` take `nu` = 0, 1 or 2, with respect to p, or x if `mapped=True`.
- Misleading names: `emb.A` is the full Ap, but `from_matrix(theta, A)` takes the 2×2 block. `emb.grad_A` is with respect to x, `drift_matrix_param_grad` with respect to p.
- Closed forms are even: they use |t| and |ω|.
- `propagation_matrix()` returns the Ap for simulations: the drift matrix by default, the minimal gauge for gen2x2 (the most similar damping rates γ ± δ and the smallest coupling ζ), and the components' matrices for `MultiEmbedder`. Fits never use it. The gen2x2 θ jumps across cos 2φ = 0 when |Ω| > γ, with the same kernel on both sides.
- Constructors: set what `__len__` and `ndof` need, check types, call `kwargs.setdefault("mappers", [...])`, call `super().__init__`, then set the parameters.
- `from_dict` copies its input, pops the required keys and passes the rest to the constructor. `from_harmonic` (pronycos, gen2x2) returns a dict for `from_dict`, not an embedder.
- `MultiEmbedder`: the components hold the parameters and their mappers (passing `mappers` raises `TypeError`), but set values on the `MultiEmbedder`. Slice conventional arrays by `nparam` and primitive arrays by `ndof`.
- `analysis/plot_objective.py` uses the private attributes `_embs` and `_mappers`.
- Style: write `lamda` for λ. Start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## New embedder

Subclass `BaseEmbedder` and implement `from_dict`, `__len__`, `_get_nparam`, `compute_drift_matrix`, `drift_matrix_param_grad`, and the `func`, `grad` and `hess` variants of `kernel_*` and `spectrum_*`. Add the class to `EMBEDDER_MAP` in `__init__.py`. If different parameters can give the same kernel, also override `_get_ndof`, `to_primitive` and `to_conventional`, and `propagation_matrix` if the fitted form is not the one to simulate, as gen2x2 does. A non-symmetric A with more than one θ entry needs the `grad_wrt_A` fix in `glefit/merit` first.

## Testing

Run `PYTHONPATH=src conda run -n tools python -m pytest tests/test_embedding` from the repository root.

- The references are `SympyReference` (closed forms in SymPy) in `tests/test_embedding/conftest.py` and `DriftMatrixReference` (K, Λ and exact directional derivatives of a numerical Ap, in mpmath) in `tests/conftest.py`. Do not use `glefit.utils.numderiv`, which is unreviewed.
- Include branch points and special values, which random values miss: for gen2x2, Ω = γ and Ω ≈ γ, Ω = 0, φ = 0, π/4 and π/2, the Debye limit on both sides of φ = π/4, √G·t on both sides of 1, Ω ≪ γ at long times, and negative r and Ω.
- Rounding in a vanishing term shows only in an element-wise relative comparison, not in `assert_close`, which scales by the largest value in each row.
- The gen2x2 reference writes cosh(t√G) and sinhc(t√G) as ₀F₁ series in G, so SymPy can differentiate them and mpmath can evaluate them at G = 0.
- `DriftMatrixReference` rejects non-finite input, because `mpmath.expm` never returns for NaN.
