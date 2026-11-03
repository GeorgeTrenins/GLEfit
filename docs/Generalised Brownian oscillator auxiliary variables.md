---
tags:
  - topicNote
  - stub
aliases: []
date: 2026-10-06
---
# Definition

The most general [[Auxiliary variables (SDE) - Kantorovich prescription|Kantorovich mapping]] onto two [[Auxiliary variables (SDE)|auxiliary variables]] for numerical propagation of the [[Generalised Langevin equation|GLE]]. It approximates the non-Markovian dynamics with
$$
\begin{pmatrix}
\dot{p}               \\
\dot{s}_1             \\
\dot{s}_2           
\end{pmatrix} = 
-\begin{pmatrix}
0    &  \theta_1  & \theta_2 \\
-\theta_1 &  \gamma+\delta & \Omega \\
-\theta_2    & -\Omega & \gamma-\delta
\end{pmatrix}
\begin{pmatrix}
    p               \\
    s_1             \\
    s_2 
\end{pmatrix} 
+
\begin{pmatrix}
0    & 0 \\
0 &    \boldsymbol{B}
\end{pmatrix}
\begin{pmatrix}
0               \\
\boldsymbol{\xi}
\end{pmatrix},
\qquad \gamma \geq \lvert\delta\rvert .
$$
Defining
$$
r^2 = \theta_{1}^{2} + \theta_{2}^{2}, \quad 
\alpha = \delta(\theta_1^2-\theta_2^2)/r^2, \quad 
g \equiv \delta^2-\Omega^2,
$$
the corresponding [[Memory kernel|memory kernel]] is
$$
K(t) = r^2 e^{-\gamma\lvert t\rvert}\left[\cos(\bar\omega t) - \frac{\alpha}{\bar\omega}\sin(\bar\omega\lvert t\rvert)\right],
\qquad
\bar\omega = \sqrt{-g} = \sqrt{\Omega^2-\delta^2}
$$
and the [[Spectral density|spectral density]] is
$$
J(\omega) = r^2\omega\,\frac{(\gamma+\alpha)\,\omega^2 + (\gamma-\alpha)(\gamma^2-g)}{\gamma^4 + 2\gamma^2(\omega^2-g) + (\omega^2+g)^2}
$$
where
- $\gamma$ is the mean decay rate of the kernel
- $g<0$ (underdamped): the kernel is a damped oscillation at frequency $\bar\omega$
- $g>0$ (overdamped): $\bar\omega$ is imaginary and $\cos$, $\sin$ become $\cosh$, $\sinh$ of $\Gamma t$ with $\Gamma = \sqrt{g}$, so the kernel is a sum of two exponentials with decay rates $\gamma\pm\Gamma$
- $g=0$ (critical): the bracket becomes $1-\alpha\lvert t\rvert$
- $\alpha$ sets the asymmetry between modes: $K'(0) = -r^2(\gamma+\alpha)$

# Admissibility and generality

The embedding is realisable if and only if
$$
\gamma \geq \max\!\left(\sqrt{\max(g,0)},\ \lvert\alpha\rvert\right),
$$
which is also the condition for $K(t)$ to be stable and $J(\omega)\geq 0$. The embedding covers **all** two-mode kernels with a non-negative spectral density.


> [!HINT] Unconstrained fitting
> Define $\gamma = \max(\sqrt{\max(g,0)}, \lvert\alpha\rvert) + \lambda$ and fit for $\lambda \geq 0$, $r>0$, $g\in\mathbb{R}$, $\alpha\in\mathbb{R}$.

# Inverse map

The kernel parameters $(r, g, \alpha, \gamma)$ fix the embedding only up to a gauge. The embedding parameters must satisfy
$$
\theta_1^2+\theta_2^2 = r^2, \qquad
\delta\,(\theta_1^2-\theta_2^2) = r^2\alpha, \qquad
\delta^2-\Omega^2 = g, \qquad
\lvert\delta\rvert \leq \gamma .
$$
All solutions form a one-parameter family in $\delta$, with
$$
\max\!\left(\sqrt{\max(g,0)},\ \lvert\alpha\rvert\right) \leq \lvert\delta\rvert \leq \gamma,
$$
$$
\Omega = \pm\sqrt{\delta^2-g}, \qquad
\theta_1 = \pm\, r\sqrt{\tfrac{1}{2}\left(1+\alpha/\delta\right)}, \qquad
\theta_2 = \pm\, r\sqrt{\tfrac{1}{2}\left(1-\alpha/\delta\right)},
$$
with all signs independent. If $\alpha = 0$ and $g \leq 0$, $\delta = 0$ is also allowed. Then $\theta_1 = r\cos\phi$, $\theta_2 = r\sin\phi$ for any $\phi$.

Common gauge choices:
- minimal $\lvert\delta\rvert$, i.e. $\delta = \max(\sqrt{\max(g,0)}, \lvert\alpha\rvert)$: the two auxiliary variables have the most similar damping rates (table below)
- maximal $\delta = \gamma$: $s_2$ is undamped, which gives the 2×2 block form of Li et al.

Choosing the solution that minimises $\lvert\delta\rvert$ and $\lvert\Omega\rvert$:

| Condition | $\delta$ | $\Omega$ | $(\theta_1,\ \theta_2)$ |
|---|---|---|---|
| $g>0$ and $\lvert\alpha\rvert < \Gamma$ | $\Gamma$ | $0$ | $r\left(\sqrt{\tfrac{1}{2}(1+\alpha/\Gamma)},\ \sqrt{\tfrac{1}{2}(1-\alpha/\Gamma)}\right)$ |
| otherwise | $\lvert\alpha\rvert$ | $\sqrt{\alpha^2-g}$ | $(r,\,0)$ if $\alpha\geq 0$, else $(0,\,r)$ |

# Properties

- Static friction $\eta_\mathrm{s} = r^2(\gamma-\alpha)/(\gamma^2-g)$
- High-frequency tail $J(\omega) \sim r^2(\gamma+\alpha)/\omega$, or $J \sim \mathcal{O}(\omega^{-3})$ if $\alpha = -\gamma$

# Special cases

| Case                                               | Parameters                                  |
| -------------------------------------------------- | ------------------------------------------- |
| [[Debye auxiliary variable\|Debye]]                | $g = 0$, $\alpha = 0$ ($s_2$ decouples)     |
| [[Shifted Drude-Lorentz auxiliary variables\|SDL]] | $\alpha = 0$, $g = -\Omega^2 < 0$           |
| [[Brownian oscillator auxiliary variables\|BO]]    | $\alpha = -\gamma$ (boundary $\lambda = 0$) |
| Zero static friction                               | $\alpha = +\gamma$ (boundary $\lambda = 0$) |

# Sources

- [[klippensteinGaussNewton2024|Klippenstein et al. (2024) A Gauss–Newton method for iterative optimization of memory kernels for generalized Langevin thermostats in coarse-grained molecular dynamics simulations]]
- [[liComputingNonMarkovian2017|Li et al. (2017) Computing the non-Markovian coarse-grained interactions derived from the Mori–Zwanzig formalism in molecular systems]]
