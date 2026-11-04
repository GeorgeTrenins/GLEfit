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
K(t) = r^2 e^{-\gamma\lvert t\rvert}\left[
\cosh(t \sqrt{g}) - \alpha\, \lvert t \rvert \mathop{\mathrm{sinhc}}(t \sqrt{g})\right],
$$
and the [[Spectral density|spectral density]] is
$$
J(\omega) = r^2\omega\,\frac{(\gamma+\alpha)\,\omega^2 + (\gamma-\alpha)(\gamma^2-g)}{\gamma^4 + 2\gamma^2(\omega^2-g) + (\omega^2+g)^2}
$$
where
- $\mathop{\mathrm{sinhc}}(z) = \sinh(z)/z \text{ if } z \neq 0 \text{ else } 1$
- $\gamma$ is the mean decay rate of the kernel
- $g > 0$ describes the overdamped regime
- $g<0$ described the underdamped regime where $\cosh(t \sqrt{g}) = \cos(\bar{\omega} t)$ and $\mathop{\mathrm{sinhc}}(t \sqrt{g}) = \mathop{\mathrm{sin}}(\bar{\omega} t)$ where $\bar{\omega} = \sqrt{-g}$.
- $g=0$ (critical): the square bracket becomes $1-\alpha\,\lvert t\rvert$

# Admissibility and generality

The embedding is realisable if and only if
$$
\gamma^2 \geq \max(g, \alpha^2) \ \land \ \gamma \geq 0
$$
which is also the condition for $K(t)$ to be stable and $J(\omega)\geq 0$. The embedding covers **all** two-mode kernels with a non-negative spectral density. We can convert this to an unconstrained optimization problem by introducing a variable $y \in \mathbb{R}$ such that 
$$
\gamma = \sqrt{ \frac{g+\alpha^2}{2} + \sqrt{
	{\left(
		\frac{g-\alpha^2}{2}
	\right)}^2 + \mathrm{e}^{2 y} 
}}
$$

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
\max\!\left(\sqrt{\max(g,0)},\ \lvert\alpha\rvert\right) \leq \delta \leq \gamma,
$$
$$
\Omega = \sqrt{\delta^2-g}, \qquad
\theta_1 =  r\sqrt{\tfrac{1}{2}\left(1+\alpha/\delta\right)}, \qquad
\theta_2 =  r\sqrt{\tfrac{1}{2}\left(1-\alpha/\delta\right)},
$$
If $\alpha = 0$ and $g \leq 0$, $\delta = 0$ is also allowed. Then $\theta_1 = r\cos\phi$, $\theta_2 = r\sin\phi$ for any $\phi$.

Common gauge choices:
- minimal $\delta$, i.e. $\delta = \max(\sqrt{\max(g,0)}, \lvert\alpha\rvert)$: the two auxiliary variables have the most similar damping rates and the smallest possible oscillation frequency. In this case,	$$
	\begin{cases}
	\delta = \sqrt{g}, \quad

\Omega = 0, \quad

\theta_{1,2} = r \sqrt{\frac{1}{2} \left( 1 \pm \!\frac{\alpha}{\sqrt{g}}\right)} & \text{if} \quad g > \alpha^2 \\
\delta = \lvert \alpha \rvert, \quad \Omega = \sqrt{\alpha^2 - g}, \quad ( \theta_1, \, \theta_2 ) = \begin{cases}

(r, \, 0) & \text{if} \ \alpha \geq 0 \\

(0, \, r) & \text{if} \ \alpha < 0
\end{cases} & \text{otherwise}
	\end{cases}
	$$
- maximal $\delta = \gamma$: $s_2$ is undamped, which gives the 2×2 block form of Li et al. In this case,	$$
	\Omega = \sqrt{\gamma^2 - g}, \qquad
\theta_{1,2} = r \sqrt{\frac{1}{2} \left( 1 \pm \frac{\alpha}{\gamma} \right)}.
	$$
The minimal gauge is preferred for constructing drift matrices used for the numerical propagation of GLEs, while the maximal gauge prescription is simplest for numerical optimization of the irreducible parameters that rely on chain rule calculations starting from the "conventional" variables.
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
