---
tags:
  - topicNote
  - stub
aliases: []
date: 2026-10-06
---
# Definition

The most general [[Auxiliary variables (SDE) - Kantorovich prescription|Kantorovich mapping]] onto two [[Auxiliary variables (SDE)|auxiliary variables]] for numerical propagation of the [[Generalised Langevin equation|GLE]]. It approximates the non-Markovian dynamics with

$$\tag{1}
\begin{pmatrix}
\dot{p} \\
\dot{s}_1 \\
\dot{s}_2
\end{pmatrix} =
-\begin{pmatrix}
0 & \theta_1 & \theta_2 \\
-\theta_1 & \gamma+\delta & \epsilon \\
-\theta_2 & -\epsilon & \gamma-\delta
\end{pmatrix}
\begin{pmatrix}
p \\
s_1 \\
s_2
\end{pmatrix}
+
\begin{pmatrix}
0 & 0 \\
0 & \boldsymbol{B}
\end{pmatrix}
\begin{pmatrix}
0 \\
\boldsymbol{\xi}
\end{pmatrix},
\qquad \gamma \geq \lvert\delta\rvert .
$$

The [[Memory kernel|memory kernel]] reads
$$
K(t) = 
    e^{-\gamma \lvert t \rvert} \left[ 
    (\theta_1^2 + \theta_2^2)
         \cosh\left(t \sqrt{\delta^2 - \epsilon^2} \right) 
         - (\theta_1^2 - \theta_2^2)\, \delta \, \lvert t \rvert \, \mathrm{sinhc}
            \left(t \sqrt{\delta^2 - \epsilon^2} \right)
        \right]
$$
which we can re-write as
$$\tag{2}
K(t) = r^2 e^{-\gamma\lvert t\rvert}\left[
\cosh\!\left(t \sqrt{\gamma^2 - \Omega^2}\right) - \gamma  \lvert t \rvert  \cos(2\phi)\,\, \mathrm{sinhc}\left(t \sqrt{\gamma^2 - \Omega^2}\right)\right],
$$
expressed in terms of $\gamma$ and the irreducible parameters
$$\tag{3}
r^2 = \theta_1^2 + \theta_2^2, \qquad
\Omega^2 = \gamma^2 - \delta^2 + \epsilon^2, \qquad
\cos 2\phi = \frac{\delta\,(\theta_1^2 - \theta_2^2)}{\gamma\, r^2}.
$$
The [[Spectral density|spectral density]] is
$$
J(\omega) = 2 \gamma r^2 \omega\, \frac{\omega^2 \cos^2\phi + \Omega^2 \sin^2\phi}{(\omega^2 - \Omega^2)^2 + 4\gamma^2\omega^2},
$$

where
- $\mathrm{sinhc}(z) = \sinh(z)/z \text{ if } z \neq 0 \text{ else } 1$
- $\gamma$ is the mean decay rate of the kernel
- $\Omega < \gamma$ describes the overdamped regime
- $\Omega > \gamma$ describes the underdamped regime, where $\cosh(t \sqrt{\gamma^2 - \Omega^2}) = \cos(\bar{\omega} t)$ and $\mathrm{sinhc}(t \sqrt{\gamma^2 - \Omega^2}) = \mathrm{sinc}(\bar{\omega} t)$ with $\bar{\omega} = \sqrt{\Omega^2 - \gamma^2}$
- $\Omega = \gamma$ (critical): the square bracket becomes $1 - \gamma \lvert t\rvert \cos(2\phi)\,$
# Maximal gauge

Setting $\delta = \gamma$ in (3) gives $\epsilon = \Omega$ and $\boldsymbol{\theta} = r(\cos\phi,\ \sin\phi)$, i.e. the drift block
$$
\boldsymbol{A} = \begin{pmatrix} 2\gamma & \Omega \\ -\Omega & 0 \end{pmatrix},
$$
in which $s_2$ is undamped. This is the 2×2 block form of Li et al.

**Claim.** Every kernel realisable by (1) is realised in the maximal gauge, and the realisable kernels are exactly those with $r, \gamma, \Omega \geq 0$ and $0 \leq \phi \leq \pi/2$.

*Proof.* By (2), $K$ is fixed by $(r, \gamma, \Omega, \phi)$. For any embedding, $\gamma \geq \lvert\delta\rvert$ implies $\Omega^2 \geq \epsilon^2 \geq 0$ and, since $\lvert\theta_1^2 - \theta_2^2\rvert \leq r^2$, also $\lvert\cos 2\phi\rvert \leq \lvert\delta\rvert/\gamma \leq 1$. Hence $(r, \gamma, \Omega, \phi)$ lies in the stated set ($\phi$ is arbitrary if $\gamma = 0$). Conversely, for any point of this set the maximal gauge satisfies $\gamma \geq \lvert\delta\rvert$ and is therefore a valid embedding with that kernel. $\blacksquare$

# Minimal gauge

This gauge is preferred for constructing drift matrices used for numerical propagation: the two auxiliary variables have the most similar damping rates and the smallest possible $\epsilon$.

Given $(r, \gamma, \Omega, \phi)$, by (3) a drift matrix of the form (1) reproduces $K$ if and only if
$$
\delta^2 - \epsilon^2 = \gamma^2 - \Omega^2, \qquad
\delta\,(\theta_1^2 - \theta_2^2) = \gamma r^2 \cos 2\phi, \qquad
\theta_1^2 + \theta_2^2 = r^2, \qquad
\lvert\delta\rvert \leq \gamma.
$$
These conditions are invariant under $\epsilon \to -\epsilon$, $\theta_i \to -\theta_i$ and $(\delta, \theta_1, \theta_2) \to (-\delta, \theta_2, \theta_1)$, so we take $\delta, \epsilon, \theta_i \geq 0$ without loss of generality. Real $\epsilon$ and $\theta_i$ then exist if and only if
$$
\max\!\left(\sqrt{\max(\gamma^2 - \Omega^2, 0)},\ \gamma\lvert\cos 2\phi\rvert\right) \leq \delta \leq \gamma,
$$
with
$$
\epsilon = \sqrt{\delta^2 - \gamma^2 + \Omega^2}, \qquad
\theta_{1,2} = r\sqrt{\tfrac{1}{2}\left(1 \pm \gamma\cos(2\phi)/\delta\right)}.
$$
The minimal gauge takes the lower bound. The first argument of the maximum dominates if and only if $\gamma^2 - \Omega^2 > \gamma^2\cos^2 2\phi$, i.e. $\Omega < \gamma \, \lvert \sin(2\phi) \rvert$:

| Condition                                         | $\delta$                             | $\epsilon$                               | $(\theta_1,\ \theta_2)$                                                                                                    |
| ------------------------------------------------- | ------------------------------------ | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| $\Omega < \gamma \, \lvert \sin(2\phi) \rvert$    | $\sqrt{\gamma^2 - \Omega^2}$         | $0$                                      | $r\sqrt{\tfrac12\left(1 \pm \gamma\cos(2\phi)/\delta\right)}$                                                              |
| $\Omega \geq \gamma \, \lvert \sin(2\phi) \rvert$ | $\gamma \, \lvert \cos 2\phi \rvert$ | $\sqrt{\Omega^2 - \gamma^2\sin^2 2\phi}$ | $\left\{ \!\! \begin{array}{ll} (r, 0) & \text{if} \ \cos(2 \phi) \geq 0 \\ (0, r) & \text{otherwise} \end{array} \right.$ |
In the last row, $\theta_1^2 - \theta_2^2 = r^2 \mathrm{sgn}(\cos 2\phi)$. At $\phi = \pi/4$ ($\delta = 0$), $\boldsymbol{\theta}$ can be any vector of length $r$, and $(r, 0)$ is chosen.

# Properties

- Static friction $\eta_\mathrm{s} = 2\gamma r^2 \sin^2\phi / \Omega^2$
- High-frequency tail $J(\omega) \sim 2\gamma r^2\cos^2\phi/\omega$, or $J(\omega) \sim 2\gamma r^2 \Omega^2/\omega^3$ if $\phi = \pi/2$

# Special cases

| Case                                               | Parameters                                                                                                 |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [[Debye auxiliary variable\|Debye]]                | $\Omega = \gamma\, \lvert \sin 2\phi \rvert$, giving $K(t) = r^2 e^{-2\gamma\cos^2\!\phi\,\lvert t\rvert}$ |
| [[Shifted Drude-Lorentz auxiliary variables\|SDL]] | $\phi = \pi/4$, $\Omega > \gamma$ (oscillation frequency $\sqrt{\Omega^2 - \gamma^2}$)                     |
| [[Brownian oscillator auxiliary variables\|BO]]    | $\phi = \pi/2$                                                                                             |
| Zero static friction                               | $\phi = 0$                                                                                                 |

# Sources  

- [[klippensteinGaussNewton2024|Klippenstein et al. (2024) A Gauss–Newton method for iterative optimization of memory kernels for generalized Langevin thermostats in coarse-grained molecular dynamics simulations]]
- [[liComputingNonMarkovian2017|Li et al. (2017) Computing the non-Markovian coarse-grained interactions derived from the Mori–Zwanzig formalism in molecular systems]]
