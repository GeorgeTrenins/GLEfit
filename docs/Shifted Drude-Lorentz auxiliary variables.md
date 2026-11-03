---
tags:
  - topicNote
  - stub
aliases:
  - SDL auxiliary variables
date: 2026-10-06
---
# Definition

A type of [[Auxiliary variables (SDE)|auxiliary variable]] based on a two-variable [[Auxiliary variables (SDE) - Kantorovich prescription|Kantorovich mapping]] for numerical propagation of the [[Generalised Langevin equation|GLE]] that approximates the non-Markovian dynamics with
$$
\begin{pmatrix}
\dot{p}               \\
\dot{s}_1             \\
\dot{s}_2           
\end{pmatrix} = 
-\begin{pmatrix}
0    &  a_p  & 0 \\
-a_p &  \gamma & \omega_0 \\
0    & -\omega_0 & \gamma
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
\end{pmatrix}.
$$
The corresponding [[Memory kernel|memory kernel]] is
$$
K(t) = a_p^2\, e^{-\gamma \lvert t \rvert}\cos(\omega_0 t)
$$
and the [[Spectral density|spectral density]] is
$$
J(\omega) = \frac{a_p^2\gamma}{2}\,\omega\left[\frac{1}{(\omega-\omega_0)^2+\gamma^2} + \frac{1}{(\omega+\omega_0)^2+\gamma^2}\right]
$$
which corresponds exactly to the [[Shifted Drude-Lorentz spectral density|SDL spectral density]] with
- reorganisation energy $\lambda = a_p^2/2$
- width $\gamma$ and shift $\omega_0$ identical to the corresponding entries of the drift matrix
# Sources

- [[stellaGeneralizedLangevin2014|Stella et al. (2014) Generalized Langevin equation]]
- [[treninsNonMarkovianEffects2025|Trenins et al. (2025) Non-Markovian Effects in Quantum Rate Calculations of Hydrogen Diffusion with Electronic Friction]]
- [[morroneEfficientMultiple2011a|Morrone et al. (2011) Efficient multiple time scale molecular dynamics]]


