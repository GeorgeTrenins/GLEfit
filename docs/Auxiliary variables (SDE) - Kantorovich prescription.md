---
tags:
  - topicNote
  - stub
aliases:
  - Kantorovich auxiliary variables
  - Kantorovich mapping
date: 2026-10-06
---
# Definition

This is a restricted [[Auxiliary variables (SDE)|auxiliary variable parametrisation]] of the [[Generalised Langevin equation]] that maps the non-Markovian dynamics onto a system of Markovian SDEs where the underlying multidimensional [[Wiener process]] has no system component and the [[Memory kernel|memory kernel]] is purely non-Markovian:
$$
\begin{pmatrix}
	\dot{p}               \\
	\dot{\boldsymbol{s}}
\end{pmatrix} = 
-\begin{pmatrix}
0    & \boldsymbol{a}_p^{\top} \\
\bar{\boldsymbol{a}}_p & \boldsymbol{A} 
\end{pmatrix}
\begin{pmatrix}
	p               \\
	\boldsymbol{s}
\end{pmatrix} 
+
\begin{pmatrix}
0    & \boldsymbol{0}^{\top} \\
\boldsymbol{0} & \boldsymbol{B} 
\end{pmatrix}
\begin{pmatrix}
	0               \\
	\boldsymbol{\xi}
\end{pmatrix}.
$$
## Parameter reduction

Under the fluctuation-dissipation theorem
![[Generalised Langevin equation#^08ec13]]
the static covariance matrix $\boldsymbol{C}_p$ of $(p, \, \boldsymbol{s})$ satisfies
![[Auxiliary variables (SDE)#^5d25e4]]
for 
![[Auxiliary variables (SDE)#^0198c6]]
and, furthermore, the covariance matrix
$$
\boldsymbol{C}_p \equiv \begin{pmatrix}
               c_{pp} &  \boldsymbol{c}^{\top}_p \\
               \boldsymbol{c}_p & \boldsymbol{C}
\end{pmatrix}
$$
must satisfy $c_{pp} = \beta^{-1}$ and $\boldsymbol{c}_p = \boldsymbol{0}$. It transpires that the dynamics of the system is independent of $\boldsymbol{C}$, so we may choose without loss of generality to set $\boldsymbol{C} = \beta^{-1} \boldsymbol{I}$. For the specific case of the Kantorovich parametrisation, the relation for the static covariance matrix reads
$$
\boldsymbol{A}_p \boldsymbol{C}_p + \boldsymbol{C}_p \boldsymbol{A}_p^{\top} = 
\frac{1}{\beta}\begin{pmatrix}
0 & \boldsymbol{a}_p^{\top} + \bar{\boldsymbol{a}}_p^{\top} \\
\boldsymbol{a}_p + \bar{\boldsymbol{a}}_p & \boldsymbol{A} + \boldsymbol{A}^{\top}
\end{pmatrix} = \begin{pmatrix}
0 & \boldsymbol{0}^{\top} \\
\boldsymbol{0} & \boldsymbol{B}\boldsymbol{B}^{\top}
\end{pmatrix}
$$

> [!HINT] Result
> $$
> \bar{\boldsymbol{a}}_p = - \boldsymbol{a}_p
> $$

Furthermore, the system dynamics is invariant under orthogonal transformation in the auxiliary variable coordinates, so we may always choose $\boldsymbol{A} + \boldsymbol{A}^{\top}$ to be **diagonal**. Hence, for $k$ auxiliary variables we have $k$, $k$, and $k(k-1)/2$ independent parameters coming from $\boldsymbol{a}_p$, $\boldsymbol{A} + \boldsymbol{A}^{\top}$  and $\boldsymbol{A} - \boldsymbol{A}^{\top}$, respectively, for a total of $k(k+3)/2$.

## Memory kernel

Under these simplifications, the [[Memory kernel|memory kernel]] becomes
$$
K(t) = \boldsymbol{a}_p^{\top} \, \mathrm{e}^{-\boldsymbol{A} \lvert t \rvert} \, \boldsymbol{a}_p^{}
$$
its Fourier transform is 
$$
\tilde{K}(\omega) = 2 \boldsymbol{a}_p^{\top} 
\frac{\boldsymbol{A}}{\boldsymbol{A}^2 + \omega^2}
\boldsymbol{a}_p^{}
$$
and, hence, the [[Spectral density|spectral density]] is
$$
J(\omega) = \omega \, \boldsymbol{a}_p^{\top} 
\frac{\boldsymbol{A}}{\boldsymbol{A}^2 + \omega^2}
\boldsymbol{a}_p^{}.
$$
## Examples

- [[Debye auxiliary variable]]
- [[Shifted Drude-Lorentz auxiliary variables]]
- [[Brownian oscillator auxiliary variables]]
- [[Generalised Brownian oscillator auxiliary variables]]

# Sources

- [[stellaGeneralizedLangevin2014|Stella et al. (2014) Generalized Langevin equation]]
- [[treninsNonMarkovianEffects2025|Trenins et al. (2025) Non-Markovian Effects in Quantum Rate Calculations of Hydrogen Diffusion with Electronic Friction]]
- Appendix A and Chapter 2.1 of [[ceriottiNovelFramework2010|Ceriotti et al. (2010) A novel framework for enhanced molecular dynamics based on the generalized Langevin equation]]
