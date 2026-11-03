---
tags:
  - topicNote
  - stub
aliases:
  - method of auxiliary variables
  - Method of auxiliary variables
  - auxiliary variable
date: 2026-09-23
---
# Definition

A numerical approach to integrating a [[Generalised Langevin equation]]
![[Generalised Langevin equation#^7e2757]]
which consists in mapping the dynamics onto a system of **Markovian** SDEs, extended with auxiliary dynamical variables $\boldsymbol{s}$:
$$\tag{1}
\begin{pmatrix}
	\dot{p}               \\
	\dot{\boldsymbol{s}}
\end{pmatrix} = 
-\begin{pmatrix}
a_{pp}   & \boldsymbol{a}_p^{\top} \\
\bar{\boldsymbol{a}}_p & \boldsymbol{A} 
\end{pmatrix}
\begin{pmatrix}
	p               \\
	\boldsymbol{s}
\end{pmatrix} 
+
\begin{pmatrix}
b_{pp}   & \boldsymbol{b}_p^{\top} \\
\bar{\boldsymbol{b}}_p & \boldsymbol{B} 
\end{pmatrix}
\begin{pmatrix}
	\xi_p               \\
	\boldsymbol{\xi}
\end{pmatrix}.
$$
For brevity, we denote
$$
\boldsymbol{A}_p \equiv
\begin{pmatrix}
a_{pp}   & \boldsymbol{a}_p^{\top} \\
\bar{\boldsymbol{a}}_p & \boldsymbol{A} 
\end{pmatrix},
\qquad
\boldsymbol{B}_p \equiv
\begin{pmatrix}
b_{pp}   & \boldsymbol{b}_p^{\top} \\
\bar{\boldsymbol{b}}_p & \boldsymbol{B} 
\end{pmatrix},
\qquad
\boldsymbol{\xi}_p \equiv
\begin{pmatrix}
	\xi_p               \\
	\boldsymbol{\xi}
\end{pmatrix}.
$$
The elements of $\boldsymbol{\xi}_p$ are time-derivatives of independent standard [[Wiener process]]. Equation (1) is a [[Multivariate Ornstein-Uhlenbeck process|multivariate Ornstein-Uhlenbeck process]] and we can solve for the dynamics of $\boldsymbol{s}$, conditional on some $p(t)$. This yields a one-dimensional GLE in $p(t)$ with
$$
\begin{aligned}
K(t) & = 2 a_{pp} \delta(t) - \boldsymbol{a}_{p}^{\top} \, \mathrm{e}^{-\boldsymbol{A}\lvert t \rvert} \ \bar{\boldsymbol{a}}_{p}, \\
\zeta(t) & = b_{pp} \xi_p(t) + \boldsymbol{b}_p^{\top} \boldsymbol{\xi}(t) -
\int_{-\infty}^{t} \boldsymbol{a}_p^{\top} \, \mathrm{e}^{-\boldsymbol{A}(t-t')}
\left[
\bar{\boldsymbol{b}}_p \xi_p(t') + \boldsymbol{B} \,\boldsymbol{\xi}(t') 
\right] \ \mathrm{d}t.
\end{aligned}
$$
## Noise correlation

The noise correlation
$$
H(t) \equiv \langle \zeta(0) \zeta(t) \rangle
$$
is conveniently expressed in terms of 
$$
\boldsymbol{D}_p = \boldsymbol{B}_p \boldsymbol{B}_p^{\top} 
               = \begin{pmatrix}
               d_{pp} &  \boldsymbol{d}^{\top}_p \\
               \boldsymbol{d}_p & \boldsymbol{D}
\end{pmatrix}
$$
^0198c6

which, in turn, is related to the static covariance matrix $\boldsymbol{C}_p$ (see also the $\boldsymbol{\Sigma}$ in [[Multivariate Ornstein-Uhlenbeck process|multivariate OU process]]) via
$$
\boldsymbol{A}_p \boldsymbol{C}_p + \boldsymbol{C}_p \boldsymbol{A}_p^{\top}  = \boldsymbol{D}_p.    
$$

^5d25e4

The final component is the covariance matrix of a hypothetical OU process based on the $\boldsymbol{s}$ parts of the drift and diffusion matrices only,
$$
\boldsymbol{Z} = \int_0^{\infty} 
\mathrm{e}^{-\boldsymbol{A}t} \, \boldsymbol{D} \
\mathrm{e}^{-\boldsymbol{A}^{\top}t} \ \mathrm{d}t \qquad
\text{i.e.,}
\quad
\boldsymbol{A} \boldsymbol{Z} + \boldsymbol{Z} \boldsymbol{A}^{\top} = \boldsymbol{D}
$$
which is **not to be confused with** $\boldsymbol{C}$.

> [!NOTE] Final expression
> $$
> H(t) = d_{pp} \delta(t) + \boldsymbol{a}_p^{\top} \, 
> \mathrm{e}^{-\boldsymbol{A} \lvert t \rvert} \left(
>     \boldsymbol{Z} \boldsymbol{a}_p - \boldsymbol{d}_p
> \right).
> $$


## Fourier representation

The Fourier transforms $\tilde{f}(\omega) = \int_{-\infty}^{\infty} \mathrm{e}^{-\mathrm{i} \omega t}  f(t) \, \mathrm{d}t$  of the memory kernel and noise correlation are
$$
\begin{aligned}
\tilde{K}(\omega) & = 2 a_{pp} - 2 \boldsymbol{a}_p^{\top} \frac{
\boldsymbol{A}
}{
\boldsymbol{A}^2 + \omega^2
}
\bar{\boldsymbol{a}}_p \\
\tilde{H}(\omega) & = K(\omega) \left(
c_{pp} - \boldsymbol{a}_p^{\top} \frac{
\boldsymbol{A}
}{
\boldsymbol{A}^2 + \omega^2
} \boldsymbol{c}_p
\right) \\
& \quad {}+ 
2 \omega^2 \left(
\boldsymbol{a}_p^{\top} \frac{
1
}{
\boldsymbol{A}^2 + \omega^2
} \boldsymbol{c}_p
\right)
\left(
1 + \boldsymbol{a}_p^{\top} \frac{
1
}{
\boldsymbol{A}^2 + \omega^2}
\bar{\boldsymbol{a}}_p
\right).
\end{aligned}
$$
## Equilibrium dynamics

If we impose the fluctuation-dissipation theorem
![[Generalised Langevin equation#^08ec13]]
then it must be the case that $\boldsymbol{c}_p = \boldsymbol{0}$, $c_{pp} = \beta^{-1}$ and, notably, we are free to choose $\boldsymbol{C} = \beta^{-1} \boldsymbol{I}$.

## Further reading

- [[Auxiliary variables (SDE) - Kantorovich prescription|Kantorovich auxiliary variables]]

# Sources

- Chapter 2.1 and Appendix A in [[ceriottiNovelFramework2010|Ceriotti et al. (2010) A novel framework for enhanced molecular dynamics based on the generalized Langevin equation]]
- [[ceriottiColoredNoiseThermostats2010|Ceriotti et al. (2010) Colored-Noise Thermostats à la Carte]]
- [[treninsNonMarkovianEffects2025|Trenins et al. (2025) Non-Markovian Effects in Quantum Rate Calculations of Hydrogen Diffusion with Electronic Friction]]