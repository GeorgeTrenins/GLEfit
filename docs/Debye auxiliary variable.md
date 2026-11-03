---
tags:
  - topicNote
  - stub
aliases: []
date: 2026-10-06
---
# Definition

A type of [[Auxiliary variables (SDE)|auxiliary variable]] based on a single-variable [[Auxiliary variables (SDE) - Kantorovich prescription|Kantorovich mapping]] for numerical propagation of the [[Generalised Langevin equation|GLE]] that approximates the non-Markovian dynamics with
$$
\begin{pmatrix}
	\dot{p}               \\
	\dot{s}
\end{pmatrix} = 
-\begin{pmatrix}
0    & a_p \\
-a_p & A
\end{pmatrix}
\begin{pmatrix}
	p               \\
	s
\end{pmatrix} 
+
\begin{pmatrix}
0    & 0 \\
0 &    B
\end{pmatrix}
\begin{pmatrix}
	0               \\
	\xi
\end{pmatrix}.
$$
The corresponding [[Memory kernel|memory kernel]] is
$$
K(t) = a_p^2 \, \mathrm{e}^{-A \lvert t \rvert}
$$
and the [[Spectral density|spectral density]] is
$$
J(\omega) = \frac{\omega a_p^2 A}{A^2 + \omega^2},
$$
which corresponds exactly to the [[Debye spectral density]] with 
 - cut-off frequency $\omega_c = A$,
 - static friction $\eta_0 = a_p^2 / A$.