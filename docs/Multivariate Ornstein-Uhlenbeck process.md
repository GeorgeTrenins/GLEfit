---
tags:
  - topicNote
aliases:
  - multivariate Ornstein-Uhlenbeck process
  - multivariate OU process
date: 2026-09-23
---
# Definition

A type of [[Gaussian process]] that is a multidimensional generalisation of the [[Ornstein-Uhlenbeck Process|Ornstein-Uhlenbeck process]], defined by the system of [[Stochastic differential equation|SDE]]s
$$
\mathrm{d} x_i = -\sum_j\left[
	A_{ij} x_j \, \mathrm{d}t + B_{ij} \, \mathrm{d}W_j(t)
\right]
$$
where $W_j(t)$ is a [[Wiener process]].

## Properties

The mean of a multivariate OU process is
$$
\mathbb{E}[\boldsymbol{x}(t)] = \mathrm{e}^{-\boldsymbol{A}t} \, \langle \boldsymbol{x}(0) \rangle
$$
its variance is
$$
\begin{aligned}
\langle \boldsymbol{x}(t), \boldsymbol{x}(t)^{\!\top}  \rangle  & \equiv
\mathbb{E}\Big[\big\{
\boldsymbol{x}(t) - \langle \boldsymbol{x}(t) \rangle \big\}
\big\{
\boldsymbol{x}(t) - \langle \boldsymbol{x}(t) \rangle \big\}^{\top}
\Big]  \\
&  = \boldsymbol{\Sigma} + \mathrm{e}^{-\boldsymbol{A}t} \left[
    \langle 
        \boldsymbol{x}(0), \boldsymbol{x}(0)^{\!\top} 
    \rangle - \boldsymbol{\Sigma}
    \right]
    \mathrm{e}^{-\boldsymbol{A}^{\top}t}
\end{aligned}
$$
and its time-correlation is
$$
\begin{aligned}
\langle \boldsymbol{x}(t), \boldsymbol{x}(s)^{\!\top}  \rangle = 
\begin{cases} 
\mathrm{e}^{-\boldsymbol{A}t} \left[
    \langle 
        \boldsymbol{x}(0), \boldsymbol{x}(0)^{\!\top} 
    \rangle - \boldsymbol{\Sigma}
    \right]
    \mathrm{e}^{-\boldsymbol{A}^{\top}s} 
 +  \mathrm{e}^{-\boldsymbol{A}(t-s)} \boldsymbol{\Sigma}
   & \text{for} \quad  t > s \\
   \mathrm{e}^{-\boldsymbol{A}t} \left[
    \langle 
        \boldsymbol{x}(0), \boldsymbol{x}(0)^{\!\top} 
    \rangle - \boldsymbol{\Sigma}
    \right]
    \mathrm{e}^{-\boldsymbol{A}^{\top}s} 
 +  \boldsymbol{\Sigma} \mathrm{e}^{-\boldsymbol{A}^{\top}(s-t)} 
   & \text{otherwise}
\end{cases}
\end{aligned}
$$
for a symmetric $\boldsymbol{\Sigma}$ that satisfies
$$
\boldsymbol{A} \boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{A}^{\top} = \boldsymbol{B}\boldsymbol{B}^{\top}
$$
which is an instance of the [[Continuous Lyapunov equation|continuous Lyapunov equation]].
# Sources
- [[gardinerHandbookStochastic2002|Gardiner (2002) Handbook of stochastic methods]]
- [[ceriottiNovelFramework2010|Ceriotti et al. (2010) A novel framework for enhanced molecular dynamics based on the generalized Langevin equation]]

