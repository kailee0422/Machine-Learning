# Experiment Report

## Introduction

This study explores the mathematical foundation and application of **Poisson
regression** within **Generalized Linear Models (GLMs)** (based on
[CS229 2018 Autumn Problem Set 1](https://github.com/maxim5/cs229-2018-autumn/tree/main/problem-sets-solutions/PS1)).
It shows the Poisson distribution belongs to the exponential family, derives its
canonical response function and stochastic gradient ascent update rule, implements
Poisson regression to predict real-world website traffic, and proves the convexity of
the GLM negative log-likelihood (NLL) loss in general.

## Materials and Methods

**Problem set 3** — Poisson regression:
- **(a)** Express `p(y;λ) = e^(-λ)λ^y / y!` in exponential-family form
  `b(y)·exp(η·T(y) - a(η))`, identifying `b(y)=1/y!`, `η=log λ`, `T(y)=y`, `a(η)=e^η`.
- **(b)** Derive the canonical response function: since `E[y|x;θ]=λ=e^η=e^(θᵀx)`, the
  hypothesis is `h_θ(x) = e^(θᵀx)`.
- **(c)** Differentiate the log-likelihood w.r.t. `θⱼ` to get the stochastic gradient
  ascent rule: `θⱼ := θⱼ + α(y⁽ⁱ⁾ - e^(θᵀx⁽ⁱ⁾))·xⱼ⁽ⁱ⁾`.
- **(d) Coding problem**: implement Poisson regression in [`ML_HW5.ipynb`](ML_HW5.ipynb)
  to predict daily website visitor traffic (`data/ds4_{train,valid}.csv`) — hypothesis
  `h(x)=exp(θᵀx)`, gradient ascent step `θ += step_size/m · Xᵀ(y - h(x))`, iterating
  until `‖step‖₁ < eps` (learning rate `2e-7`).

**Problem set 4** — convexity of GLMs in general (restricting to scalar `η`,
`T(y)=y`, so `p(y;η) = b(y)exp(ηy - a(η))`):
- **(a)** Show `E[Y|X;θ] = ∂a(η)/∂η` (the mean is the gradient of the log-partition
  function).
- **(b)** Show `Var(Y|X;θ) = ∂²a(η)/∂η²` (the variance is its second derivative).
- **(c)** Using (a) and (b), show the Hessian of the GLM's negative log-likelihood is
  `H_jk = Σᵢ Var(Y⁽ⁱ⁾|X⁽ⁱ⁾;θ)·xⱼ⁽ⁱ⁾xₖ⁽ⁱ⁾`, and since variance is always non-negative,
  `zᵀHz ≥ 0` for any `z` — so the NLL loss of any GLM is convex, guaranteeing any local
  minimum is global.

## Results

The trained Poisson regression model's predictions track the website traffic labels
reasonably on both the training and validation sets, though with visible spread at
higher visitor counts (see `ML_HW5.pdf` for the prediction scatter plots).

## Conclusion

The Poisson distribution's exponential-family form yields a simple `exp(θᵀx)` hypothesis
and a gradient-ascent update rule identical in form to ordinary least squares/logistic
regression (just with a different response function) — one of the key practical
payoffs of the GLM framework. More generally, because any exponential-family
distribution's variance equals the second derivative of its log-partition function, the
Hessian of any GLM's NLL loss is always PSD, so GLM training is always a convex
optimization problem regardless of which member of the exponential family is used.

## Code

You can run on colab or local

<a target="_blank" href="https://colab.research.google.com/github/kailee0422/Machine-Learning/blob/main/HW5/ML_HW5.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
