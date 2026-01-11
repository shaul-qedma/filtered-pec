# Probabilistic Error Cancellation via Pauli Channel Inversion

## Overview

This document develops Probabilistic Error Cancellation (PEC) for Pauli noise channels using the Fourier transform on the group $\mathbb{Z}_2 \times \mathbb{Z}_2$. We then derive the exponential window filter as a principled modification that trades small bias for dramatic variance reduction.

---

## 1. The Fourier Transform on $\mathbb{Z}_2 \times \mathbb{Z}_2$

The single-qubit Pauli group modulo phase is isomorphic to $\mathbb{Z}_2 \times \mathbb{Z}_2$. We index elements as $s \in \{0, 1, 2, 3\}$ corresponding to $\{I, X, Y, Z\}$.

**Definition.** The *character matrix* of $\mathbb{Z}_2 \times \mathbb{Z}_2$ is:

$$\eta_{s\sigma} = (-1)^{\langle s, \sigma \rangle}$$

where $\langle s, \sigma \rangle$ is the symplectic inner product. Explicitly:

$$\eta = \begin{pmatrix} +1 & +1 & +1 & +1 \\ +1 & +1 & -1 & -1 \\ +1 & -1 & +1 & -1 \\ +1 & -1 & -1 & +1 \end{pmatrix}$$

The entry $\eta_{s\sigma} = +1$ if $P_s$ and $P_\sigma$ commute, and $\eta_{s\sigma} = -1$ if they anticommute.

**Proposition.** The matrix $\eta$ satisfies $\eta^2 = 4I$, so $\tfrac{1}{4}\eta$ is its own inverse.

**Definition.** For a function $f: \{0,1,2,3\} \to \mathbb{R}$, the *Fourier transform* is:

$$\hat{f}(\sigma) = \sum_{s=0}^{3} \eta_{s\sigma} \, f(s) = (\eta \cdot f)_\sigma$$

The *inverse Fourier transform* is:

$$f(s) = \frac{1}{4} \sum_{\sigma=0}^{3} \eta_{s\sigma} \, \hat{f}(\sigma) = \frac{1}{4}(\eta \cdot \hat{f})_s$$

---

## 2. Pauli Channels and Noise Eigenvalues

A single-qubit Pauli channel applies Pauli operators with probabilities $p = (p_0, p_1, p_2, p_3)$:

$$\mathcal{E}(\rho) = \sum_{s=0}^{3} p_s \, P_s \rho P_s$$

where $p_s \geq 0$ and $\sum_s p_s = 1$.

**Proposition.** The Pauli channel acts diagonally on Pauli observables:

$$\mathrm{Tr}[P_\sigma \, \mathcal{E}(\rho)] = (\eta \cdot p)_\sigma \cdot \mathrm{Tr}[P_\sigma \, \rho]$$

*Proof.* Using the identity $P_s P_\sigma P_s = \eta_{s\sigma} P_\sigma$:

$$\mathrm{Tr}[P_\sigma \, \mathcal{E}(\rho)] = \sum_s p_s \mathrm{Tr}[P_\sigma P_s \rho P_s] = \sum_s p_s \eta_{s\sigma} \mathrm{Tr}[P_\sigma \rho] = (\eta \cdot p)_\sigma \mathrm{Tr}[P_\sigma \rho] \quad \square$$

**Definition.** The *noise eigenvalues* are $(\eta \cdot p)_\sigma$ for $\sigma \in \{0,1,2,3\}$.

- $(\eta \cdot p)_0 = 1$ always (trace preservation)
- $|(\eta \cdot p)_\sigma| \leq 1$ for $\sigma \neq 0$, with equality iff $p$ is a point mass (noiseless)

The eigenvalue $(\eta \cdot p)_\sigma$ is the *fidelity* with which the channel preserves observable $P_\sigma$.

---

## 3. The Pauli Path Expansion

Consider a circuit with $n$ error locations, each with noise probabilities $p^{(v)}$ for $v = 1, \ldots, n$. We work in the Pauli-Liouville representation where density matrices and observables are expanded in the Pauli basis.

**Definition.** A *Pauli path* is a tuple $\sigma = (\sigma_1, \ldots, \sigma_n) \in \{0,1,2,3\}^n$ specifying a Pauli index at each error location. The *weight* of a path is $|\sigma| = \#\{v : \sigma_v \neq 0\}$.

**Definition.** The *path amplitude* $\hat{f}(\sigma)$ is defined implicitly by the expansion:

$$\langle O \rangle_{\mathrm{ideal}} = \sum_{\sigma \in \{0,1,2,3\}^n} \hat{f}(\sigma)$$

where $\langle O \rangle_{\mathrm{ideal}}$ is the expectation value of observable $O$ for the noiseless circuit.

**Explicit construction.** In the Pauli-Liouville picture:
- The input state $\rho_{\mathrm{in}}$ has expansion coefficients $r_\alpha = \mathrm{Tr}[P_\alpha \rho_{\mathrm{in}}]$
- Each ideal gate $G$ has transfer matrix $T_{\alpha\beta} = \tfrac{1}{2}\mathrm{Tr}[P_\alpha G P_\beta G^\dagger]$
- The observable $O$ has expansion $O = \tfrac{1}{2}\sum_\alpha o_\alpha P_\alpha$

For a circuit with gates $G_1, \ldots, G_L$ and error locations between them, let $\alpha^{(0)}, \ldots, \alpha^{(L)}$ denote the Pauli indices before and after each gate. The path amplitude is:

$$\hat{f}(\sigma) = \sum_{\substack{\alpha^{(0)}, \ldots, \alpha^{(L)} \\ \text{compatible with } \sigma}} o_{\alpha^{(L)}} \prod_{\ell=1}^{L} T^{(\ell)}_{\alpha^{(\ell)}, \alpha^{(\ell-1)}} \cdot r_{\alpha^{(0)}}$$

where "compatible with $\sigma$" means the Pauli index at each error location matches the corresponding $\sigma_v$.

**Theorem.** The noisy expectation value is:

$$\langle O \rangle_{\mathrm{noisy}} = \sum_{\sigma \in \{0,1,2,3\}^n} \hat{f}(\sigma) \prod_{v=1}^{n} (\eta \cdot p^{(v)})_{\sigma_v}$$

*Proof.* Each noise channel multiplies its Pauli component by the corresponding eigenvalue. $\square$

**Corollary.** Defining the *total eigenvalue* $\Lambda(\sigma) = \prod_v (\eta \cdot p^{(v)})_{\sigma_v}$:

$$\langle O \rangle_{\mathrm{noisy}} = \sum_{\sigma} \hat{f}(\sigma) \, \Lambda(\sigma)$$

For typical noise with $(\eta \cdot p^{(v)})_{\sigma_v} \lesssim \lambda < 1$ when $\sigma_v \neq 0$:

$$\Lambda(\sigma) \lesssim \lambda^{|\sigma|}$$

High-weight paths are exponentially suppressed in the noisy expectation.

---

## 4. Full PEC: Inversion in Fourier Domain

To recover the ideal expectation, we seek a quasi-probability distribution $q(s)$ over Pauli insertions $s \in \{0,1,2,3\}^n$ such that:

$$\sum_{s} q(s) \langle O \rangle_s = \langle O \rangle_{\mathrm{ideal}}$$

where $\langle O \rangle_s$ is the noisy expectation with Pauli $P_{s_v}$ inserted at each location $v$.

**Theorem.** For product noise, the quasi-probability

$$q(s) = \prod_{v=1}^{n} q^{(v)}(s_v)$$

where

$$q^{(v)}(s) = \frac{1}{4} \sum_{\sigma=0}^{3} \frac{\eta_{s\sigma}}{(\eta \cdot p^{(v)})_\sigma} = \frac{1}{4}\left(\eta \cdot \frac{1}{\eta \cdot p^{(v)}}\right)_s$$

satisfies the PEC condition.

*Proof.* The Pauli insertion $P_s$ at location $v$ multiplies the $\sigma$-component by $\eta_{s\sigma}$. With quasi-probability weighting:

$$\sum_s q^{(v)}(s) \eta_{s\sigma} = \frac{1}{4} \sum_s \sum_{\sigma'} \frac{\eta_{s\sigma'} \eta_{s\sigma}}{(\eta \cdot p^{(v)})_{\sigma'}} = \frac{1}{4} \cdot \frac{4\delta_{\sigma'\sigma}}{(\eta \cdot p^{(v)})_\sigma} = \frac{1}{(\eta \cdot p^{(v)})_\sigma}$$

This exactly inverts the noise attenuation at each location. $\square$

**The cost.** When $(\eta \cdot p)_\sigma < 1$, the inversion $1/(\eta \cdot p)_\sigma > 1$ creates negative entries in $q$.

**Definition.** The *sampling overhead* is $\gamma = \|q\|_1 = \sum_s |q(s)|$. For product distributions:

$$\gamma = \prod_{v=1}^{n} \|q^{(v)}\|_1$$

The PEC estimator samples $s \sim |q|/\gamma$ and computes:

$$\hat{O} = \gamma \cdot \mathrm{sign}(q(s)) \cdot \langle O \rangle_s$$

**Proposition.** The variance of the PEC estimator is:

$$\mathrm{Var}[\hat{O}] = \frac{\gamma^2}{N} \cdot \mathrm{Var}[\mathrm{sign}(q) \cdot O]$$

The factor $\gamma^2$ amplifies variance. Since $\gamma_v > 1$ at each location, the total $\gamma$ grows exponentially with $n$.

---

## 5. The Exponential Window Filter

Full PEC inverts all eigenvalues equally, including those for high-weight paths that contribute negligibly to the noisy signal. The exponential window applies a regularized inverse.

**Definition.** The *exponential window filter* with parameter $\beta > 0$ uses:

$$h(\sigma) = \frac{e^{-\beta|\sigma|}}{\Lambda(\sigma)} \quad \text{instead of} \quad \frac{1}{\Lambda(\sigma)}$$

For product noise, this factorizes with local filter:

$$h^{(v)}(\sigma_v) = \begin{cases} 1 & \sigma_v = 0 \\ e^{-\beta} / (\eta \cdot p^{(v)})_{\sigma_v} & \sigma_v \neq 0 \end{cases}$$

The local quasi-probability becomes:

$$q^{(v)}_\beta(s) = \frac{1}{4}\left(\eta \cdot h^{(v)}\right)_s = \frac{1}{4}\left[\eta_{s0} + e^{-\beta} \sum_{\sigma=1}^{3} \frac{\eta_{s\sigma}}{(\eta \cdot p^{(v)})_\sigma}\right]$$

**Theorem.** The exponential window estimator has expectation:

$$\mathbb{E}[\hat{O}_\beta] = \sum_{\sigma} \hat{f}(\sigma) \, e^{-\beta|\sigma|}$$

*Proof.* The filter recovers each Pauli path $\sigma$ with weight $e^{-\beta|\sigma|}$ instead of 1. $\square$

**Corollary.** The bias is:

$$\mathrm{Bias} = \langle O \rangle_{\mathrm{ideal}} - \mathbb{E}[\hat{O}_\beta] = \sum_{\sigma} \hat{f}(\sigma) \left(1 - e^{-\beta|\sigma|}\right)$$

Paths with $|\sigma| = 0$ contribute zero bias. For high-weight paths where $\hat{f}(\sigma) \cdot \Lambda(\sigma)$ is already negligible, the bias contribution is small.

---

## 6. Variance Reduction: The Critical β

**Theorem.** For each location $v$, there exists $\beta_{\mathrm{crit}}^{(v)} > 0$ such that $q^{(v)}_\beta(s) \geq 0$ for all $s$ when $\beta \geq \beta_{\mathrm{crit}}^{(v)}$.

*Proof sketch.* As $\beta \to \infty$, the filter $h^{(v)} \to (1, 0, 0, 0)$, giving $q^{(v)}_\beta \to \tfrac{1}{4}(1,1,1,1)$, a uniform distribution. By continuity, non-negativity holds for sufficiently large $\beta$. $\square$

**Corollary.** When all $q^{(v)}_\beta$ are non-negative:

$$\gamma = \prod_v \|q^{(v)}_\beta\|_1 = \prod_v \sum_s q^{(v)}_\beta(s) = \prod_v 1 = 1$$

The variance amplification factor is exactly 1, regardless of circuit size.

**Approximation.** For symmetric noise with $(\eta \cdot p^{(v)})_\sigma \approx \lambda$ for $\sigma \neq 0$:

$$\beta_{\mathrm{crit}} \approx -\log \lambda$$

For hardware with 5% depolarizing error ($\lambda \approx 0.93$), we have $\beta_{\mathrm{crit}} \approx 0.07$.

---

## 7. The Bias-Variance Tradeoff

With sample budget $N$, the mean squared error decomposes as:

$$\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var} = \mathrm{Bias}^2 + \frac{\gamma^2}{N} \cdot \mathrm{Var}_{\mathrm{raw}}$$

| Method | Bias | $\gamma$ | Variance scaling |
|--------|------|----------|------------------|
| Full PEC ($\beta = 0$) | $0$ | $\prod_v \gamma_v \gg 1$ | $\propto \gamma^2/N$ (exponential in $n$) |
| Exp. window ($\beta \geq \beta_{\mathrm{crit}}$) | $\sum_\sigma \hat{f}(\sigma)(1-e^{-\beta\|\sigma\|})$ | $1$ | $\propto 1/N$ (constant in $n$) |

**Key insight.** The exponential window succeeds because:

1. High-weight paths have $|\hat{f}(\sigma) \Lambda(\sigma)| \ll 1$ in the noisy signal
2. Not recovering them incurs small bias: $(1 - e^{-\beta|\sigma|}) \times (\text{small})$
3. But avoiding their inversion eliminates variance amplification: $\gamma = 1$

The filter's exponential rolloff $e^{-\beta|\sigma|}$ matches the noise's natural suppression $\Lambda(\sigma) \lesssim \lambda^{|\sigma|}$.

---

## 8. Summary

The Fourier transform on the Pauli group diagonalizes noise channels:

$$\text{Noise eigenvalue: } (\eta \cdot p)_\sigma \quad\quad \text{Full PEC filter: } h(\sigma) = \frac{1}{(\eta \cdot p)_\sigma}$$

The Pauli path expansion reveals why full inversion is wasteful:

$$\langle O \rangle_{\mathrm{noisy}} = \sum_\sigma \hat{f}(\sigma) \Lambda(\sigma) \quad \text{where } \Lambda(\sigma) \lesssim \lambda^{|\sigma|}$$

The exponential window regularizes inversion to match this structure:

$$h_\beta(\sigma) = \frac{e^{-\beta|\sigma|}}{(\eta \cdot p)_\sigma}$$

For $\beta \geq \beta_{\mathrm{crit}}$, the quasi-probability becomes a true probability ($\gamma = 1$), eliminating variance amplification at the cost of small, controlled bias.

---

## Appendix: Implementation

```python
import numpy as np

# Character matrix of Z_2 × Z_2
eta = np.array([[+1, +1, +1, +1],
                [+1, +1, -1, -1], 
                [+1, -1, +1, -1],
                [+1, -1, -1, +1]])

def full_pec(p):
    """
    Full PEC quasi-probability.
    
    q = (1/4) η · (1 / (η · p))
    """
    return 0.25 * (eta @ (1.0 / (eta @ p)))

def exp_window(p, beta):
    """
    Exponential window quasi-probability.
    
    h = [1, e^{-β}/λ₁, e^{-β}/λ₂, e^{-β}/λ₃]
    q = (1/4) η · h
    """
    eigenvalues = eta @ p
    h = np.array([1.0 / eigenvalues[0],
                  np.exp(-beta) / eigenvalues[1],
                  np.exp(-beta) / eigenvalues[2],
                  np.exp(-beta) / eigenvalues[3]])
    return 0.25 * (eta @ h)

def gamma(q):
    """Sampling overhead γ = ||q||₁"""
    return np.abs(q).sum()
```