# Probability & Statistics for Machine Learning

## Concept
Machine learning algorithms operate under persistent uncertainty. Uncertainty originates from three fundamental sources:
1. **Inherent Stochasticity**: Real-world processes are inherently non-deterministic.
2. **Incomplete Observability**: Features capture only a partial projection of the underlying state of the world.
3. **Finite Sampling**: Datasets are finite empirical observations drawn from an unknown data-generating distribution.

Probability theory provides the mathematical formalisms to quantify uncertainty, while statistical inference
enables reasoning from observed data back to the underlying population parameters.
Core estimation frameworks—**Maximum Likelihood Estimation (MLE)** and **Maximum A Posteriori (MAP)**—form
the theoretical basis for virtually every supervised loss function in machine learning.

---

## Formula / Core Idea

### 1. Conditional Probability & Bayes' Theorem
For two events $A$ and $B$ where $P(B) > 0$:
$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

In parameter estimation terminology:
$$\underbrace{P(\theta \mid \mathcal{D})}_{\text{Posterior}} = \frac{\overbrace{P(\mathcal{D} \mid \theta)}^{\text{Likelihood}} \overbrace{P(\theta)}^{\text{Prior}}}{\underbrace{P(\mathcal{D})}_{\text{Evidence (Marginal Likelihood)}}}$$

Where:
- **Prior $P(\theta)$**: Prior belief about parameters before observing data $\mathcal{D}$.
- **Likelihood $P(\mathcal{D} \mid \theta)$**: Probability of observing dataset $\mathcal{D}$ given parameters $\theta$.
- **Posterior $P(\theta \mid \mathcal{D})$**: Updated belief about parameters after incorporating evidence.

### 2. Maximum Likelihood Estimation (MLE)
Under the assumption that samples $\mathbf{x}_1, \dots, \mathbf{x}_N$ are independent and identically distributed (i.i.d.):
$$L(\theta) = \prod_{i=1}^N p(\mathbf{x}_i \mid \theta)$$

To prevent numerical underflow with small probabilities, we maximize the **Log-Likelihood** (or minimize Negative Log-Likelihood):
$$\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log p(\mathbf{x}_i \mid \theta)$$

### 3. Maximum A Posteriori (MAP) & Regularization
MAP incorporates prior domain beliefs into parameter estimation:
$$\theta_{\text{MAP}} = \arg\max_\theta \left[ \sum_{i=1}^N \log p(\mathbf{x}_i \mid \theta) + \log P(\theta) \right]$$

**Mathematical Connection to Regularization**:
- A **Gaussian Prior** $P(\theta) = \mathcal{N}(\mathbf{0}, \tau^2 I)$ corresponds exactly to **$L_2$ Regularization (Ridge / Weight Decay)**: $\lambda \|\theta\|_2^2$.
- A **Laplace Prior** $P(\theta) = \frac{1}{2b} \exp\left(-\frac{\|\theta\|_1}{b}\right)$ corresponds exactly to **$L_1$ Regularization (Lasso)**: $\lambda \|\theta\|_1$.

### 4. Information Theory Fundamentals
- **Shannon Entropy**: Measures the average information/uncertainty of a discrete distribution $P$:
  $$H(P) = - \sum_{x} P(x) \log_2 P(x)$$
- **Kullback-Leibler (KL) Divergence**: Measures distribution mismatch from model distribution $Q$ to ground truth $P$:
  $$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$
- **Cross-Entropy**: Loss function used in classification:
  $$H(P, Q) = H(P) + D_{KL}(P \parallel Q) = - \sum_{x} P(x) \log Q(x)$$

---

## When to Use

- **Designing Loss Functions**: Formulating regression losses (Gaussian assumption $\to$ MSE) and classification losses (Bernoulli/Categorical $\to$ Cross-Entropy).
- **Bayesian Modeling**: Quantifying epistemic uncertainty in high-stakes domains (healthcare, autonomous vehicles).
- **A/B Testing & Evaluation**: Conducting statistical significance tests (Student's t-test, Mann-Whitney U, bootstrapping) on model benchmarks.

---

## Example

Below is a complete Python implementation demonstrating **Gaussian MLE** and information-theoretic divergence calculations:

```python
import numpy as np

class GaussianInference:
    """Analytical Maximum Likelihood and Information Theory calculator."""
    
    @staticmethod
    def fit_mle(samples: np.ndarray) -> tuple[float, float]:
        """Analytically computes Maximum Likelihood Estimates for Normal Distribution."""
        N = len(samples)
        # MLE for mean is the sample mean
        mu_mle = np.sum(samples) / N
        # MLE for variance uses divisor N (biased) rather than N - 1
        var_mle = np.sum((samples - mu_mle) ** 2) / N
        return float(mu_mle), float(var_mle)
    
    @staticmethod
    def compute_entropy(probs: np.ndarray, base: float = 2.0) -> float:
        """Computes Shannon Entropy for a discrete probability distribution."""
        # Filter out zero probabilities to avoid log(0)
        p_safe = probs[probs > 0]
        return float(-np.sum(p_safe * (np.log(p_safe) / np.log(base))))
    
    @staticmethod
    def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
        """Computes KL Divergence D_KL(P || Q). Q must have support where P > 0."""
        p_mask = P > 0
        if np.any(Q[p_mask] == 0):
            raise ValueError("Q must have non-zero probability where P > 0.")
        return float(np.sum(P[p_mask] * np.log(P[p_mask] / Q[p_mask])))

# Verification script
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    true_mu, true_var = 5.0, 4.0
    observed_data = rng.normal(loc=true_mu, scale=np.sqrt(true_var), size=10_000)
    
    mu_est, var_est = GaussianInference.fit_mle(observed_data)
    print(f"True Parameters:      mu={true_mu:.4f}, var={true_var:.4f}")
    print(f"MLE Estimates (N=10k): mu={mu_est:.4f}, var={var_est:.4f}")
    
    # Information Theory: Compare uniform vs peaked distribution
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    p_peaked = np.array([0.85, 0.05, 0.05, 0.05])
    
    h_uniform = GaussianInference.compute_entropy(p_uniform)
    h_peaked = GaussianInference.compute_entropy(p_peaked)
    kl_dist = GaussianInference.kl_divergence(p_peaked, p_uniform)
    
    print(f"\nEntropy (Uniform Distribution): {h_uniform:.4f} bits (Max Uncertainty)")
    print(f"Entropy (Peaked Distribution):  {h_peaked:.4f} bits (Low Uncertainty)")
    print(f"KL Divergence D_KL(Peaked || Uniform): {kl_dist:.4f} nats")
```

---

## Key Points & Pitfalls

### 1. KL Divergence is Non-Symmetric
$D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$.
- Minimizing $D_{KL}(P \parallel Q)$ (Forward KL) forces $Q$ to cover all modes of $P$ (zero-avoiding / mean-seeking).
- Minimizing $D_{KL}(Q \parallel P)$ (Reverse KL) forces $Q$ to focus on a single mode (zero-forcing / mode-seeking).
Variational Autoencoders and diffusion models rely on specific directional choices of KL divergence.

### 2. The Difference Between PDF and Probability
For continuous random variables, the Probability Density Function $f(x)$ can exceed 1 (e.g., $\mathcal{N}(0, 0.01)$ at $x=0$).
The probability of any single exact real number is zero: $P(X = c) = 0$.
Probabilities only exist over non-zero intervals: $P(a \le X \le b) = \int_a^b f(x) dx$.

### 3. Log-Sum-Exp Trick for Softmax Stability
When computing normalizations of log-probabilities $\log \sum \exp(z_i)$, exponential terms overflow for large $z_i$.
Subtract the maximum value: $\log \sum \exp(z_i) = c + \log \sum \exp(z_i - c)$, where $c = \max(z_i)$.

---