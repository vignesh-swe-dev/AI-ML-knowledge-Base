# Module 02: Mathematics for Machine Learning

## 📌 Module Overview
Mathematics is the native language of Machine Learning and Artificial Intelligence.
Modern machine learning algorithms are not arbitrary heuristic recipes; they are precise mathematical systems
formulated to solve optimization problems over multidimensional probability spaces.

A rigorous mastery of mathematics distinguishes an elite machine learning engineer from a practitioner
who merely calls API endpoints. Mathematical literacy provides the foundation to:
1. **Diagnose Training Failures**: Understand why gradients vanish, explode, or oscillate around saddle points.
2. **Design Novel Architectures**: Formulate sound loss functions, custom regularization terms, and inductive biases.
3. **Analyze Computational Complexity**: Reason about memory footprints, tensor contraction ranks, and latency bounds.

---

## 📚 Syllabus & Topic Matrix

| File | Mathematical Domain | Core Concepts | ML Applications |
|---|---|---|---|
| [01-linear-algebra.md](01-linear-algebra.md) | Linear Algebra & Matrix Analysis | Vector spaces, dot products, SVD, Eigendecomposition | Feature spaces, PCA, embeddings, attention mechanisms |
| [02-probability-statistics.md](02-probability-statistics.md) | Probability, Statistics & Info Theory | Bayes' rule, MLE, MAP, Entropy, KL-Divergence | Loss formulations, naive Bayes, generative models |
| [03-calculus-optimization.md](03-calculus-optimization.md) | Multivariate Calculus & Optimization | Gradients, Jacobians, Hessians, SGD, Newton methods | Backpropagation, model training, convergence dynamics |

---

## 🏛️ The Three Pillars of Machine Learning Mathematics

```text
+-------------------------------------------------------------------------------+
|                           Machine Learning Modeling                           |
+-----------------------+-------------------------------+-----------------------+
|     Linear Algebra    |    Probability & Statistics   |  Calculus & Opt       |
|  (How data is shaped) |      (How data is modeled)    | (How models learn)    |
+-----------------------+-------------------------------+-----------------------+
| • Vector Spaces       | • Random Variables            | • Gradients & Jacobians|
| • Matrices & Tensors  | • Maximum Likelihood (MLE)    | • Chain Rule (Autodiff)|
| • Eigendecomposition  | • Bayesian Inference (MAP)    | • Gradient Descent     |
| • SVD & Projections   | • Cross-Entropy & Information | • Convex Optimization  |
+-----------------------+-------------------------------+-----------------------+
```

---

## 📖 Mathematical Notation Standards

Throughout this module and subsequent sections, standard linear algebra and calculus notations apply:

| Symbol | Meaning | Example / Dimensionality |
|---|---|---|
| $x, y, z$ | Scalar value | $x \in \mathbb{R}$ |
| $\mathbf{x}, \mathbf{w}$ | Column vector (bold lowercase) | $\mathbf{x} \in \mathbb{R}^{D \times 1}$ |
| $X, W, A$ | Matrix (uppercase italic) | $X \in \mathbb{R}^{N \times D}$ ($N$ samples, $D$ features) |
| $\mathbf{x}^T$ | Transpose of vector $\mathbf{x}$ | Row vector $\mathbb{R}^{1 \times D}$ |
| $\langle \mathbf{u}, \mathbf{v} \rangle$ or $\mathbf{u}^T \mathbf{v}$ | Standard inner (dot) product | $\sum_{i=1}^D u_i v_i \in \mathbb{R}$ |
| $\|\mathbf{x}\|_2$ | Euclidean ($L_2$) norm | $\sqrt{\mathbf{x}^T \mathbf{x}} = \sqrt{\sum x_i^2}$ |
| $\nabla f(\mathbf{x})$ | Gradient vector of scalar function $f$ | Column vector of partial derivatives |
| $H_f(\mathbf{x})$ | Hessian matrix of second-order partials | Symmetric matrix $\mathbb{R}^{D \times D}$ |
| $\mathbb{E}_{x \sim P}[f(x)]$ | Expected value of $f(x)$ under distribution $P$ | $\int f(x) p(x) dx$ or $\sum f(x) P(x)$ |
| $D_{KL}(P \parallel Q)$ | Kullback-Leibler divergence between $P$ and $Q$ (asymmetric) | Relative entropy measurement |

---

## 💡 Top 5 Technical Interview Questions (Math for ML)

### Q1: What is the geometric interpretation of Singular Value Decomposition (SVD)?
**Answer**: SVD states that any linear transformation matrix $A \in \mathbb{R}^{M \times N}$ can be factored into
three sequential geometric operations: $A = U \Sigma V^T$.
1. $V^T$: An orthonormal rotation/reflection in the input space $\mathbb{R}^N$.
2. $\Sigma$: A coordinate-aligned scaling along principal axes by singular values $\sigma_i$.
3. $U$: An orthonormal rotation/reflection in the output space $\mathbb{R}^M$.
Geometrically, SVD maps an $N$-dimensional unit sphere into an $M$-dimensional hyper-ellipsoid.

### Q2: What is the relationship between Maximum Likelihood Estimation (MLE) and Mean Squared Error (MSE)?
**Answer**: Minimizing Mean Squared Error is mathematically equivalent to maximizing the log-likelihood of the data
under the assumption that target observations $y$ are corrupted by independent and identically distributed (i.i.d.)
additive Gaussian noise: $y = f(\mathbf{x}; \mathbf{w}) + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
Taking the negative log-likelihood of the Gaussian likelihood yields the MSE loss function plus constant terms.

### Q3: Why is the Hessian matrix crucial for second-order optimization, and why isn't it used in deep learning?
**Answer**: The Hessian $H \in \mathbb{R}^{D \times D}$ contains all second-order partial derivatives, describing local
curvature. While second-order methods (like Newton-Raphson) converge in fewer iterations by scaling gradient updates
by $H^{-1}$, computing and inverting $H$ for a model with $D = 10^9$ parameters requires $O(D^2)$ memory and $O(D^3)$
time, which is computationally intractable. Deep learning relies instead on first-order approximations (SGD, AdamW).

### Q4: What does it mean for a loss function to be convex, and why is convexity desirable?
**Answer**: A function $f$ is convex if for all $\mathbf{x}, \mathbf{y}$ and $\alpha \in [0, 1]$:
$$f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \le \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})$$
Convexity guarantees that every local minimum is also a global minimum. Furthermore, for smooth convex functions, gradient descent with appropriate
step sizes converges to a global minimum (which is unique if the function is strictly convex).

### Q5: What is the difference between covariance and correlation?
**Answer**: Covariance $\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$ measures the directional linear association
between two variables, but its magnitude depends on the units/scale of the variables. Pearson correlation $\rho$ standardizes
covariance by dividing by the product of their standard deviations: $\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$,
producing a dimensionless metric bounded strictly in the range $[-1, 1]$.

---

## 📝 Study Checklist & Milestones
- [ ] Understand matrix rank, null space, and orthogonal projection geometry.
- [ ] Compute eigenvalues and eigenvectors analytically and verify with NumPy.
- [ ] Derive the Ordinary Least Squares (OLS) closed-form solution $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$.
- [ ] Derive the gradient of Binary Cross-Entropy loss with respect to model logits.
- [ ] Implement Gradient Descent, Momentum, and Adam optimizers from scratch.
