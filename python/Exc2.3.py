import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform


def p_pdf(x):
    return (0.3 * norm.pdf(x, loc=2.0, scale=1.0) +
            0.4 * norm.pdf(x, loc=5.0, scale=2.0) +
            0.3 * norm.pdf(x, loc=9.0, scale=1.0))

def sir_sampling(k, proposal_sampler, proposal_pdf, target_pdf):
    samples = proposal_sampler(k)
    weights = target_pdf(samples) / proposal_pdf(samples)
    normalized_weights = weights / np.sum(weights)
    resampled_samples = np.random.choice(
        samples, 
        size=k, 
        replace=True, 
        p=normalized_weights
    )
    
    return resampled_samples

np.random.seed(42) 
k_values = [20, 100, 1000]
x_grid = np.linspace(-2, 15, 1000)
p_vals = p_pdf(x_grid)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)

# Question 1
q1_sampler = lambda k: np.random.uniform(0, 15, size=k)
q1_pdf = lambda x: uniform.pdf(x, loc=0, scale=15)

for i, k in enumerate(k_values):
    resampled = sir_sampling(k, q1_sampler, q1_pdf, p_pdf)
    
    # Bins tilpasses k for bedre visuel repræsentation
    bins = max(5, int(np.sqrt(k)))
    axes[0, i].hist(resampled, bins=bins, density=True, alpha=0.6, 
                    color='skyblue', edgecolor='black', label=f'Samples (k={k})')
    axes[0, i].plot(x_grid, p_vals, 'r-', lw=2, label='$p(x)$')
    axes[0, i].set_title(f'Q1: Uniform q(x), k = {k}')
    axes[0, i].set_xlabel('$x$')
    axes[0, i].legend()

# Question 2
q2_sampler = lambda k: np.random.normal(5.0, 4.0, size=k)
q2_pdf = lambda x: norm.pdf(x, loc=5.0, scale=4.0)

for i, k in enumerate(k_values):
    resampled = sir_sampling(k, q2_sampler, q2_pdf, p_pdf)
    
    bins = max(5, int(np.sqrt(k)))
    axes[1, i].hist(resampled, bins=bins, density=True, alpha=0.6, 
                    color='lightgreen', edgecolor='black', label=f'Samples (k={k})')
    axes[1, i].plot(x_grid, p_vals, 'r-', lw=2, label='$p(x)$')
    axes[1, i].set_title(f'Q2: Gaussian q(x), k = {k}')
    axes[1, i].set_xlabel('$x$')
    axes[1, i].legend()

plt.tight_layout()
plt.show()