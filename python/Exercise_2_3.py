import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform


def p_pdf(x): #density function med 3 Gaussian 
    return (0.3 * norm.pdf(x, loc=2.0, scale=1.0) +
            0.4 * norm.pdf(x, loc=5.0, scale=2.0) +
            0.3 * norm.pdf(x, loc=9.0, scale=1.0))


def sir_sampling(k, proposal_samples, proposal_pdf, dist_pdf): #proposal_sampler og proposal_pdf er lambda func 
    samples = proposal_samples(k) #generer k samples 
    weights = dist_pdf(samples) / proposal_pdf(samples)
    normalized_weights = weights / np.sum(weights)
    resampled_samples = np.random.choice( #generer random samples fra 1D array
        samples, 
        size=k, 
        replace=True,  #with replacement, et sample kan vælges flere gange
        p=normalized_weights #weights for samples
    )
    
    return resampled_samples

np.random.seed(42) 
k_values = [20, 100, 1000] #antal samples
x_grid = np.linspace(-2, 15, 1000)
p_vals = p_pdf(x_grid) 

fig = plt.plot(figsize=(15, 8), sharey=True) #Vi laver 6 plots på 2 rækker

# Question 1
q1_samples = lambda k: np.random.uniform(0, 15, size=k) #k samples mellem 0 og 15
q1_pdf = lambda x: uniform.pdf(x, loc=0, scale=15) #loc der hvor intervallet starter og scale er intervallets længde

for k in k_values:
    resampled = sir_sampling(k, q1_samples, q1_pdf, p_pdf)
    plt.hist(resampled, density=True, alpha=0.6, color='skyblue', edgecolor='black', label=f'Samples (k={k})')
    plt.plot(x_grid, p_vals, color='red',linestyle='dashed', linewidth=2, label='p(x)')
    plt.title(f'Q1: uniform q(x), k = {k}')
    plt.xlabel('x')
    plt.legend()
    plt.show()

# Question 2
q2_samples = lambda k: np.random.normal(5.0, 4.0, size=k) 
q2_pdf = lambda x: norm.pdf(x, loc=5.0, scale=4.0)

for k in k_values:
    resampled = sir_sampling(k, q2_samples, q2_pdf, p_pdf)
    plt.hist(resampled, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label=f'Samples (k={k})') #density=True gør histogrammets areal 1, så det kan sammenlignes med pdf
    plt.plot(x_grid, p_vals, color='red',linestyle='dashed', linewidth=2, label='p(x)')
    plt.title(f'Q1: Gaussian q(x), k = {k}')
    plt.xlabel('x')
    plt.legend()
    plt.show()
