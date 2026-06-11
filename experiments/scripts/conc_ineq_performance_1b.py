import torch
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration 
T = 5000
m_values = [5, 10, 20, 50, 100, 200, 500]
B = 1.0  
epsilon = 0.1

# --- Distribution Generators ---
# functions that return a tensor of shape (T, m) clamped to [-B, B]

def get_grounded_samples(T, m, B):
    samples = torch.normal(mean=0.0, std=0.2, size=(T, m), device=device)
    return torch.clamp(samples, -B, B)

def get_uniform_samples(T, m, B):
    return (torch.rand((T, m), device=device) * 2 * B) - B

def get_hallucinated_samples(T, m, B):
    cauchy = torch.distributions.Cauchy(loc=0.0, scale=0.5)
    samples = cauchy.sample((T, m)).to(device)
    return torch.clamp(samples, -B, B)

# --- Helper to estimate True Mean and Variance ---
# since analytic variance of clipped distributions is messy, we estimate true 
# parameters using a massive sample size (10 million).
def estimate_true_params(dist_func, B):
    massive_samples = dist_func(1, 10000000, B)
    true_mean = torch.mean(massive_samples).item()
    true_var = torch.var(massive_samples).item()
    return true_mean, true_var

distributions = {
    "Grounded (Low Variance)": get_grounded_samples,
    "Uniform (Mid Variance)": get_uniform_samples,
    "Hallucinated (Heavy-Tailed)": get_hallucinated_samples
}

# --- Main Simulation Loop ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, (name, dist_func) in enumerate(distributions.items()):
    true_mean, true_var = estimate_true_params(dist_func, B)
    
    emp_probs = []
    hoeffding_bounds = []
    bernstein_bounds = []
    emp_bernstein_bounds = []

    for m in m_values:
        samples = dist_func(T, m, B)  # draw T x m i.i.d samples
        sample_means = torch.mean(samples, dim=1) # compute sample mean and sample variance for each trial
        sample_vars = torch.var(samples, dim=1, unbiased=True) # Bessel's correction
        deviations = torch.abs(sample_means - true_mean) # compute empirical probability P(|mu_hat - mu| > epsilon)
        emp_prob = (deviations > epsilon).float().mean().item()
        emp_probs.append(emp_prob)
        
        # computing bounds 
        hoeffding = 2 * np.exp(-2 * m * (epsilon**2) / ((2*B)**2))
        hoeffding_bounds.append(min(hoeffding, 1.0)) 

        bernstein = 2 * np.exp(-m * (epsilon**2) / (2 * true_var + (2/3) * B * epsilon))
        bernstein_bounds.append(min(bernstein, 1.0))
        
        emp_bern_trials = 2 * torch.exp(-m * (epsilon**2) / (2 * sample_vars + (2/3) * B * epsilon))
        emp_bernstein_bounds.append(min(emp_bern_trials.mean().item(), 1.0))

    # --- Plotting --- gpt gibberish 
    ax = axes[i]
    ax.plot(m_values, emp_probs, 'ko-', label=r'Empirical $P(|\hat{\mu} - \mu| > \epsilon)$')
    ax.plot(m_values, hoeffding_bounds, 'r--', label='Hoeffding')
    ax.plot(m_values, bernstein_bounds, 'b-.', label='Theoretical Bernstein')
    ax.plot(m_values, emp_bernstein_bounds, 'g:', linewidth=2, label='Empirical Bernstein')
    
    ax.set_xscale('log')
    ax.set_title(rf"{name}\nTrue $\sigma^2 \approx {true_var:.3f}$")
    ax.set_xlabel("Sample Size $m$ (log scale)")
    if i == 0:
        ax.set_ylabel("Probability of Deviation")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()