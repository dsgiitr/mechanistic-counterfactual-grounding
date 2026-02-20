from experiments.conc_ineq_performance_1b import estimate_true_params
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 5000
m_values = [5, 10, 20, 50, 100, 200, 500]
B = 1.0  # Range is [-B, B]
epsilon = 0.1

# --- Distribution Generators ---
# The new absolute bound for a difference of two [-B, B] variables is 2*B
B_delta = 2 * B 

def get_grounded_delta(T, m, B):
    # Simulates a stable attention head where clean and corrupted states are predictable
    clean = torch.normal(mean=0.5, std=0.1, size=(T, m), device=device)
    corrupted = torch.normal(mean=-0.5, std=0.1, size=(T, m), device=device)
    delta = clean - corrupted
    return torch.clamp(delta, -B_delta, B_delta)

def get_uniform_delta(T, m, B):
    # Maximum variance difference
    clean = (torch.rand((T, m), device=device) * 2 * B) - B
    corrupted = (torch.rand((T, m), device=device) * 2 * B) - B
    return clean - corrupted # Range is naturally [-2B, 2B]

def get_hallucinated_delta(T, m, B):
    cauchy = torch.distributions.Cauchy(loc=0.0, scale=0.5)
    clean = cauchy.sample((T, m)).to(device)
    corrupted = cauchy.sample((T, m)).to(device)
    delta = clean - corrupted
    return torch.clamp(delta, -B_delta, B_delta)

distributions = {
    "Grounded Delta": get_grounded_delta,
    "Uniform Delta": get_uniform_delta,
    "Hallucinated Delta": get_hallucinated_delta
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, (name, dist_func) in enumerate(distributions.items()):
    true_mean, true_var = estimate_true_params(dist_func, B)
    
    emp_probs = []
    hoeffding_bounds = []
    bernstein_bounds = []
    emp_bernstein_bounds = []

    for m in m_values:
        samples = dist_func(T, m, B)
        sample_means = torch.mean(samples, dim=1)
        sample_vars = torch.var(samples, dim=1, unbiased=True) 
        deviations = torch.abs(sample_means - true_mean)
        emp_prob = (deviations > epsilon).float().mean().item()
        emp_probs.append(emp_prob)
        
        hoeffding = 2 * np.exp(-2 * m * (epsilon**2) / ((2*B)**2))
        hoeffding_bounds.append(min(hoeffding, 1.0)) 
        
        bernstein = 2 * np.exp(-m * (epsilon**2) / (2 * true_var + (2/3) * B * epsilon))
        bernstein_bounds.append(min(bernstein, 1.0))
        
        emp_bern_trials = 2 * torch.exp(-m * (epsilon**2) / (2 * sample_vars + (2/3) * B * epsilon))
        emp_bernstein_bounds.append(min(emp_bern_trials.mean().item(), 1.0))

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