from experiments.simulate_delta_difference import B_delta
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 5000
m_values = [5, 10, 20, 50, 100, 200, 500]
B = 1.0 
epsilon = 0.1


def get_grounded_samples(T, m, B):
    samples = torch.normal(mean=0.0, std=0.2, size=(T, m), device=device)
    return torch.clamp(samples, -B, B)

def get_uniform_samples(T, m, B):
    return (torch.rand((T, m), device=device) * 2 * B) - B

def get_hallucinated_samples(T, m, B):
    cauchy = torch.distributions.Cauchy(loc=0.0, scale=0.5)
    samples = cauchy.sample((T, m)).to(device)
    return torch.clamp(samples, -B, B)

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

delta = 0.05 # 95% confidence level

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, (name, dist_func) in enumerate(distributions.items()):
    true_mean, true_var = estimate_true_params(dist_func, B)
    
    hoeffding_widths = []
    emp_bernstein_widths = []

    for m in m_values:

        samples = dist_func(T, m, B)
        sample_vars = torch.var(samples, dim=1, unbiased=True)
        
        # Hoeffding CI Width formula - Epsilon = B * sqrt(2 * ln(2/delta) / m)
        h_width = B_delta * np.sqrt(2 * np.log(2 / delta) / m)
        hoeffding_widths.append(h_width)
        
        # Empirical Bernstein CI Width formula - Epsilon ≈ sqrt(2 * s^2 * ln(2/delta) / m) + (2 * B * ln(2/delta)) / (3 * m)
        term1 = torch.sqrt((2 * sample_vars * np.log(2 / delta)) / m)
        term2 = (2 * B_delta * np.log(2 / delta)) / (3 * m)
        eb_width = (term1 + term2).mean().item() 
        emp_bernstein_widths.append(eb_width)

    ax = axes[i]
    ax.plot(m_values, hoeffding_widths, 'r--', label='Hoeffding CI Width')
    ax.plot(m_values, emp_bernstein_widths, 'g-', linewidth=2, label='Empirical Bernstein CI Width')
    
    ax.set_xscale('log')
    ax.set_title(f"{name}\nTrue $\\sigma^2 \\approx {true_var:.3f}$") 
    ax.set_xlabel("Sample Size $m$ (log scale)")
    if i == 0:
        ax.set_ylabel(f"CI Width ($\epsilon$) at {1-delta:.0%} Confidence")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()