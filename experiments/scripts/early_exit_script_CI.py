from experiments.simulate_delta_difference import get_grounded_delta, get_hallucinated_delta, get_uniform_delta
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def adaptive_sample_tester(dist_generator, B, target_width=0.5, delta=0.05, batch_size=10, max_samples=500):
    """
    Sequentially draws samples until the Empirical Bernstein CI width <= target_width.
    """
    B_delta = 2 * B
    samples = []
    m = 0
    
    while m < max_samples:
        new_batch = dist_generator(1, batch_size, B).squeeze(0)
        samples.append(new_batch)
        m += batch_size

        all_samples = torch.cat(samples)
      
        if m < 2:
            continue
            
        sample_var = torch.var(all_samples, unbiased=True).item()
        
        # empirical Bernstein CI Width
        term1 = np.sqrt((2 * sample_var * np.log(2 / delta)) / m)
        term2 = (2 * B_delta * np.log(2 / delta)) / (3 * m)
        current_ci_width = term1 + term2
        
        if current_ci_width <= target_width:
            sample_mean = torch.mean(all_samples).item()
            return {
                "stopped_at_m": m,
                "final_mean": sample_mean,
                "final_var": sample_var,
                "final_ci_width": current_ci_width,
                "status": "Target Reached"
            }
            
    sample_mean = torch.mean(all_samples).item()
    return {
        "stopped_at_m": m,
        "final_mean": sample_mean,
        "final_var": torch.var(all_samples, unbiased=True).item(),
        "final_ci_width": current_ci_width,
        "status": "Max Samples Hit"
    }


print(f"Target CI Width: 0.5 | Confidence: 95%\n{'-'*45}")

res_grounded = adaptive_sample_tester(get_grounded_delta, B=1.0, target_width=0.5)
print(f"Grounded Token stopped at m = {res_grounded['stopped_at_m']} (Status: {res_grounded['status']})")

res_uniform = adaptive_sample_tester(get_uniform_delta, B=1.0, target_width=0.5)
print(f"Uniform Token stopped at  m = {res_uniform['stopped_at_m']} (Status: {res_uniform['status']})")

res_hallucinated = adaptive_sample_tester(get_hallucinated_delta, B=1.0, target_width=0.5)
print(f"Hallucinated Token stopped at m = {res_hallucinated['stopped_at_m']} (Status: {res_hallucinated['status']})")