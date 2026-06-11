import os
import torch
import math
from collections import defaultdict

def calc_hoeffding_width(m, delta, B):
    """
    Calculates Hoeffding confidence interval width
    m: sample size (dimension of the layer D)
    delta: failure probability
    B: max absolute value (bound of the interval [-B, B])
    """
    return B * math.sqrt((2 * math.log(2 / delta)) / m)

def calc_empirical_bernstein_width(m, delta, B, sample_variance):
    """
    Calculates Empirical Bernstein confidence interval width
    """
    term1 = math.sqrt((2 * sample_variance * math.log(2 / delta)) / m)
    term2 = (2 * B * math.log(2 / delta)) / (3 * m)
    return term1 + term2

def main():
    delta_dir = "../results/vlm_deltas"
    if not os.path.exists(delta_dir):
        print(f"Directory {delta_dir} not found. Run the extraction script first.")
        return
        
    delta_files = [f for f in os.listdir(delta_dir) if f.endswith(".pt")]
    
    if not delta_files:
        print("No delta files found.")
        return
        
    # Stats dictionaries: layer_name -> list of values across image pairs
    layer_hoeffding = defaultdict(list)
    layer_bernstein = defaultdict(list)
    layer_sample_vars = defaultdict(list)
    
    failure_prob = 0.05 # 95% confidence level
    
    print(f"Analyzing {len(delta_files)} counterfactual pairs...\n")
    
    for d_file in delta_files:
        deltas = torch.load(os.path.join(delta_dir, d_file))
        
        for layer_name, delta_tensor in deltas.items():
            # delta_tensor shape is typically (1, hidden_dim)
            d_flat = delta_tensor.squeeze()
            
            # 1. m = sample size = number of dimensions (neurons) in this layer
            m = d_flat.size(0)
            
            # 2. Determine bounds B. 
            # For simplicity per pair, we take the maximum absolute delta in this layer as our B.
            # In a stricter theoretical setting, B might be the absolute maximum the model CAN produce.
            B = d_flat.abs().max().item()
            
            # 3. Calculate sample variance (s^2)
            # using unbiased sample variance (Bessel's correction)
            sample_var = torch.var(d_flat, unbiased=True).item()
            layer_sample_vars[layer_name].append(sample_var)
            
            # 4. Calculate bounded widths
            h_width = calc_hoeffding_width(m, failure_prob, B)
            b_width = calc_empirical_bernstein_width(m, failure_prob, B, sample_var)
            
            layer_hoeffding[layer_name].append(h_width)
            layer_bernstein[layer_name].append(b_width)

    # 5. Aggregate and print results
    print("=== AVERAGED DELTA DISTRIBUTION BOUNDS (95% Confidence) ===")
    print(f"Sample size (m) is equal to the layer feature dimensionality (D).\n")
    
    for layer_name in sorted(layer_hoeffding.keys()):
        avg_h = sum(layer_hoeffding[layer_name]) / len(layer_hoeffding[layer_name])
        avg_b = sum(layer_bernstein[layer_name]) / len(layer_bernstein[layer_name])
        avg_var = sum(layer_sample_vars[layer_name]) / len(layer_sample_vars[layer_name])
        
        print(f"Layer: {layer_name}")
        print(f"  Avg Sample Variance (s^2): {avg_var:.6f}")
        print(f"  Avg Hoeffding Width (eps)  : {avg_h:.6f}")
        print(f"  Avg Bernstein Width (eps)  : {avg_b:.6f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
