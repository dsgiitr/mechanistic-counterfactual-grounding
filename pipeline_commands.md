# LLaVA Hallucination Discovery Pipeline Commands

Run these bash commands sequentially from within the `CD_Circuit` directory to execute the entire end-to-end framework.

### Step 1: Store Offline Mean Activations
Compute the background distributional mean activations across the evaluation split of POPE to act as our expected baseline context.

```bash
python store_mean_activations.py \
    --dataset_name lmms-lab/POPE \
    --dataset_split test \
    --max_samples 3000\
    --output_path data/mean_activations.pt
```

### Step 2: Extract Hallucinated Samples
Identify exactly where the model hallucinated ("Yes" when Ground Truth is "No") and extract these out with visual metadata to JSON. We use a max_samples to limit how many we want to trace.

```bash
python extract_hallucinations.py \
    --max_samples 50 \
    --batch_size 16 \
    --output_dir data/hallucinations \
    --dataset_name lmms-lab/POPE
```

### Step 3 Make adjacency matrix

```bash
python run_adjacency_matrix.py \
    --dataset_path data/hallucinations/hallucination_examples.json \
    --mean_activations_path data/mean_activations.pt \
    --batch_size 32 --yes_no \
    --output adjacency_matrix.pt
```

### Step 4: Trace Circuit
```bash
python find_circuit_from_matrix.py --matrix_path adjacency_matrix.pt --num_outliers 2
```

### Step 5: Evaluate Circuit Faithfulness
Compute the actual faithfulness metric by surgically ablating only the nodes `NOT` belonging to the mapped circuit and measuring how perfectly the original hallucination target logit recovers.

```bash
python evaluate_circuit.py \
    --circuit_path circuit_results.json \
    --eval_dataset data/hallucinations/hallucinations.json \
    --mean_activations_path data/mean_activations.pt
```

### Step 5: Visualize Circuit
```bash
python visualize_circuit.py --input circuit_result.json --output circuit_visualization.svg
```