# LLaVA Hallucination Discovery Pipeline Commands

Run these bash commands sequentially from within the `CD_Circuit` directory to execute the entire end-to-end framework.

### Step 1: Store Offline Mean Activations
Compute the background distributional mean activations across the evaluation split of POPE to act as our expected baseline context.

```bash
python store_mean_activations.py \
    --dataset_name lmms-lab/POPE \
    --dataset_split test \
    --output_path data/mean_activations.pt
```

### Step 2: Extract Hallucinated Samples
Identify exactly where the model hallucinated ("Yes" when Ground Truth is "No") and extract these out with visual metadata to JSON. We use a max_samples to limit how many we want to trace.

```bash
python extract_hallucinations.py \
    --max_samples 50 \
    --output_dir data/hallucinations \
    --dataset_name lmms-lab/POPE
```

### Step 3: Trace the Hallucination Circuits
Launch the Contextual Decomposition (CD-T) continuous tracing over the extracted hallucination samples. It strictly uses the computationally cheap offline means we cached.

```bash
python run_hallucination_circuit.py \
    --dataset_path data/hallucinations/hallucinations.json \
    --mean_activations_path data/mean_activations.pt \
    --top_percentile 95.0 \
    --output circuit_results.json
```

### Step 4: Evaluate Circuit Faithfulness
Compute the actual faithfulness metric by surgically ablating only the nodes `NOT` belonging to the mapped circuit and measuring how perfectly the original hallucination target logit recovers.

```bash
python evaluate_circuit.py \
    --circuit_path circuit_results.json \
    --eval_dataset data/hallucinations/hallucinations.json \
    --mean_activations_path data/mean_activations.pt
```
