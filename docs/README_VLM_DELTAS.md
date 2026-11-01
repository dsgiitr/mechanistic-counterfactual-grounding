# Activation-difference diagnostics

`run_vlm_extraction.py` captures representation differences between supplied
factual/counterfactual images. These vectors are descriptive diagnostics.
They are **not** the node-wise interventional restoration effects in Eq. 5.

Legacy `analyze_vlm_deltas.py`, `analyze_vlm_three_pairs.py`, and saved reports
apply formula-shaped heuristics across neuron coordinates. Neurons are not iid
counterfactual samples, and observed extrema do not establish a fixed support
bound. These outputs must not be interpreted as confidence intervals or
counterfactual sample-complexity guarantees.

For the corrected restoration-effect extraction and empirical Bernstein
estimator, see [implementation_status.md](implementation_status.md) and
[pipeline_commands.md](pipeline_commands.md). Existing historical assets have
not been recomputed or relabeled as validated results.
