#!/usr/bin/env bash
# Reproduce every table and figure of the paper.
#
#   bash scripts/download_data.sh          # COCO val2014 + CHAIR synonyms (once)
#   bash scripts/reproduce_paper.sh        # all steps, all models
#
# Environment variables
#   DEVICE   torch device                  (default cuda)
#   RESULTS  output directory              (default results)
#   MODELS   space-separated subset        (default: every model in configs/paper.yaml)
#   STEPS    subset of "1 2 3 4 5 6"       (default: all)
#   SHARDS   shards for step 4 run in parallel on GPUs 0..SHARDS-1 (default 1)
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-cuda}
RESULTS=${RESULTS:-results}
STEPS=${STEPS:-"1 2 3 4 5 6"}
SHARDS=${SHARDS:-1}
STABILITY_MODELS="llava-1.5-7b qwen3-vl-8b-thinking smolvlm-500m smolvlm-256m"
PATCHING_ONLY="qwen3-vl-8b-instruct"
MODELS=${MODELS:-"$STABILITY_MODELS $PATCHING_ONLY"}
DATASETS="pope coco hallusionbench"
common=(--results "$RESULTS" --device "$DEVICE")

has() { [[ " $STEPS " == *" $1 "* ]]; }
in_stability() { [[ " $STABILITY_MODELS " == *" $1 "* ]]; }
k_values() {  # counterfactual pool sizes per model (Section 4.1)
  case "$1" in llava-1.5-7b|qwen3-vl-8b-thinking) echo "50 100 150" ;; *) echo "50" ;; esac
}

for model in $MODELS; do
  if has 1; then
    for ds in $DATASETS; do
      python experiments/01_select_examples.py --model "$model" --dataset "$ds" "${common[@]}"
    done
  fi
  has 2 && python experiments/02_background_means.py --model "$model" "${common[@]}"
  if has 3 && in_stability "$model"; then
    for ds in $DATASETS; do
      for K in $(k_values "$model"); do
        python experiments/03_stability.py --model "$model" --dataset "$ds" --K "$K" "${common[@]}"
      done
    done
  fi
  if has 4; then
    for ds in $DATASETS; do
      if [ "$SHARDS" -gt 1 ]; then
        for i in $(seq 0 $((SHARDS - 1))); do
          CUDA_VISIBLE_DEVICES=$i python experiments/04_benchmark_patching.py --model "$model" --dataset "$ds" \
            --num_shards "$SHARDS" --shard "$i" "${common[@]}" &
        done
        wait
      else
        python experiments/04_benchmark_patching.py --model "$model" --dataset "$ds" "${common[@]}"
      fi
    done
  fi
done
has 5 && python experiments/05_budget_sensitivity.py "${common[@]}"
has 6 && python experiments/06_analyze.py --results "$RESULTS"
echo "Done. Tables and figures: $RESULTS/paper/"
