#!/usr/bin/env bash
# Reproduce every table and figure of the paper.
#
#   bash scripts/download_data.sh          # COCO val2014, annotations, CHAIR synonyms (once)
#   bash scripts/reproduce_paper.sh        # all steps, all models
#
# Environment variables
#   DEVICE           torch device                          (default cuda)
#   RESULTS          output directory                      (default results)
#   MODELS           space-separated subset                (default: the four studied VLMs)
#   STEPS            subset of "1 2 3 4 5 6 7 8"           (default: all)
#   BETA_CHECKPOINT  trained CVL checkpoint (training/) whose Beta posterior drives u
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-cuda}
RESULTS=${RESULTS:-results}
STEPS=${STEPS:-"1 2 3 4 5 6 7 8"}
MODELS=${MODELS:-"llava-1.5-7b qwen3-vl-8b-thinking smolvlm-500m smolvlm-256m"}
DATASETS="pope coco hallusionbench"
common=(--results "$RESULTS" --device "$DEVICE")
generator=()
[ -n "${BETA_CHECKPOINT:-}" ] && generator=(--set "counterfactuals.beta_checkpoint=$BETA_CHECKPOINT")

has() { [[ " $STEPS " == *" $1 "* ]]; }
k_values() {  # counterfactual pool sizes per model (Section 4)
  case "$1" in llava-1.5-7b|qwen3-vl-8b-thinking) echo "50 100 150" ;; *) echo "50" ;; esac
}

for model in $MODELS; do
  if has 1; then
    for ds in $DATASETS; do
      python experiments/01_select_examples.py --model "$model" --dataset "$ds" "${common[@]}"
    done
  fi
  has 2 && python experiments/02_background_means.py --model "$model" "${common[@]}"
  if has 3; then
    for ds in $DATASETS; do
      for K in $(k_values "$model"); do
        python experiments/03_stability.py --model "$model" --dataset "$ds" --K "$K" "${common[@]}" "${generator[@]}"
      done
    done
  fi
done
has 4 && python experiments/04_confidence_control.py --model llava-1.5-7b "${common[@]}"
has 5 && python experiments/05_budget_sensitivity.py "${common[@]}"
has 6 && python experiments/06_timing.py --models $MODELS "${common[@]}"
has 7 && python experiments/07_coverage_simulation.py --results "$RESULTS"
if has 8; then
  python experiments/08_analyze.py --results "$RESULTS"
  python experiments/08_analyze.py --results "$RESULTS" --source reported   # -> $RESULTS/paper/from_reported
fi
echo "Done. Tables and figures: $RESULTS/paper/"
