"""Empirical Bernstein estimation from real, bounded restoration effects.

The support interval must be fixed independently of the observations. The
observed minimum/maximum cannot establish a distribution-wide support bound.
"""

import argparse
import json
import math
from pathlib import Path


def empirical_radius(count, variance, support_width, delta):
    if count < 2:
        raise ValueError("At least two effects are required")
    if not 0 < delta < 1 or support_width <= 0 or variance < 0:
        raise ValueError("Require 0 < delta < 1, positive support width and nonnegative variance")
    # Maurer-Pontil Theorem 4 is one-sided with log(2/delta).
    # Allocate delta/2 to each tail for an absolute-error confidence radius.
    log_term = math.log(4 / delta)
    return (math.sqrt(2 * variance * log_term / count)
            + 7 * support_width * log_term / (3 * (count - 1)))


def estimate_budget(effects, epsilon, delta, effect_min, effect_max):
    """Search observed prefixes using a failure budget covering every prefix.

    delta is the node's failure budget; allocation across nodes happens in
    evaluate_records. No extrapolated or synthetic observations are introduced.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    if not all(map(math.isfinite, [effect_min, effect_max])) or effect_min >= effect_max:
        raise ValueError("A finite, fixed effect_min < effect_max is required")
    if not math.isfinite(effect_max - effect_min):
        raise ValueError("The support width must be finite")
    if len(effects) < 2:
        raise ValueError("At least two real counterfactual effects per node are required")
    if any(not math.isfinite(x) or not effect_min <= x <= effect_max for x in effects):
        raise ValueError("Effects must be finite and inside the supplied fixed support interval")

    # Eq. 14 searches sample sizes. A fixed-K confidence bound alone does not
    # cover that selection; union-bound over all tested prefixes in this file.
    per_check_delta = delta / (len(effects) - 1)
    mean = m2 = 0.0
    budget = None
    radius = None
    at_required_budget = None
    for count, value in enumerate(effects, 1):
        difference = value - mean
        mean += difference / count
        m2 += difference * (value - mean)
        if count >= 2:
            variance = max(0.0, m2 / (count - 1))
            radius = empirical_radius(count, variance, effect_max-effect_min, per_check_delta)
            if budget is None and radius <= epsilon:
                budget = count
                at_required_budget = {"mean": mean, "variance": variance, "radius": radius}
    return {
        "n_observed": len(effects), "mean": mean, "variance": variance,
        "radius_at_observed_budget": radius, "required_budget": budget,
        "status": "resolved" if budget is not None else "insufficient_counterfactuals",
        "delta_per_prefix": per_check_delta,
        "at_required_budget": at_required_budget,
    }


def node_key(node):
    layer, head = int(node["layer"]), int(node["head"])
    position = int(node.get("seq_pos", -1))
    if layer < 0 or head < 0 or position < -1:
        raise ValueError("Invalid circuit node")
    return f"{layer}_{head}" if position == -1 else f"{layer}_{head}_{position}"


def evaluate_records(records, nodes, epsilon, delta, effect_min, effect_max):
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    keys = [node_key(node) for node in nodes]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Circuit must contain nonempty, unique nodes")
    groups = {}
    provenance = {}
    seen = set()
    for record in records:
        example = record.get("example_id", record.get("example"))
        if example is None or "target_token_ids" not in record:
            raise ValueError("Every record needs example_id and target_token_ids")
        target = tuple(record["target_token_ids"])
        if not target or any(type(t) is not int or t < 0 for t in target):
            raise ValueError("target_token_ids must be a nonempty list of token IDs")
        group_key = (str(example), target)
        identity = tuple(record.get(key) for key in (
            "model_id", "model_revision", "circuit_sha256", "generator_checkpoint_sha256",
            "counterfactual_mode", "question", "factual_image_sha256", "answer_prefix",
        )) + (tuple(record.get("answer_prefix_token_ids") or ()),)
        if group_key in provenance and provenance[group_key] != identity:
            raise ValueError("Cannot pool effects from different models, generators, circuits or factual contexts")
        provenance[group_key] = identity
        if "counterfactual_index" not in record:
            raise ValueError("Every effect record needs counterfactual_index")
        if type(record["counterfactual_index"]) is not int or record["counterfactual_index"] < 0:
            raise ValueError("counterfactual_index must be a nonnegative integer")
        sample_key = (group_key, record["counterfactual_index"])
        if sample_key in seen:
            raise ValueError("Duplicate counterfactual observation")
        seen.add(sample_key)
        effects = record.get("node_patch_deltas")
        if not isinstance(effects, dict) or set(effects) != set(keys):
            raise ValueError("Missing or mismatched node_patch_deltas; run exact restoration extraction")
        group = groups.setdefault(group_key, {key: [] for key in keys})
        for key in keys:
            group[key].append(float(effects[key]))
    if not groups:
        raise ValueError("No restoration-effect observations found")

    results = []
    # Family-wide confidence covers all nodes/examples and all tested prefixes.
    delta_per_node = delta / (len(keys) * len(groups))
    for (example, target), group in groups.items():
        estimates = {
            key: estimate_budget(values, epsilon, delta_per_node, effect_min, effect_max)
            for key, values in group.items()
        }
        budgets = [item["required_budget"] for item in estimates.values()]
        resolved = all(value is not None for value in budgets)
        results.append({
            "example_id": example, "target_token_ids": list(target),
            "m_bar": sum(budgets)/len(budgets) if resolved else None,
            "status": "resolved" if resolved else "insufficient_counterfactuals",
            "nodes": estimates,
        })
    return {"examples": results, "parameters": {
        "epsilon": epsilon, "delta": delta, "effect_min": effect_min,
        "effect_max": effect_max, "confidence_scope": "all supplied nodes, examples and observed prefixes",
        "assumptions": "fixed support and maximum budget chosen before sampling; iid counterfactuals",
    }}


def run_bernstein_evaluation(args):
    with open(args.circuit_path) as stream:
        nodes = json.load(stream).get("circuit", [])
    with open(args.behavior_jsonl) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    result = evaluate_records(records, nodes, args.epsilon, args.delta, args.effect_min, args.effect_max)
    output = Path(args.output) if args.output else Path(args.circuit_path).with_name(
        Path(args.circuit_path).stem + "_bernstein_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved estimation results to {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Empirical Bernstein estimation from exact restoration effects")
    parser.add_argument("--circuit_path", required=True)
    parser.add_argument("--behavior_jsonl", required=True)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--effect_min", type=float, required=True, help="Known lower support bound, fixed before sampling")
    parser.add_argument("--effect_max", type=float, required=True, help="Known upper support bound, fixed before sampling")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        run_bernstein_evaluation(args)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
