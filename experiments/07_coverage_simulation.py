"""Step 7: coverage simulations of the Bernstein budgets (Section 5.2, Appendix B). No model needed.

For each variance in `variances` and three bounded laws on [0, B] with that
variance (two-point on the endpoints, symmetric Beta, Beta with mean 0.15 when
feasible) and `reps` replications:
  (a) true-variance budget K* (Eq. 4): coverage Pr(|mean - mu| <= eps)
  (b) plug-in budget: sigma^2 estimated from a pilot of n0 draws, K_hat from
      Eq. 4, mean of K_hat fresh draws
  (c) simultaneous: `nodes` independent nodes at per-node delta, all within eps
  (d) misspecified range: Student-t3 and a 98/2 Gaussian scale mixture (outlier
      scale 6x) with the same variance, budgets still computed with B; also the
      fraction of pilots whose observed range exceeds B

Output  results/paper/coverage_simulation.{json,md}
"""
import math
from pathlib import Path

import numpy as np

from common import config, parser

from mcg.bounds import bernstein_budget
from mcg.utils import write_json

VARIANCES = [0.018, 0.021, 0.023, 0.027, 0.044, 0.062, 0.073, 0.083]


def sampler(law, var, B):
    """Return draw(rng, shape) -> samples on [0, B] with variance var, and the mean; None if infeasible."""
    v = var / B ** 2
    if law == "two-point":
        if v > 0.25:
            return None
        p = (1 - math.sqrt(1 - 4 * v)) / 2
        return (lambda rng, shape: B * (rng.random(shape) < p)), B * p
    if law == "symmetric-beta":
        a = (1 / (4 * v) - 1) / 2
        if a <= 0:
            return None
        return (lambda rng, shape: B * rng.beta(a, a, shape)), B / 2
    if law == "beta-mean-0.15":
        m = 0.15 / B
        total = m * (1 - m) / v - 1
        if total <= 0:
            return None
        return (lambda rng, shape: B * rng.beta(m * total, (1 - m) * total, shape)), 0.15
    if law == "student-t3":
        scale = math.sqrt(var / 3)
        return (lambda rng, shape: scale * rng.standard_t(3, shape)), 0.0
    if law == "contaminated":
        s = math.sqrt(var / (0.98 + 0.02 * 36))
        return (lambda rng, shape: s * rng.standard_normal(shape) * np.where(rng.random(shape) < 0.02, 6.0, 1.0)), 0.0
    raise ValueError(law)


def coverage_fixed(draw, mean, K, eps, reps, rng, chunk=20000):
    hits = 0
    for start in range(0, reps, chunk):
        n = min(chunk, reps - start)
        hits += int((np.abs(draw(rng, (n, K)).mean(1) - mean) <= eps).sum())
    return hits / reps


def coverage_plugin(draw, mean, n0, eps, delta, B, reps, rng):
    pilot = draw(rng, (reps, n0))
    var_hat = pilot.var(axis=1, ddof=1)
    L = math.log(2 / delta)
    budgets = np.ceil((2 * var_hat + 2 / 3 * B * eps) / eps ** 2 * L).astype(int)
    hits = 0
    for K in np.unique(budgets):
        idx = budgets == K
        hits += int((np.abs(draw(rng, (int(idx.sum()), int(K))).mean(1) - mean) <= eps).sum())
    exceeds = float(((pilot.max(1) - pilot.min(1)) > B).mean())
    return hits / reps, exceeds, float(budgets.mean())


def main():
    p = parser(__doc__, model=False, dataset=False)
    p.add_argument("--reps", type=int, default=100000)
    p.add_argument("--nodes", type=int, default=20)
    p.add_argument("--pilots", type=int, nargs="+", default=[9, 16, 50])
    args = p.parse_args()
    cfg = config(args)
    eps, delta, B = cfg["bounds"]["epsilon"], cfg["bounds"]["delta"], cfg["bounds"]["B"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for var in VARIANCES:
        for law in ("two-point", "symmetric-beta", "beta-mean-0.15"):
            s = sampler(law, var, B)
            if s is None:
                rows.append({"variance": var, "law": law, "feasible": False})
                continue
            draw, mean = s
            K = bernstein_budget(var, eps, delta, B)
            row = {"variance": var, "law": law, "feasible": True, "K_star": K,
                   "a_true_variance": coverage_fixed(draw, mean, K, eps, args.reps, rng)}
            row["c_simultaneous"] = row["a_true_variance"] ** args.nodes
            for n0 in args.pilots:
                row[f"b_plugin_n0={n0}"], row[f"range_exceeds_n0={n0}"], row[f"mean_K_hat_n0={n0}"] = \
                    coverage_plugin(draw, mean, n0, eps, delta, B, args.reps, rng)
            rows.append(row)
    for var in (0.018, 0.044, 0.083):
        for law in ("student-t3", "contaminated"):
            draw, mean = sampler(law, var, B)
            row = {"variance": var, "law": law, "feasible": True, "misspecified": True}
            for n0 in args.pilots:
                row[f"b_plugin_n0={n0}"], row[f"range_exceeds_n0={n0}"], row[f"mean_K_hat_n0={n0}"] = \
                    coverage_plugin(draw, mean, n0, eps, delta, B, args.reps, rng)
            rows.append(row)
    bounded = [r for r in rows if r["feasible"] and not r.get("misspecified")]
    mis = [r for r in rows if r.get("misspecified")]
    plug = [(r[f"b_plugin_n0={n0}"], r["law"], r["variance"], n0) for r in bounded for n0 in args.pilots]
    summary = {
        "a_min_true_variance": min(r["a_true_variance"] for r in bounded),
        "b_min_plugin": min(plug), "c_min_simultaneous": min(r["c_simultaneous"] for r in bounded),
        "d_min_misspecified": min((r[f"b_plugin_n0={n0}"], r["law"], r["variance"], n0) for r in mis for n0 in args.pilots),
        "range_exceeds_misspecified": {n0: [min(r[f"range_exceeds_n0={n0}"] for r in mis),
                                            max(r[f"range_exceeds_n0={n0}"] for r in mis)] for n0 in args.pilots},
        "infeasible": [(r["law"], r["variance"]) for r in rows if not r["feasible"]],
        "settings": {"epsilon": eps, "delta": delta, "B": B, "reps": args.reps, "nodes": args.nodes},
    }
    out = Path(args.results) / "paper"
    write_json(out / "coverage_simulation.json", {"summary": summary, "rows": rows})
    lines = ["# Coverage simulations (Appendix B)", "", f"reps={args.reps}, eps={eps}, delta={delta}, B={B}", ""]
    lines += [f"- (a) true-variance budget: min coverage {summary['a_min_true_variance']:.4f}",
              f"- (b) plug-in budget: min coverage {summary['b_min_plugin'][0]:.4f} "
              f"({summary['b_min_plugin'][1]}, var={summary['b_min_plugin'][2]}, n0={summary['b_min_plugin'][3]})",
              f"- (c) {args.nodes} simultaneous nodes: min {summary['c_min_simultaneous']:.4f}",
              f"- (d) misspecified range: min coverage {summary['d_min_misspecified'][0]:.4f} "
              f"({summary['d_min_misspecified'][1]}, var={summary['d_min_misspecified'][2]}, n0={summary['d_min_misspecified'][3]})"]
    lines += [f"- pilot range > B (misspecified), n0={n0}: {100 * a:.0f}-{100 * b:.0f}%"
              for n0, (a, b) in summary["range_exceeds_misspecified"].items()]
    lines += [f"- infeasible: {summary['infeasible']}"]
    (out / "coverage_simulation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
