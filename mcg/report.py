"""Budgets and audit costs from group-mean variances (Sections 3.4 and 5, Appendix A).

Inputs are per-model group means V (grounded / hallucinated), retention rates
and timings, either from fresh runs or from `configs/reported.yaml`.
"""
import math

from mcg.bounds import budget_floor, budget_slope, eb_budget, hoeffding_budget


def plugin(V, eps, delta, B):
    """K_bar = (2L/eps^2) V + 2BL/(3 eps): plug-in budget before the per-node ceiling."""
    return budget_slope(eps, delta) * V + budget_floor(eps, delta, B)


def table2(means, eps, delta, B):
    rows = []
    for model, m in means.items():
        g, h = m["grounded"], m["hallucinated"]
        kg, kh = plugin(g, eps, delta, B), plugin(h, eps, delta, B)
        rows.append({"model": model, "V_grounded": g, "V_hallucinated": h, "V_ratio": h / g,
                     "K_grounded": kg, "K_hallucinated": kh, "K_ratio": kh / kg})
    return rows


def table3(means, delta, epsilons=(0.1, 0.185, 0.25), ranges=(0.6, 1.0)):
    rows = []
    for eps in epsilons:
        for B in ranges:
            rows.append({"epsilon": eps, "B": B, **{model: plugin(m["hallucinated"], eps, delta, B) /
                                                    plugin(m["grounded"], eps, delta, B)
                                                    for model, m in means.items()}})
    return rows


def table5(means, eps, delta, B):
    rows = {"plug-in Bernstein": {}, "EB, one-sided": {}, "EB, two-sided": {}, "Hoeffding": {}}
    for model, m in means.items():
        rows["plug-in Bernstein"][model] = [plugin(m[g], eps, delta, B) for g in ("grounded", "hallucinated")]
        rows["EB, one-sided"][model] = [eb_budget(m[g], eps, delta, B) for g in ("grounded", "hallucinated")]
        rows["EB, two-sided"][model] = [eb_budget(m[g], eps, delta / 2, B) for g in ("grounded", "hallucinated")]
        rows["Hoeffding"][model] = [B ** 2 * math.log(2 / delta) / (2 * eps ** 2)] * 2
    return rows


def table6(means, retention, timing, eps, delta, B, circuit_sizes=(20, 50), decodings=10):
    """Cost of auditing one answer at its plug-in budget K = ceil(K_bar)."""
    rows = []
    for model, m in means.items():
        r = retention[model]
        t = timing[model]
        out = {"model": model, "retention_rate": r}
        for group in ("hallucinated", "grounded"):
            K = math.ceil(plugin(m[group], eps, delta, B))
            generated = math.ceil(K / r)
            passes = [K * (s + 1) for s in circuit_sizes]
            seconds = [generated * t["decode10"] + p * t["tf_b16_per_pass"] + t.get("trace_seconds", 0.0)
                       for p in passes]
            out[group] = {"K": K, "generated": generated, "sequences": generated * decodings,
                          "tf_passes": passes, "gpu_seconds": seconds}
        rows.append(out)
    return rows


def hoeffding_bernstein_ratio(eps, var_over_B2, delta, B=1.0):
    """K_Hoeffding / K_Bernstein (no ceilings) at variance var_over_B2 * B^2."""
    L = math.log(2 / delta)
    hoeff = B ** 2 * L / (2 * eps ** 2)
    bern = (2 * var_over_B2 * B ** 2 + 2 / 3 * B * eps) / eps ** 2 * L
    return hoeff / bern


__all__ = ["plugin", "table2", "table3", "table5", "table6", "hoeffding_bernstein_ratio", "hoeffding_budget"]
