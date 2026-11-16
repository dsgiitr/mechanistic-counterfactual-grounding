"""Counterfactual budgets from concentration inequalities (Section 3.4, Appendix B).

With L = log(2/delta), tolerance epsilon and effect range B_v:

  Bernstein (known variance, Eq. 3)   K* = ceil((2 sigma^2 + 2/3 B eps) / eps^2 * L)
  plug-in projected budget (Eq. 4)    K_hat = same with the unbiased sample variance
  circuit summary                     K_bar = mean_v K_hat_v,   V = mean_v sigma_hat_v^2
  empirical Bernstein (Eq. 5)         mu - mu_hat <= sqrt(2 sigma_hat^2 L / K) + 7 B L / (3 (K-1))
                                      (Maurer & Pontil 2009, Thm. 4, one-sided)
  Hoeffding                           K >= B^2 L / (2 eps^2)

K_hat is a projection (descriptive), not a finite-sample guarantee; the empirical
Bernstein radius is the guarantee with an estimated variance.
"""
import math

import numpy as np


def log_term(delta):
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    return math.log(2 / delta)


def bernstein_budget(variance, epsilon, delta, effect_range):
    L = log_term(delta)
    return math.ceil((2 * variance + 2 / 3 * effect_range * epsilon) / epsilon ** 2 * L)


def budget_floor(epsilon, delta, effect_range):
    """Variance-free part of Eq. 4 (Eq. 7): 2 B L / (3 eps)."""
    return 2 * effect_range * log_term(delta) / (3 * epsilon)


def budget_slope(epsilon, delta):
    """d K_hat / d sigma^2 = 2 L / eps^2."""
    return 2 * log_term(delta) / epsilon ** 2


def hoeffding_budget(epsilon, delta, effect_range):
    return math.ceil(effect_range ** 2 * log_term(delta) / (2 * epsilon ** 2))


def eb_radius(count, variance, effect_range, delta):
    if count < 2:
        return math.inf
    L = log_term(delta)
    return math.sqrt(2 * variance * L / count) + 7 * effect_range * L / (3 * (count - 1))


def eb_min_budget(epsilon, delta, effect_range):
    """K at which the range term of Eq. 5 alone equals epsilon: 1 + 7 B L / (3 eps)."""
    return 1 + 7 * effect_range * log_term(delta) / (3 * epsilon)


def eb_budget(variance, epsilon, delta, effect_range, k_max=10 ** 7):
    """Smallest K with eb_radius(K) <= eps if the variance stayed at its estimate."""
    lo = 2
    if eb_radius(k_max, variance, effect_range, delta) > epsilon:
        return None
    hi = k_max
    while lo < hi:
        mid = (lo + hi) // 2
        if eb_radius(mid, variance, effect_range, delta) <= epsilon:
            hi = mid
        else:
            lo = mid + 1
    return lo


def node_statistics(effects, epsilon, delta, effect_range, simultaneous=1):
    """Statistics of one node's restoration-effect sample (needs >= 2 effects)."""
    x = np.asarray(effects, dtype=float)
    n = len(x)
    if n < 2:
        return None
    variance = float(x.var(ddof=1))
    delta_sim = delta / simultaneous
    return {
        "n": n, "mean": float(x.mean()), "variance": variance,
        "k_hat": bernstein_budget(variance, epsilon, delta, effect_range),
        "eb_radius": eb_radius(n, variance, effect_range, delta),
        "eb_radius_simultaneous": eb_radius(n, variance, effect_range, delta_sim),
        "eb_budget": eb_budget(variance, epsilon, delta, effect_range),
        "observed_range": float(x.max() - x.min()),
        "range_exceeds_B": bool(x.max() - x.min() > effect_range),
    }


def example_statistics(effects_by_node, epsilon, delta, effect_range):
    """V_i and K_bar_i of one example (Eq. 4); None if fewer than two effects."""
    nodes = {k: node_statistics(v, epsilon, delta, effect_range, simultaneous=len(effects_by_node))
             for k, v in effects_by_node.items()}
    if not nodes or any(s is None for s in nodes.values()):
        return None
    variances = [s["variance"] for s in nodes.values()]
    budgets = [s["k_hat"] for s in nodes.values()]
    eb = [s["eb_budget"] for s in nodes.values()]
    return {
        "V": float(np.mean(variances)), "K_bar": float(np.mean(budgets)),
        "K_bar_eb": float(np.mean(eb)) if all(b is not None for b in eb) else None,
        "max_eb_radius_simultaneous": float(max(s["eb_radius_simultaneous"] for s in nodes.values())),
        "fraction_nodes_range_exceeds_B": float(np.mean([s["range_exceeds_B"] for s in nodes.values()])),
        "num_nodes": len(nodes), "num_effects": nodes[next(iter(nodes))]["n"], "nodes": nodes,
    }


def bound_constants(epsilon, delta, effect_range):
    return {
        "epsilon": epsilon, "delta": delta, "B": effect_range, "L": log_term(delta),
        "k_hat_floor": budget_floor(epsilon, delta, effect_range),
        "k_hat_min": math.ceil(budget_floor(epsilon, delta, effect_range)),
        "k_hat_slope": budget_slope(epsilon, delta),
        "eb_min_budget": eb_min_budget(epsilon, delta, effect_range),
        "hoeffding_budget": hoeffding_budget(epsilon, delta, effect_range),
        "max_variance_for_range": effect_range ** 2 / 4,
    }
