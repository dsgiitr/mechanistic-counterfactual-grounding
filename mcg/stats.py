"""Example-level statistics. The unit of comparison is the example, never the counterfactual."""
import math

import numpy as np


def wilson_interval(successes, n, z=1.959963984540054):
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (centre - half, centre + half)


def bootstrap_ci(values, statistic=np.mean, n_boot=10000, level=0.95, seed=0):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    stats = statistic(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    alpha = (1 - level) / 2
    return (float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha)))


def bootstrap_ratio_ci(numerator, denominator, n_boot=10000, level=0.95, seed=0):
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    if len(a) < 2 or len(b) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    ra = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1)
    rb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(1)
    ratio = ra / rb
    alpha = (1 - level) / 2
    return (float(np.quantile(ratio, alpha)), float(np.quantile(ratio, 1 - alpha)))


def auroc(positive, negative):
    """P(score_pos > score_neg) + 0.5 P(tie) (= Mann-Whitney U / (n1 n2))."""
    pos, neg = np.asarray(positive, float), np.asarray(negative, float)
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def mann_whitney(positive, negative):
    from scipy.stats import mannwhitneyu
    if len(positive) == 0 or len(negative) == 0:
        return {"U": math.nan, "p_value": math.nan}
    result = mannwhitneyu(positive, negative, alternative="two-sided")
    return {"U": float(result.statistic), "p_value": float(result.pvalue)}


def match_on_confidence(hallucinated, grounded, key="factual_prob", caliper=0.05):
    """Greedy 1:1 nearest-neighbour matching (without replacement) on p_theta(y)."""
    pool = sorted(grounded, key=lambda r: r[key])
    pairs = []
    for record in sorted(hallucinated, key=lambda r: r[key]):
        if not pool:
            break
        best = min(range(len(pool)), key=lambda j: abs(pool[j][key] - record[key]))
        if abs(pool[best][key] - record[key]) <= caliper:
            pairs.append((record, pool.pop(best)))
    return pairs


def summarize(values):
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if len(x) == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(x)
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "ci95": [lo, hi]}


def cluster_bootstrap_auroc(positive, pos_clusters, negative, neg_clusters, n_boot=2000, level=0.95, seed=0):
    """AUROC with a percentile CI from resampling clusters (images) with replacement."""
    pos, neg = np.asarray(positive, float), np.asarray(negative, float)
    pc, nc = np.asarray(pos_clusters), np.asarray(neg_clusters)
    clusters = np.unique(np.concatenate([pc, nc]))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        counts = {c: n for c, n in zip(*np.unique(pick, return_counts=True))}
        wp = np.array([counts.get(c, 0) for c in pc])
        wn = np.array([counts.get(c, 0) for c in nc])
        if wp.sum() == 0 or wn.sum() == 0:
            continue
        greater = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        values.append(float((wp[:, None] * wn[None, :] * greater).sum() / (wp.sum() * wn.sum())))
    alpha = (1 - level) / 2
    return {"auroc": auroc(pos, neg), "ci95": [float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))]}


def cluster_bootstrap_mean(values, clusters, n_boot=2000, level=0.95, seed=0):
    """Mean with a percentile CI from resampling clusters with replacement."""
    x, c = np.asarray(values, float), np.asarray(clusters)
    keys = np.unique(c)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        counts = {k: n for k, n in zip(*np.unique(pick, return_counts=True))}
        w = np.array([counts.get(k, 0) for k in c])
        means.append(float((w * x).sum() / w.sum()))
    alpha = (1 - level) / 2
    return float(x.mean()), [float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))]
