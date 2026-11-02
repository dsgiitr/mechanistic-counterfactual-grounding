"""Numerical stabilization shared by contextual decomposition implementations."""


def normalize_rel_irrel(rel, irrel):
    total = rel + irrel
    opposite = (rel * irrel) < 0
    relevant = opposite & (rel.abs() >= irrel.abs())
    irrelevant = opposite & ~relevant
    rel[relevant] = total[relevant]
    rel[irrelevant] = 0
    irrel[irrelevant] = total[irrelevant]
    irrel[relevant] = 0
