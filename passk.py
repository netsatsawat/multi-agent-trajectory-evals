"""pass^k: the chance that ALL k tries of a task succeed.

Estimator from tau-bench (Yao et al., 2024, arXiv:2406.12045): with n
recorded tries of a task and c successes, pass^k = C(c, k) / C(n, k),
averaged over tasks. Compare pass@k = 1 - C(n - c, k) / C(n, k), the
chance that AT LEAST ONE of k tries succeeds.
"""
from math import comb


def pass_hat_k(results_per_task, k):
    """results_per_task: list of lists of booleans, one list per task."""
    scores = []
    for results in results_per_task:
        n, c = len(results), sum(results)
        if k > n:
            raise ValueError(f"k={k} needs at least {k} tries, got {n}")
        scores.append(comb(c, k) / comb(n, k))
    return sum(scores) / len(scores)


def pass_at_k(results_per_task, k):
    scores = []
    for results in results_per_task:
        n, c = len(results), sum(results)
        scores.append(1 - comb(n - c, k) / comb(n, k))
    return sum(scores) / len(scores)
