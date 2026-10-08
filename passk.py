"""Score repeated tries of a task with pass^k and pass@k.

pass^k is the chance that all k tries of a task succeed. pass@k is the
chance that at least one of the k tries succeeds. For a task with n
recorded tries and c successes, the tau-bench paper (Yao et al., 2024,
arXiv:2406.12045) estimates pass^k as C(c, k) / C(n, k). For pass@k the
estimate is 1 - C(n - c, k) / C(n, k). Both functions average their
scores over tasks.
"""
from math import comb


def pass_hat_k(results_per_task, k):
    """Return pass^k averaged over tasks.

    results_per_task holds one list of booleans per task.
    """
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
