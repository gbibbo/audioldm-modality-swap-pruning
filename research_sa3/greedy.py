"""Sequential greedy selection (protocol section 3.5), non-additive.

Start from the dense post model; evaluate all `depth` single removals; remove the best
(smallest score); re-evaluate all remaining removals FROM THE NEW ARCHITECTURE; repeat to
k=6. That is depth+(depth-1)+...+(depth-k+1) candidate evaluations. `score_fn(frozenset)`
returns the criterion of a removed set (lower = better; measured on the model with that set
removed). The additivity gap at R(k) is score_fn(R(k)) - sum_g score_fn({g}), a diagnostic.
"""
from __future__ import annotations
from typing import Callable, Dict, FrozenSet, List, Tuple


def greedy_path(depth: int, k_max: int, score_fn: Callable[[FrozenSet], float]) -> Dict:
    """Return {'path':[(k,set,score)], 'sets':{k:set}, 'n_evals':int, 'order':[blocks]}."""
    removed: List[int] = []
    path: List[Tuple[int, List[int], float]] = []
    n_evals = 0
    for _k in range(1, k_max + 1):
        best_g, best_s = None, float("inf")
        for g in range(depth):
            if g in removed:
                continue
            cand = frozenset(removed + [g])
            s = score_fn(cand)
            n_evals += 1
            if s < best_s:
                best_s, best_g = s, g
        removed.append(best_g)
        path.append((_k, sorted(removed), best_s))
    sets = {k: set(s) for (k, s, _sc) in path}
    return {"path": path, "sets": sets, "n_evals": n_evals, "order": list(removed)}


def additivity_gap(R_k: FrozenSet, score_fn: Callable[[FrozenSet], float]) -> float:
    """score(R_k) - sum_{g in R_k} score({g})."""
    return score_fn(frozenset(R_k)) - sum(score_fn(frozenset([g])) for g in R_k)


def set_divergence(set_x: set, set_y: set) -> int:
    """delta_XY(k) = |R_X △ R_Y| / 2 (symmetric-difference count halved; integer for equal-size sets)."""
    sym = set(set_x) ^ set(set_y)
    return len(sym) // 2
