"""Copycat-style correspondence backend (thesis Experiment 2).

The parallel terraced scan (Hofstadter & Mitchell 1994; Mitchell 1993)
collapsed to its load-bearing core: a stochastic, temperature-driven search
over object correspondences, where SLIPPAGE — matching two objects despite an
attribute mismatch — is licensed quantitatively by the learned
:class:`~dowhat.concepts.ConceptNet` slips (the corpus probability that a
true correspondence changes that attribute). With ``DEFAULT_CONCEPTS`` (all
slips zero) the objective is exactly the SME backend's, so the two mappers
differ only in search: deterministic greedy+repair versus annealed sampling.
Given an ``rng`` the run is fully deterministic; no salted hashing anywhere.
"""

from __future__ import annotations

import math
import random

from .analogy import MAX_REPAIR_OBJECTS, _greedy_mapping, _map_score, relations
from .concepts import DEFAULT_CONCEPTS, ConceptNet
from .representation import Obj, StateGraph

ITERATIONS = 200
_T0 = 2.0  # initial temperature
_COOL = 0.98  # geometric cooling per iteration


def copycat_map(
    a: StateGraph,
    b: StateGraph,
    concepts: ConceptNet | None = None,
    rng: random.Random | None = None,
) -> list[tuple[Obj | None, Obj | None]]:
    """Stochastic correspondence search; same contract as ``structure_map``."""
    net = concepts if concepts is not None else DEFAULT_CONCEPTS
    rng = rng if rng is not None else random.Random(0)
    a_objs = {o.oid: o for o in a.objects}
    b_objs = {o.oid: o for o in b.objects}
    a_ids = sorted(a_objs)
    b_ids = sorted(b_objs)
    rels_a, rels_b = relations(a), relations(b)

    def score(m: dict[int, int]) -> float:
        return _map_score(m, a_objs, b_objs, rels_a, rels_b, net, slip=True)

    current = _greedy_mapping(a, b, concepts)
    cur_score = score(current)
    best, best_score = dict(current), cur_score

    # same degradation bound as SME's swap-repair: above it, greedy only
    anneal = a_ids and b_ids and max(len(a_ids), len(b_ids)) <= MAX_REPAIR_OBJECTS
    for it in range(ITERATIONS if anneal else 0):
        temperature = _T0 * _COOL**it
        trial = dict(current)
        move = rng.choice(("swap", "remap", "unmap", "map"))
        if move == "swap" and len(trial) >= 2:
            x1, x2 = rng.sample(sorted(trial), 2)
            trial[x1], trial[x2] = trial[x2], trial[x1]
        elif move == "remap" and trial:
            x = rng.choice(sorted(trial))
            free = [y for y in b_ids if y not in trial.values() or y == trial[x]]
            trial[x] = rng.choice(free)
        elif move == "unmap" and trial:
            del trial[rng.choice(sorted(trial))]
        elif move == "map":
            unmapped = [x for x in a_ids if x not in trial]
            free = [y for y in b_ids if y not in trial.values()]
            if not unmapped or not free:
                continue
            trial[rng.choice(unmapped)] = rng.choice(free)
        else:
            continue
        t_score = score(trial)
        delta = t_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
            current, cur_score = trial, t_score
            if cur_score > best_score:
                best, best_score = dict(current), cur_score

    pairs: list[tuple[Obj | None, Obj | None]] = [
        (a_objs[x], b_objs[y]) for x, y in sorted(best.items())
    ]
    matched_b = set(best.values())
    pairs.extend((a_objs[x], None) for x in a_ids if x not in best)
    pairs.extend((None, b_objs[y]) for y in b_ids if y not in matched_b)
    return pairs
