"""How does preimage enumeration scale for highly non-injective operations?

Ground-truth abduction instances (the deleted objects are known and drawn from
the hypothesis catalogue) are swept over grid size, deletion count, palette
size and selector family. Reported per cell: the RANK of the true pre-state in
the preimage stream, total candidates yielded, wall time, and recall at the
abduce_inputs-style limits {16, 64, 256}. A budget sweep separates cap-plateau
from intrinsic growth; a chain sweep shows where the per-layer limit starves
the true origin.

Run:  python examples/abduction_scaling.py [SEEDS_PER_CELL]   (default 5)
"""

import random
import statistics
import sys
import time

from twinworld.benchmark import _place_objects, random_delete_instance
from twinworld.engine import abduce_inputs
from twinworld.mechanisms import (
    DEFAULT_PREIMAGE_BUDGET,
    ByColour,
    Delete,
    ObjectRule,
    PreimageBudget,
    RecolourTo,
)
from twinworld.representation import parse_grid

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LIMITS = (16, 64, 256)
STREAM_CAP = 4096  # stop enumerating a single preimage stream after this many yields


def rule_title(title):
    print(f"\n{'─' * 76}\n{title}\n{'─' * 76}")


def measure(rule, pre, observed, budget=None):
    """(rank of true pre in the yielded stream or None, yields seen, wall ms)."""
    t0 = time.perf_counter()
    rank = None
    yields = 0
    for i, candidate in enumerate(rule.preimage(observed, budget), start=1):
        yields = i
        if candidate == pre:
            rank = i
            break
        if i >= STREAM_CAP:
            break
    return rank, yields, (time.perf_counter() - t0) * 1000


def cell(family, size, n_deleted, n_palette, budget=None, seeds=SEEDS):
    ranks, walls, found = [], [], 0
    made = 0
    rng = random.Random(f"{family}-{size}-{n_deleted}-{n_palette}")  # str seed: unsalted
    attempts = 0
    while made < seeds and attempts < seeds * 20:
        attempts += 1
        inst = random_delete_instance(
            rng, size=size, n_deleted=n_deleted, n_palette=n_palette, family=family
        )
        if inst is None:
            continue
        made += 1
        rank, _, wall = measure(*inst, budget=budget)
        walls.append(wall)
        if rank is not None:
            ranks.append(rank)
            found += 1
    if not made:
        return None
    med_rank = f"{statistics.median(ranks):.0f}" if ranks else "—"
    recall = {
        lim: sum(1 for r in ranks if r <= lim) / made for lim in LIMITS
    }
    return {
        "made": made,
        "rank": med_rank,
        "ms": statistics.median(walls),
        "recall": recall,
        "found": found / made,
    }


rule_title("1. rank / recall / cost of the true pre-state — legacy cap vs fixed")
LEGACY = PreimageBudget(cap_singles=True)  # the pre-fix behaviour, kept measurable
print(f"  {'family':13}{'size':>5}{'del':>4}{'pal':>4}"
      f"{'legacy rank':>12}{'fixed rank':>11}{'r@256 leg':>10}{'r@256 fix':>10}{'ms fix':>8}")
for family in ("bycolour", "smallest", "not_bycolour"):
    for size in (7, 10, 15, 20, 30):
        for n_deleted in (1, 2):
            for n_palette in (1, 2, 4):
                if family == "not_bycolour" and n_palette < 2:
                    continue
                legacy = cell(family, size, n_deleted, n_palette, budget=LEGACY)
                fixed = cell(family, size, n_deleted, n_palette)
                if legacy is None or fixed is None:
                    continue
                print(f"  {family:13}{size:>5}{n_deleted:>4}{n_palette:>4}"
                      f"{legacy['rank']:>12}{fixed['rank']:>11}"
                      f"{legacy['recall'][256]:>10.2f}{fixed['recall'][256]:>10.2f}"
                      f"{fixed['ms']:>8.0f}")

rule_title("2. honesty row: out-of-catalogue deleted shape (L-tromino)")
rng = random.Random(0)
grid = _place_objects(
    rng, 10, [(3, frozenset({(0, 0), (1, 0), (1, 1)})), (2, frozenset({(0, 0)}))]
)
pre = parse_grid(grid)
delete_rule = ObjectRule(ByColour(3), Delete())
observed = delete_rule.apply(pre)
rank, yields, wall = measure(delete_rule, pre, observed)
print(f"  deleted an L-tromino: rank {'—' if rank is None else rank} after {yields} yields "
      f"({wall:.0f} ms) — recall 0 by construction; the catalogue stops at bars")

rule_title("3. budget sweep at the large sizes (cap-plateau vs intrinsic growth)")
print(f"  {'family':13}{'size':>5}{'budget':>12}{'med rank':>9}{'med ms':>8}{'r@256':>7}")
for family in ("bycolour", "not_bycolour"):
    for size in (20, 30):
        for factor in (1, 2, 4):
            b = PreimageBudget(
                anchors=DEFAULT_PREIMAGE_BUDGET.anchors * factor,
                pairs=DEFAULT_PREIMAGE_BUDGET.pairs * factor,
            )
            row = cell(family, size, 2, 2, budget=b)
            if row is None:
                continue
            print(f"  {family:13}{size:>5}{f'x{factor}':>12}{row['rank']:>9}"
                  f"{row['ms']:>8.0f}{row['recall'][256]:>7.2f}")

rule_title("4. backward chains: where the per-layer limit starves the origin")
print(f"  {'depth':>7}{'limit':>7}{'recovered':>11}{'med ms':>8}")
for depth in (1, 2, 3):
    for limit in LIMITS:
        hits, walls, made = 0, [], 0
        rng = random.Random(depth * 100)
        attempts = 0
        while made < SEEDS and attempts < SEEDS * 20:
            attempts += 1
            inst = random_delete_instance(rng, size=10, n_palette=3, family="bycolour")
            if inst is None:
                continue
            delete_rule, pre, mid = inst
            program = [delete_rule]
            state = mid
            # prepend recolour steps: survivors change colour after the delete
            ok = True
            for k in range(depth - 1):
                colours = sorted({o.colour for o in state.objects})
                if not colours:
                    ok = False
                    break
                src = colours[0]
                dst = 8 if k == 0 else 9
                step = ObjectRule(ByColour(src), RecolourTo(dst))
                nxt = step.apply(state)
                if nxt is None:
                    ok = False
                    break
                program.append(step)
                state = nxt
            if not ok:
                continue
            made += 1
            t0 = time.perf_counter()
            origins = abduce_inputs(program, state, limit=limit)
            walls.append((time.perf_counter() - t0) * 1000)
            hits += pre in origins
        if made:
            print(f"  {depth:>7}{limit:>7}{hits:>7}/{made:<3}"
                  f"{statistics.median(walls):>8.0f}")

print("\nreading: candidates ≈ (singles + pairs) × |pinned colours|, each verified by")
print("two full render+segment passes. The legacy anchor cap made most true origins")
print("HARD-UNREACHABLE on large grids (top-left bias); uncapped lazy singles make")
print("them findable at a rank that now scales with |free|·|shapes|·|colours| — the")
print("honest law. Two-object truths still trail every single-object world (Occam")
print("orders smallest-first) and pairs remain capped: that wall is documented, not")
print("hidden. Chains: the per-layer limit must cover the frontier's breadth.")
