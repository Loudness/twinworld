"""Ground-truth counterfactual benchmark report (thesis Experiment 3).

Tasks are generated FROM known latent programs, so three quantities that are
unknowable in tabular-ML counterfactual research are exact here:
  - recovery: is the induced program behaviourally equivalent to the latent
    one on counterfactual probes?
  - underdetermination: how often do several non-equivalent programs fit the
    demonstrations?
  - CF minimality gap: for a foil constructed one pool-edit away, the
    certified-minimal generator must return k=1 (gap 0).

Usage: python examples/cf_benchmark.py [N_INSTANCES]
"""

import random
import statistics
import sys

from twinworld import Contrastive, UnsolvedTaskError, compute, identify, induce_rules, model
from twinworld.benchmark import random_task
from twinworld.discriminate import diagnose, probes, signature
from twinworld.engine import ApplyCache, solve_all

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = random.Random(0)

made = solved = recovered = underdetermined = 0
gaps = []
attempts = 0
while made < N and attempts < N * 30:
    attempts += 1
    instance = random_task(rng)
    if instance is None:
        continue
    task, latent = instance
    made += 1
    try:
        rep = model(task)
    except UnsolvedTaskError:
        continue
    solved += 1

    probe_grids = probes(task, rep.abstraction)
    cache = ApplyCache()
    if signature(rep.solution.program, probe_grids, rep.abstraction, cache) == signature(
        latent, probe_grids, rep.abstraction, cache
    ):
        recovered += 1

    fits = solve_all(
        task,
        induce_rules(task, rep.abstraction),
        max_depth=max(1, len(rep.solution.program)),
        abstraction=rep.abstraction,
    )
    if fits and diagnose(task, fits, rep.abstraction).underdetermined:
        underdetermined += 1

    trace = rep.solution.train_traces[0]
    pool = [*induce_rules(task, rep.abstraction), *rep.primitives]
    for mech in pool:
        if mech == trace.mechanisms[-1]:
            continue
        alt = rep.solution.cache.run(trace.states[0], trace.mechanisms[:-1] + (mech,))
        if alt is not None and alt.outcome.key != trace.outcome.key:
            cfs = compute(identify(rep, Contrastive(alt.outcome.grid, on="train[0]")))
            if cfs.items[0].metrics.applicable:
                gaps.append(min(i.metrics.sparsity for i in cfs.items) - 1)
            break

print(f"\nbenchmark instances : {made}")
print(f"solved (train fit)  : {solved}  ({solved / made:.0%})")
print(f"latent recovered    : {recovered}  "
      f"({recovered / max(solved, 1):.0%} of solved are probe-equivalent to ground truth)")
print(f"underdetermined     : {underdetermined}  "
      f"(>1 behavioural class among fitting rule programs)")
if gaps:
    print(f"CF minimality gap   : mean {statistics.mean(gaps):.2f} over {len(gaps)} "
          f"contrastive queries (0 = certified minimum equals the true minimum)")