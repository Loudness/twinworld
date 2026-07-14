"""Does counterfactual discrimination improve test accuracy on failure class 3?

The proposal's most testable hypothesis, measured. Controlled ambiguous tasks
(Largest ≡ ByColour(c) on train; diverging on test when treacherous) with a
known latent give ground truth. PASSIVE arm: selection policies choose a
behavioural class with no new information — pre-registered ceiling: at
latent_bias 0.5 the collision is symmetric and NO passive policy can beat 50%
(that block calibrates the instrument); a declared 0.75 colour-favoured world
follows. ACTIVE arm: oracle answers to the diagnosing probe versus the same
query budget spent on random extra demonstrations. Finally the real-ARC
passive rows: every train-corpus specimen the gate flags as LOW.

Run:  python examples/discrimination_report.py [N_PER_ARM] [--no-arc]
      (default 30 per arm; --no-arc skips the corpus sweep)
"""

import random
import sys
import time

import dowhat
from dowhat import UnsolvedTaskError, as_grid, induce_rules
from dowhat.benchmark import random_ambiguous_task, random_grid
from dowhat.discriminate import diagnose
from dowhat.engine import ApplyCache, solve_all
from dowhat.representation import parse_grid
from dowhat.select import POLICIES, resolve_with_probe

N = next((int(a) for a in sys.argv[1:] if a.isdigit()), 30)
RUN_ARC = "--no-arc" not in sys.argv
POLICY_NAMES = list(POLICIES)


def rule(title):
    print(f"\n{'─' * 76}\n{title}\n{'─' * 76}")


def instances(treacherous, bias, n, tag):
    rng = random.Random(f"disc-{tag}-{treacherous}-{bias}")
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        inst = random_ambiguous_task(rng, treacherous=treacherous, latent_bias=bias)
        if inst is not None:
            out.append(inst)
    return out


def correct(predictions, task) -> bool:
    if predictions is None:
        return False
    return all(
        g is not None and g == as_grid(expected)
        for g, (_, expected) in zip(predictions, task.test)
    )


def passive_block(insts, label):
    """Policies choose among classes on the LOW instances; the gate abstains."""
    rows = {name: 0 for name in POLICY_NAMES}
    low = 0
    gate_answered = gate_correct = 0
    for task, latent in insts:
        rep = dowhat.model(task)
        report = dowhat.assess(rep)
        if report.confidence == "high":
            gate_answered += 1
            gate_correct += correct(report.predictions[0], task)
            continue
        low += 1
        for name in POLICY_NAMES:
            idx = POLICIES[name](rep, report, random.Random(f"{name}-{task.task_id}-{low}"))
            rows[name] += correct(report.predictions[idx], task)
    print(f"\n  {label}: {len(insts)} instances — gate LOW on {low}, "
          f"HIGH on {gate_answered} ({gate_correct} correct)")
    print(f"  {'policy':18}{'correct':>9}{'accuracy on LOW':>17}")
    print(f"  {'gate (abstains)':18}{'—':>9}{'coverage 0%':>17}")
    for name in POLICY_NAMES:
        acc = rows[name] / low if low else 0.0
        print(f"  {name:18}{rows[name]:>6}/{low:<3}{acc:>15.2f}")
    return low


rule("1. generator sanity")
sample = instances(True, 0.5, N, "sanity")
classes_counts = []
for task, latent in sample:
    fits = solve_all(task, induce_rules(task), max_depth=1)
    classes_counts.append(len(diagnose(task, fits).classes))
underdet = sum(1 for c in classes_counts if c >= 2)
print(f"  {len(sample)} treacherous instances; >=2 behavioural classes: "
      f"{underdet}/{len(sample)}; mean classes {sum(classes_counts) / len(sample):.1f}")

rule("2. passive policies, symmetric world (latent_bias 0.5) — the calibration")
print("  pre-registered: the collision is symmetric here, so NO passive policy can")
print("  exceed ~50% except by sampling noise. This block calibrates the instrument.")
passive_block(instances(True, 0.5, N, "sym"), "treacherous @ bias 0.5")

rule("3. passive policies, colour-favoured world (latent_bias 0.75, declared a priori)")
passive_block(instances(True, 0.75, N, "skew"), "treacherous @ bias 0.75")

rule("4. benign block: ambiguity without divergence")
benign = instances(False, 0.5, N, "benign")
answered = right = 0
for task, latent in benign:
    rep = dowhat.model(task)
    prediction, report = dowhat.predict(rep)
    if prediction is not None:
        answered += 1
        right += correct(prediction, task)
print(f"  gate answers {answered}/{len(benign)} via test-unanimity, {right} correct —")
print("  coverage gained with zero wrong answers is the gate's half of the claim")

rule("5. active arm: oracle probe-queries vs random extra demonstrations")
treach = instances(True, 0.5, N, "active")
cache = ApplyCache()
probe_hits = {0: 0, 1: 0, 2: 0}
control_hits = {1: 0, 2: 0}
control_unbroken = 0
control_demos = 0
for task, latent in treach:
    fits = solve_all(task, induce_rules(task), max_depth=1)
    expected = as_grid(task.test[0][1])
    for budget in (0, 1, 2):
        chosen, _ = resolve_with_probe(task, fits, latent, max_queries=budget)
        trace = cache.run(parse_grid(task.test[0][0]), chosen)
        probe_hits[budget] += trace is not None and trace.outcome.key == expected
    # control: the same budget spent on random demonstrations answered by the oracle
    rng = random.Random(f"ctrl-{task.task_id}-{len(task.train)}-{probe_hits[1]}")
    extra = []
    while len(extra) < 2:
        grid = random_grid(rng)
        if grid is None:
            continue
        trace = cache.run(parse_grid(grid), latent)
        if trace is None:
            continue
        extra.append((as_grid(grid), trace.outcome.key))
    for budget in (1, 2):
        extended = dowhat.Task(
            train=task.train + tuple(extra[:budget]), test=task.test,
            task_id=task.task_id,
        )
        control_demos += 1
        fits2 = solve_all(extended, induce_rules(extended), max_depth=1)
        report = diagnose(extended, fits2)
        if budget == 1 and report.underdetermined:
            control_unbroken += 1
        chosen = report.classes[0][0] if report.classes else None
        trace = cache.run(parse_grid(task.test[0][0]), chosen) if chosen else None
        control_hits[budget] += trace is not None and trace.outcome.key == expected
n = len(treach)
print(f"  {n} treacherous instances (bias 0.5); accuracy on the held-out test:")
print(f"  {'queries':>9}{'diagnosing probe':>18}{'random demo':>13}")
print(f"  {0:>9}{probe_hits[0]:>12}/{n:<5}{'—':>10}")
for b in (1, 2):
    print(f"  {b:>9}{probe_hits[b]:>12}/{n:<5}{control_hits[b]:>7}/{n}")
print(f"  random demos that fail to break the collision after 1 query: "
      f"{control_unbroken}/{n}")

if RUN_ARC:
    rule("6. the real-ARC specimens (train corpus, gate == LOW)")
    from dowhat.domains.arc import iter_tasks

    t0 = time.perf_counter()
    found = 0
    for task in iter_tasks("train"):
        induction = "auto" if len(task.colours()) <= 6 else "always"
        try:
            rep = dowhat.model(task, induction=induction)
        except UnsolvedTaskError:
            continue
        report = dowhat.assess(rep)
        if report.confidence == "high":
            continue
        found += 1
        verdicts = []
        for name in POLICY_NAMES:
            idx = POLICIES[name](rep, report, random.Random(name))
            verdicts.append(f"{name}={'Y' if correct(report.predictions[idx], task) else 'n'}")
        print(f"  {task.task_id}: {report.classes} classes | " + " ".join(verdicts))
    print(f"  ({found} LOW specimen(s); swept in {time.perf_counter() - t0:.0f}s)")

print("\nhonesty notes: policies were fixed before the runs; largest_class exploits")
print("the generator's candidate-multiplicity prior, not causal structure; class")
print("granularity is bounded by the train-derived probe set; latent_bias values")
print("were declared in the source, not tuned; the ARC rows are passive-only (no")
print("oracle exists there).")
