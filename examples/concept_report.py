"""Learned concept network + SME-vs-Copycat report (digest §8 open questions 1–2).

Learns ConceptNet weights from the ARC-AGI-2 TRAINING corpus, then evaluates
hand-coded vs learned weights and SME vs Copycat vs union mappers on the
untouched 120-task PUBLIC EVAL set (headline) and on the training set
(leakage caveat). The learned network is written to docs/learned-concepts.json.

Run:  python examples/concept_report.py [N_TRAIN] [N_EVAL]   (defaults 1000 120)
      python examples/concept_report.py 100 40               # smoke, ~2-4 min
"""

import statistics
import sys
import time
from itertools import islice

import twinworld
from twinworld import DEFAULT_CONCEPTS, UnsolvedTaskError, as_grid, learn_concepts
from twinworld.concepts import save_concepts
from twinworld.domains.arc import iter_tasks

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
N_EVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 120

KNOWN = ("25ff71a9", "42a50994", "5582e5ca", "a79310a0", "e0fb7511", "b230c067")


def rule(title):
    print(f"\n{'─' * 76}\n{title}\n{'─' * 76}")


def transfers(task, sol) -> bool:
    if len(sol.test_traces) != len(task.test):
        return False
    return all(
        t.outcome.key == as_grid(expected)
        for t, (_, expected) in zip(sol.test_traces, task.test)
    )


def sweep(tasks, concepts, mapper):
    fit = solved = 0
    tried, walls = [], []
    solved_ids = set()
    for task in tasks:
        induction = "auto" if len(task.colours()) <= 6 else "always"
        t0 = time.perf_counter()
        try:
            rep = twinworld.model(task, induction=induction, concepts=concepts, mapper=mapper)
        except UnsolvedTaskError:
            continue
        walls.append((time.perf_counter() - t0) * 1000)
        fit += 1
        tried.append(rep.solution.programs_tried)
        if transfers(task, rep.solution):
            solved += 1
            solved_ids.add(task.task_id)
    med = lambda xs: f"{statistics.median(xs):.0f}" if xs else "—"  # noqa: E731
    return {
        "fit": fit, "solved": solved, "ids": solved_ids,
        "tried": med(tried), "ms": med(walls),
    }


train_tasks = list(islice(iter_tasks("train"), N_TRAIN))
eval_tasks = list(islice(iter_tasks("eval"), N_EVAL))

rule("0. leakage guard")
train_ids = {t.task_id for t in train_tasks}
eval_ids = {t.task_id for t in eval_tasks}
assert not (train_ids & eval_ids), "train/eval task ids overlap — protocol violated"
print(f"train={len(train_tasks)} eval={len(eval_tasks)} — id sets disjoint OK")
print("weights are LEARNED on train only; eval rows are the held-out headline")

rule("1. learn the concept network from the training corpus")
t0 = time.perf_counter()
net = learn_concepts(train_tasks)
print(f"learned in {time.perf_counter() - t0:.1f}s   ({net.source})\n")
hand = DEFAULT_CONCEPTS
print(f"  {'':14}{'hand':>8}{'learned':>10}")
for attr in ("shape", "colour", "location", "iou"):
    print(f"  {attr:14}{getattr(hand, attr):>8.2f}{getattr(net, attr):>10.2f}")
for (name, hw), (_, lw) in zip(hand.relations, net.relations):
    print(f"  rel:{name:10}{hw:>8.2f}{lw:>10.2f}")
for slip in ("slip_shape", "slip_colour", "slip_move", "slip_delete"):
    print(f"  {slip:14}{0.0:>8.2f}{getattr(net, slip):>10.2f}")
print("\n  top-5 rule-family priors (train-fit frequency):")
for fam, freq in net.priors[:5]:
    print(f"    {fam:28} {freq:.3f}")
path = save_concepts(net, "docs/learned-concepts.json")
print(f"\n  written to {path}")

rule("2. solve-rate sweep: weights x mapper x dataset")
configs = [
    ("hand   ", "sme    ", None, "sme"),
    ("learned", "sme    ", net, "sme"),
    ("hand   ", "copycat", None, "copycat"),
    ("learned", "copycat", net, "copycat"),
    ("learned", "both   ", net, "both"),
]
results = {}
for ds_name, tasks in (("eval", eval_tasks), ("train", train_tasks)):
    print(f"\n  {ds_name} ({len(tasks)} tasks)"
          + ("  [held-out headline]" if ds_name == "eval" else "  [seen by learner — caveat]"))
    print(f"  {'weights':9}{'mapper':9}{'train-fit':>10}{'solved':>8}{'med tried':>11}{'med ms':>8}")
    for wname, mname, concepts, mapper in configs:
        row = sweep(tasks, concepts, mapper)
        results[(ds_name, wname.strip(), mname.strip())] = row
        print(f"  {wname:9}{mname:9}{row['fit']:>10}{row['solved']:>8}"
              f"{row['tried']:>11}{row['ms']:>8}")

rule("3. compose or compete? (learned weights, test-verified ids)")
for ds_name in ("eval", "train"):
    sme_ids = results[(ds_name, "learned", "sme")]["ids"]
    cc_ids = results[(ds_name, "learned", "copycat")]["ids"]
    both_ids = results[(ds_name, "learned", "both")]["ids"]
    print(f"\n  {ds_name}: only-SME {len(sme_ids - cc_ids)}, only-Copycat {len(cc_ids - sme_ids)}, "
          f"agree {len(sme_ids & cc_ids)}, union-config solves {len(both_ids)}")
    if cc_ids - sme_ids:
        print(f"    copycat-only ids: {sorted(cc_ids - sme_ids)}")
    if sme_ids - cc_ids:
        print(f"    sme-only ids    : {sorted(sme_ids - cc_ids)}")

rule("4. the known specimens under learned weights (train set)")
known_in_run = [t for t in train_tasks if t.task_id in KNOWN]
for task in known_in_run:
    induction = "auto" if len(task.colours()) <= 6 else "always"
    line = f"  {task.task_id}: "
    try:
        rep = twinworld.model(task, induction=induction, concepts=net, mapper="sme")
        line += "solved" if transfers(task, rep.solution) else "train-fit only"
    except UnsolvedTaskError:
        line += "NO FIT (regression vs hand-coded)"
    print(line)

print("\ncaveats: train rows are in-sample for the learned weights; priors reorder")
print("candidates only (the engine verifies everything); eval set used here for the")
print("first time in this project.")
