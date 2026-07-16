"""Letter-string analogies end-to-end: Copycat's home domain on the sequence
backend — abc -> abd, certified; ambiguity diagnosed with the separating
probe; and Lewis & Mitchell's counterfactual alphabets as ordinary
interventions (arXiv:2402.08955).

Run:  python examples/letter_strings.py
"""

import time

import twinworld
from twinworld import Interventional, rank_preimages
from twinworld.backends.sequence import (
    DeleteT,
    Rightmost,
    SeqRule,
    SuccessorT,
    letters_task,
)
from twinworld.discriminate import diagnose
from twinworld.engine import solve_all


def rule(title):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def word(state_or_key):
    symbols = state_or_key[1] if isinstance(state_or_key, tuple) else state_or_key.symbols
    return "".join(symbols)


SEQ = twinworld.get_representation("sequence")

rule("1. abc -> abd ; ijk -> ?   (the classic, solved and certified)")
task = letters_task(train=[("abc", "abd"), ("ijk", "ijl")], test=[("mno", "mnp")])
t0 = time.perf_counter()
rep = twinworld.model(task)
ms = (time.perf_counter() - t0) * 1000
sol = rep.solution
print(f"\ninduced rule    : {' ; '.join(map(str, sol.program))}")
print(f"strategy        : {sol.strategy} ({sol.programs_tried} program(s) tried, {ms:.1f} ms)")
print(f"test answer     : mno -> {word(sol.test_traces[0].outcome)}")

rule("2. one demonstration is AMBIGUOUS — diagnosed, with the separating probe")
one_demo = letters_task(train=[("abc", "abd")], test=[("ijk", "ijl")])
fits = solve_all(one_demo, SEQ.candidate_primitives(one_demo), max_depth=1)
report = diagnose(one_demo, fits)
print(f"\n{len(fits)} fitting rule(s) in {len(report.classes)} behavioural class(es); {report}")
print(f"the readings part ways on the probe: {word(report.probe)!r}")
for cls, out in zip(report.classes, report.outputs):
    answer = word(out) if out is not None else "(inapplicable)"
    print(f"  {str(cls[0][0]):55s} -> {answer}")

rule("3. counterfactual alphabets — the manipulation as an intervention")
permuted = tuple("qwertyuiopasdfghjklzxcvbnm")


def succ(sym):
    return permuted[permuted.index(sym) + 1]


swap = SeqRule(Rightmost(), SuccessorT(permuted))
cfs = twinworld.compute(twinworld.identify(rep, Interventional(step=0, alternative=swap)))
print("\nintervening on the SOLVED task (swap the successor structure):")
print(cfs.items[0].narrative)

permuted_task = letters_task(
    train=[("abc", "ab" + succ("c")), ("ijk", "ij" + succ("k"))],
    test=[("mno", "mn" + succ("o"))],
)
try:
    twinworld.model(permuted_task)
    print("unexpected: the standard vocabulary should not fit a permuted world")
except twinworld.UnsolvedTaskError as err:
    print(f"\nstandard vocabulary on the permuted world: UNSOLVED (honest) — {err}")
t0 = time.perf_counter()
solved = twinworld.model(permuted_task, primitives=[swap], induction="never", max_depth=1)
ms = (time.perf_counter() - t0) * 1000
print(
    f"permuted-successor vocabulary: solved in {ms:.1f} ms — "
    f"mno -> {word(solved.solution.test_traces[0].outcome)}"
)

rule("4. abduction on sequences — ranked preimages of a deletion")
delete_rule = SeqRule(Rightmost(), DeleteT())
observed = SEQ.parse("abc")
alts = rank_preimages(delete_rule, observed, limit=32)
print(f"\n{len(alts.items)} candidate pre-states whose rightmost letter was deleted:")
for pre, score in list(zip(alts.items, alts.scores))[:5]:
    print(f"  {word(pre):8s} order {score['order']:2d}  proximity {score['proximity']:.1f}")

rule("done — the analogy machinery, on the domain it was invented for")
