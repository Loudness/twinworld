# dowhat

Counterfactual reasoning and explanation over **symbolic state-transition traces** — a
[DoWhy](https://github.com/py-why/dowhy)-style staged API (`model → identify → compute → refute`)
where the system is a deterministic solver rather than a statistical model, abduction is exact
trace replay rather than noise inversion, and counterfactuals are therefore point-identified.

First domain: the [ARC challenge](https://github.com/fchollet/ARC-AGI). The solving core is
non-connectionist (no neural networks, no LLMs); neural components can plug in later through the
backend registry without touching the core.

## Pipeline

```python
import dowhat

rep = dowhat.model(task)                      # fit under every abstraction; shortest program wins
query = dowhat.Interventional(step=1, alternative=...)
identified = dowhat.identify(rep, query)      # structural well-posedness check
cfs = dowhat.compute(identified)              # twin-world counterfactuals + per-item metrics
report = dowhat.refute(rep)                   # placebo-intervention battery

# segmentation is a recorded, revisable decision — so it is intervenable too:
dowhat.compute(dowhat.identify(rep, dowhat.Representational("mcc")))

# contrastive: why this outcome and not that one? Answered by the smallest
# program-edit set reaching the foil — CERTIFIED minimal (search is exhaustive
# per k) — or by a certificate that the foil is unreachable, with a
# Chockler-Halpern responsibility profile over the program's steps.
cfs = dowhat.compute(dowhat.identify(rep, dowhat.Contrastive(foil_grid, on="test[0]")))
cfs.responsibility  # e.g. {0: 0.5, 1: 0.5}
```

Because the domain is deterministic, counterfactual claims here are **certificates,
not estimates** — validated on a generated benchmark whose latent programs are
known by construction (100% recovery, minimality gap 0). And programs that fit
the demonstrations but are not behaviourally unique are detected *before* the
test pair is consulted: `dowhat.discriminate.diagnose` groups all fitting
programs into counterfactual-probe equivalence classes and exhibits the probe
on which they part ways.

The generality claim is substantiated, not asserted: `dowhat.domains.blocks`
runs the identical core on **blocks world** — the canonical STRIPS planning
domain — where the domain plugin supplies only a perception adapter and a
`MoveBlock` primitive with real preconditions and gravity (the same move
displaces a block differently in different states, provably inexpressible as
a translation rule). Contrastive plan edits, pertinent negatives (clearance
and landing-height presuppositions), necessity analysis, and determinism
diagnosis all work unchanged (`examples/blocks_world.py`).

Two further solving strategies complement analogy: `model(induction="asp")`
hands the program search to **clingo** — choice rules over (selector,
transform) steps with negation-as-failure in the search space, object
dynamics as ASP rules, and every answer set verified through the exact
engine (ASP proposes, the engine disposes; the solid/separated-object
fragment is declared, and fragment mismatches are counted, never accepted).
And `dowhat.assess` / `dowhat.predict` turn underdetermination diagnosis into
a **confidence gate**: hypotheses that fit the demonstrations are grouped
into behavioural classes and applied to the test input — one class (or
unanimity) gates high, disagreement gates low and `predict` abstains with the
alternatives. The gate never sees the test output.

Negation runs through the library three ways (thesis Experiment 4): `Not(...)`
selectors in the rule language ("recolour everything *except* the largest" —
which solved an ARC task no positive selector could, at the measured cost of
more underdetermined fits); `dowhat.PertinentNegative` — what must be minimally
*absent* for the outcome to hold, answered with catalogue-bounded certificates
(and doubling as a discriminator: fragile size-based hypotheses have absence
dependencies that colour-based ones lack); and an optional ASP cross-check
(`pip install dowhat[asp]`) where clingo re-derives every selector under
negation-as-failure and must agree with the Python semantics.

Three abstraction schemes ship today (`cc4`, `cc8`, `mcc` — colour-blind composites);
on the 1000-task ARC training corpus no single scheme explains more than 22.5% of
tasks as pure object transformations, but the union reaches 32.6% — plural,
revisable segmentation is load-bearing, not a luxury.

Search is analogy-first: SME-style structure mapping between each train pair's
object graphs yields per-object deltas, deltas generalize across pairs into
candidate object rules (selector + transform), and the engine merely *verifies*
them — analogy proposes, search disposes. On tasks it solves, the analogy path
tries a median of ~1 program where blind enumeration tried hundreds, and
train-fitting programs that fail the held-out test are reported as
*underdetermined*, not solved — that gap is a research target, not noise.

Research grounding and full citations: [docs/research-digest.md](docs/research-digest.md).

## Development

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e '.[arc,dev]'
./.venv/bin/pytest
./.venv/bin/python examples/vertical_slice.py
```

Part of a PhD project on counterfactual explanations and analogy-making in symbolic AI.
