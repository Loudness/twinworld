"""ASP program-induction solver (the research digest's flagged niche).

clingo searches the ObjectRule program space itself: a choice rule picks one
(selector, transform) per step — negated selectors included, so negation-as-
failure is part of the *search space*, not just a cross-check — object state
is derived step by step (colour, cumulative offset, aliveness) under frame
axioms, applicability constraints mirror the Python engine's preconditions
(non-empty selection, bounds, collisions, no-ops), and the rendered final
state must equal every train pair's output grid.

**Declared fragment:** solid objects that stay separated (no merge/split
re-segmentation) — ARGA-style object dynamics. Because the Python engine
canonically re-parses after every step, a fragment mismatch is possible, so
every answer set is decoded and VERIFIED through the exact engine before
being trusted: ASP proposes, the engine disposes. Mismatches are counted,
never silently accepted.

Requires the optional dependency: ``pip install twinworld[asp]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import ApplyCache, Program, Solution, Task, UnsolvedTaskError
from .mechanisms import (
    All,
    ByColour,
    Delete,
    Largest,
    Mechanism,
    Not,
    ObjectRule,
    RecolourTo,
    Smallest,
    TranslateBy,
)
from .representation import as_grid, parse_grid

MAX_OBJECTS = 30
MAX_CELLS = 400

_DYNAMICS = """
% ---------------------------------------------------------------- choice
1 {{ chosen(T,SI,TI) : selcand(SI), transcand(TI) }} 1 :- step_index(T).

% ------------------------------------------------------------ state init
alive(K,0,O) :- obj(K,O).
col(K,0,O,C) :- colour0(K,O,C).
offr(K,0,O,0) :- obj(K,O).
offc(K,0,O,0) :- obj(K,O).

% ------------------------------------------- selection (on state T-1)
selected(K,T,O) :- chosen(T,SI,_), selmean(SI,all), alive(K,T-1,O).
selected(K,T,O) :- chosen(T,SI,_), selmean(SI,bycolour(C)), alive(K,T-1,O), col(K,T-1,O,C).
bigger(K,T,O)   :- step_index(T), alive(K,T-1,O), size(K,O,S),
                   alive(K,T-1,P), size(K,P,S2), S2 > S.
smaller(K,T,O)  :- step_index(T), alive(K,T-1,O), size(K,O,S),
                   alive(K,T-1,P), size(K,P,S2), S2 < S.
selected(K,T,O) :- chosen(T,SI,_), selmean(SI,largest), alive(K,T-1,O), not bigger(K,T,O).
selected(K,T,O) :- chosen(T,SI,_), selmean(SI,smallest), alive(K,T-1,O), not smaller(K,T,O).
selected(K,T,O) :- chosen(T,SI,_), selmean(SI,not_largest), alive(K,T-1,O), bigger(K,T,O).
selected(K,T,O) :- chosen(T,SI,_), selmean(SI,not_smallest), alive(K,T-1,O), smaller(K,T,O).
selected(K,T,O) :- chosen(T,SI,_), selmean(SI,not_bycolour(C)), alive(K,T-1,O),
                   col(K,T-1,O,C2), C2 != C.

% applicability: the step must select something in EVERY pair
some_sel(K,T) :- selected(K,T,_).
:- step_index(T), pair(K), not some_sel(K,T).

% ---------------------------------------------------------------- effects
deleted(K,T,O)    :- selected(K,T,O), chosen(T,_,TI), transmean(TI,delete).
alive(K,T,O)      :- step_index(T), alive(K,T-1,O), not deleted(K,T,O).
recoloured(K,T,O) :- selected(K,T,O), chosen(T,_,TI), transmean(TI,recolour(_)).
col(K,T,O,C2)     :- selected(K,T,O), alive(K,T,O), chosen(T,_,TI), transmean(TI,recolour(C2)).
col(K,T,O,C)      :- step_index(T), alive(K,T,O), col(K,T-1,O,C), not recoloured(K,T,O).
moved(K,T,O)      :- selected(K,T,O), chosen(T,_,TI), transmean(TI,translate(_,_)).
offr(K,T,O,R2)    :- selected(K,T,O), alive(K,T,O), chosen(T,_,TI),
                     transmean(TI,translate(DR,_)), offr(K,T-1,O,R), R2 = R + DR.
offc(K,T,O,C2)    :- selected(K,T,O), alive(K,T,O), chosen(T,_,TI),
                     transmean(TI,translate(_,DC)), offc(K,T-1,O,C), C2 = C + DC.
offr(K,T,O,R)     :- step_index(T), alive(K,T,O), offr(K,T-1,O,R), not moved(K,T,O).
offc(K,T,O,C)     :- step_index(T), alive(K,T,O), offc(K,T-1,O,C), not moved(K,T,O).

% no-op guard: a recolour step must change something in every pair
changes(K,T) :- selected(K,T,O), chosen(T,_,TI), transmean(TI,recolour(C2)),
                col(K,T-1,O,C), C != C2.
:- step_index(T), pair(K), chosen(T,_,TI), transmean(TI,recolour(_)), not changes(K,T).

% ------------------------------------- rendering, bounds, collisions
rendcell(K,T,O,R2,C2) :- step_index(T), alive(K,T,O), cell(K,O,R,C),
                         offr(K,T,O,DR), offc(K,T,O,DC), R2 = R + DR, C2 = C + DC.
:- rendcell(K,_,_,R,_), R < 0.
:- rendcell(K,_,_,R,_), gridsize(K,H,_), R >= H.
:- rendcell(K,_,_,_,C), C < 0.
:- rendcell(K,_,_,_,C), gridsize(K,_,W), C >= W.
:- rendcell(K,T,O,R,C), rendcell(K,T,O2,R,C), O < O2.

% ------------------------------------------------------------------ goal
rendered(K,R,C,Col) :- rendcell(K,{depth},O,R,C), col(K,{depth},O,Col).
:- rendered(K,R,C,Col), not out(K,R,C,Col).
:- out(K,R,C,Col), not rendered(K,R,C,Col).

#show chosen/3.
"""


@dataclass(frozen=True)
class AspResult:
    programs: tuple[Program, ...]  # engine-verified, minimal ASP depth
    proposed: int  # answer sets decoded
    verified: int  # survivors of exact Python verification
    depth: int | None  # depth at which programs were found


def _candidates(task: Task, abstraction: str):
    inputs = [parse_grid(i, abstraction) for i, _ in task.train]
    shared = set.intersection(*({o.colour for o in s.objects} for s in inputs)) if all(
        s.objects for s in inputs
    ) else set()
    selectors: list = [All(), Largest(), Smallest(), Not(Largest()), Not(Smallest())]
    selectors += [ByColour(c) for c in sorted(shared)]
    selectors += [Not(ByColour(c)) for c in sorted(shared)]
    bg = inputs[0].background
    palette = sorted(task.colours() - {bg})
    transforms: list = [Delete()]
    transforms += [RecolourTo(c) for c in palette]
    shifts = [d for d in range(-3, 4) if d != 0]
    transforms += [TranslateBy(d, 0) for d in shifts] + [TranslateBy(0, d) for d in shifts]
    return inputs, selectors, transforms


def _sel_term(selector) -> str:
    if isinstance(selector, All):
        return "all"
    if isinstance(selector, ByColour):
        return f"bycolour({selector.colour})"
    if isinstance(selector, Largest):
        return "largest"
    if isinstance(selector, Smallest):
        return "smallest"
    if isinstance(selector, Not):
        inner = selector.inner
        if isinstance(inner, Largest):
            return "not_largest"
        if isinstance(inner, Smallest):
            return "not_smallest"
        if isinstance(inner, ByColour):
            return f"not_bycolour({inner.colour})"
    raise TypeError(f"no ASP term for selector {selector!r}")


def _trans_term(transform) -> str:
    if isinstance(transform, Delete):
        return "delete"
    if isinstance(transform, RecolourTo):
        return f"recolour({transform.colour})"
    if isinstance(transform, TranslateBy):
        return f"translate({transform.dr},{transform.dc})"
    raise TypeError(f"no ASP term for transform {transform!r}")


def _facts(task: Task, inputs, selectors, transforms) -> str | None:
    lines = [f"pair(0..{len(task.train) - 1})."]
    for si, sel in enumerate(selectors):
        lines.append(f"selcand({si}). selmean({si},{_sel_term(sel)}).")
    for ti, trans in enumerate(transforms):
        lines.append(f"transcand({ti}). transmean({ti},{_trans_term(trans)}).")
    for k, (state, (_, out_grid)) in enumerate(zip(inputs, task.train)):
        out_grid = as_grid(out_grid)
        if len(state.objects) > MAX_OBJECTS:
            return None
        if sum(o.size for o in state.objects) > MAX_CELLS:
            return None
        if any(len(o.colours) > 1 for o in state.objects):
            return None  # fragment: solid objects only
        lines.append(f"gridsize({k},{state.height},{state.width}).")
        for o in state.objects:
            lines.append(f"obj({k},{o.oid}). size({k},{o.oid},{o.size}). "
                         f"colour0({k},{o.oid},{o.colour}).")
            lines.extend(f"cell({k},{o.oid},{r},{c})." for r, c in sorted(o.cells))
        bg = state.background
        for r, row in enumerate(out_grid):
            for c, colour in enumerate(row):
                if colour != bg:
                    lines.append(f"out({k},{r},{c},{colour}).")
    return "\n".join(lines)


def asp_solve(
    task: Task, abstraction: str = "cc4", max_depth: int = 2, limit: int = 32
) -> AspResult:
    """clingo-driven program induction, engine-verified, minimal-depth-first."""
    try:
        import clingo
    except ImportError:
        raise ImportError(
            "the ASP solver needs clingo — pip install 'twinworld[asp]'"
        ) from None

    inputs, selectors, transforms = _candidates(task, abstraction)
    facts = _facts(task, inputs, selectors, transforms)
    if facts is None:
        return AspResult((), 0, 0, None)  # outside the declared fragment

    cache = ApplyCache()
    expected = [as_grid(o) for _, o in task.train]

    for depth in range(1, max_depth + 1):
        program_text = (
            facts + f"\nstep_index(1..{depth}).\n" + _DYNAMICS.format(depth=depth)
        )
        ctl = clingo.Control(["--warn=none"])
        ctl.configuration.solve.models = str(limit)
        ctl.add("base", [], program_text)
        ctl.ground([("base", [])])
        proposals: list[Program] = []
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                steps: dict[int, tuple[int, int]] = {}
                for atom in model.symbols(shown=True):
                    t, si, ti = (arg.number for arg in atom.arguments)
                    steps[t] = (si, ti)
                proposals.append(
                    tuple(
                        ObjectRule(selectors[steps[t][0]], transforms[steps[t][1]])
                        for t in sorted(steps)
                    )
                )
        verified = []
        for program in proposals:
            for state, want in zip(inputs, expected):
                trace = cache.run(state, program)
                if trace is None or trace.outcome.key != want:
                    break
            else:
                verified.append(program)
        if verified:
            return AspResult(tuple(verified), len(proposals), len(verified), depth)
    return AspResult((), 0, 0, None)


def solution_from_asp(task: Task, abstraction: str, result: AspResult) -> Solution:
    """Materialize the first verified ASP program as an ordinary Solution."""
    if not result.programs:
        raise UnsolvedTaskError(f"ASP found no verified program for {task.task_id}")
    program = result.programs[0]
    cache = ApplyCache()
    train_traces = tuple(
        cache.run(parse_grid(i, abstraction), program) for i, _ in task.train
    )
    test_traces = tuple(
        t
        for i, _ in task.test
        if (t := cache.run(parse_grid(i, abstraction), program)) is not None
    )
    return Solution(
        task, program, train_traces, test_traces, cache,
        programs_tried=result.proposed, strategy="asp",
        searched=frozenset(cache.dag.nodes),
    )


def mechanism_pool(task: Task, abstraction: str = "cc4") -> list[Mechanism]:
    """The ObjectRule pool the ASP search ranges over (for external reuse)."""
    _, selectors, transforms = _candidates(task, abstraction)
    return [ObjectRule(s, t) for s in selectors for t in transforms]
