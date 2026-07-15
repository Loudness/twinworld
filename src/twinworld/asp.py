"""Optional ASP cross-check (thesis Experiment 4c): negation-as-failure as an
independent semantics for the selector language.

Selectors like "largest" are procedural in Python; here the same concepts are
written declaratively in answer-set programming, where "largest" literally IS
a negation — ``not bigger_exists(O)`` under negation-as-failure — and clingo
re-derives the selected sets on real task states. Agreement between the two
semantics is the "extra path validation" the thesis proposal asks negation to
provide. A full ASP solver over the transforms is deliberately out of scope
(flagged as its own milestone).

Requires the optional dependency: ``pip install twinworld[asp]``.
"""

from __future__ import annotations

from .engine import Solution
from .mechanisms import All, ByColour, Largest, Not, ObjectRule, Selector, Smallest
from .representation import StateGraph


def state_facts(state: StateGraph) -> str:
    """Compile a state's object graph to ASP facts."""
    lines = []
    for o in state.objects:
        r, c = o.location
        lines.append(
            f"obj({o.oid}). size({o.oid},{o.size}). colour({o.oid},{o.colour}). "
            f"location({o.oid},{r},{c})."
        )
    return "\n".join(lines)


def selector_rules(selector: Selector, head: str = "sel") -> str:
    """Encode a selector declaratively; superlatives and Not use NAF."""
    if isinstance(selector, All):
        return f"{head}(O) :- obj(O)."
    if isinstance(selector, ByColour):
        return f"{head}(O) :- obj(O), colour(O,{selector.colour})."
    if isinstance(selector, Largest):
        return (
            f"bigger_{head}(O) :- obj(O), size(O,S), obj(P), size(P,T), T > S.\n"
            f"{head}(O) :- obj(O), not bigger_{head}(O)."
        )
    if isinstance(selector, Smallest):
        return (
            f"smaller_{head}(O) :- obj(O), size(O,S), obj(P), size(P,T), T < S.\n"
            f"{head}(O) :- obj(O), not smaller_{head}(O)."
        )
    if isinstance(selector, Not):
        inner = selector_rules(selector.inner, head=f"inner_{head}")
        return inner + f"\n{head}(O) :- obj(O), not inner_{head}(O)."
    raise TypeError(f"no ASP encoding for selector {selector!r}")


def asp_select(selector: Selector, state: StateGraph) -> frozenset[int]:
    """The oids clingo selects under negation-as-failure semantics."""
    import clingo

    ctl = clingo.Control(["--warn=none"])
    ctl.add(
        "base", [], state_facts(state) + "\n" + selector_rules(selector) + "\n#show sel/1."
    )
    ctl.ground([("base", [])])
    selected: frozenset[int] = frozenset()
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            selected = frozenset(a.arguments[0].number for a in model.symbols(shown=True))
            break  # the encoding is stratified: exactly one stable model
    return selected


def crosscheck_selectors(solution: Solution) -> tuple[bool | None, str]:
    """Cross-check every ObjectRule selector against clingo on every state of
    every train trace. Returns (passed, detail) for the refutation row;
    passed None means not applicable (no clingo, or no object rules)."""
    try:
        import clingo  # noqa: F401
    except ImportError:
        return None, "clingo not installed — pip install 'twinworld[asp]'"
    rules = [m for m in solution.program if isinstance(m, ObjectRule)]
    if not rules:
        return None, "the induced program contains no object rules to cross-check"
    checks = 0
    for rule in rules:
        for trace in solution.train_traces:
            for state in trace.states:
                if not state.objects:
                    continue
                python_sel = frozenset(o.oid for o in rule.selector.select(state.objects))
                if asp_select(rule.selector, state) != python_sel:
                    return False, (
                        f"clingo and Python disagree on [{rule.selector}] in some "
                        f"reached state — the selector's declarative and procedural "
                        f"semantics have diverged"
                    )
                checks += 1
    return True, (
        f"clingo (negation-as-failure) agrees with the Python selectors on "
        f"{checks} state-checks"
    )
