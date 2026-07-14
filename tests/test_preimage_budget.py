"""Phase B: preimage-enumeration budgets — defaults byte-identical to the
historical constants; smaller budgets shrink the stream without breaking the
verifier; ground-truth abduction instances round-trip."""

import random

import dowhat
from conftest import T
from dowhat import ByColour, Delete, ObjectRule, parse_grid
from dowhat.benchmark import random_delete_instance
from dowhat.engine import abduce_inputs
from dowhat.mechanisms import DEFAULT_PREIMAGE_BUDGET, PreimageBudget


def _observed():
    """One colour-5 survivor gone quiet: a spacious state to abduce into."""
    return parse_grid(T("500000", "000000", "000000", "000000", "000000", "000000"))


def test_default_budget_matches_none():
    rule = ObjectRule(ByColour(3), Delete())
    state = _observed()
    default = list(rule.preimage(state))
    explicit = list(rule.preimage(state, DEFAULT_PREIMAGE_BUDGET))
    assert default == explicit
    assert len(default) == 324  # regression anchor for the historical caps


def test_small_budget_shrinks_and_still_verifies():
    rule = ObjectRule(ByColour(3), Delete())
    state = _observed()
    small = list(rule.preimage(state, PreimageBudget(anchors=5, pairs=5)))
    assert 0 < len(small) < len(list(rule.preimage(state)))
    assert all(rule.apply(pre) == state for pre in small)


def test_budget_threads_through_abduce_inputs(denoise_task):
    rep = dowhat.model(denoise_task)
    trace = rep.solution.train_traces[0]
    default = abduce_inputs(rep.solution.program, trace.outcome, limit=128)
    assert trace.states[0] in default  # the M9 recovery still holds
    tiny = abduce_inputs(
        rep.solution.program, trace.outcome, limit=128, budget=PreimageBudget(anchors=2)
    )
    assert 0 < len(tiny) < len(default)
    assert tiny == abduce_inputs(
        rep.solution.program, trace.outcome, limit=128, budget=PreimageBudget(anchors=2)
    )


def test_random_delete_instance_round_trips():
    for family in ("bycolour", "smallest", "not_bycolour"):
        rng = random.Random(family)
        made = 0
        for _ in range(30):
            inst = random_delete_instance(rng, family=family, n_palette=2)
            if inst is None:
                continue
            made += 1
            rule, pre, observed = inst
            assert rule.apply(pre) == observed
            assert pre != observed
        assert made >= 5
