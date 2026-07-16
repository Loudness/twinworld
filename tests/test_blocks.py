"""Blocks world: the non-ARC domain proving the core is domain-general."""

import pytest

import twinworld
from twinworld import Contrastive, PertinentNegative, UnsolvedTaskError
from twinworld.discriminate import diagnose
from twinworld.domains.blocks import (
    MoveBlock,
    build_grid,
    candidate_moves,
    task_from_towers,
    towers_of,
)
from twinworld.engine import solve_all
from twinworld.representation import parse_grid


@pytest.fixture
def stack_task():
    """The same two-move plan (2 to column 1, then 1 to column 2) must solve
    two instances whose destination heights differ — gravity in action."""
    return task_from_towers(
        train=[
            ([[1, 2], [], []], [[], [2], [1]]),
            ([[1, 2], [3], []], [[], [3, 2], [1]]),
        ],
        test=[([[1, 2], [5], []], [[], [5, 2], [1]])],
    )


def plan(task):
    return twinworld.model(task, primitives=candidate_moves(task), induction="never", max_depth=2)


def test_grid_round_trip():
    towers = [[1, 2], [3], []]
    assert towers_of(build_grid(towers)) == [[1, 2], [3], []]


def test_move_preconditions():
    s = parse_grid(build_grid([[1, 2], [3], []]))
    assert MoveBlock(1, 2).apply(s) is None  # buried under 2: not clear
    assert MoveBlock(2, 0).apply(s) is None  # already in that column
    assert MoveBlock(9, 2).apply(s) is None  # no such block
    moved = MoveBlock(2, 1).apply(s)
    assert towers_of(moved.grid) == [[1], [3, 2], []]  # lands ON TOP of 3


def test_move_landing_is_context_dependent():
    onto_empty = MoveBlock(2, 1).apply(parse_grid(build_grid([[1, 2], [], []])))
    assert towers_of(onto_empty.grid) == [[1], [2], []]  # bottom of the column
    full = parse_grid(build_grid([[1], [2, 3, 4, 5], []]))
    assert MoveBlock(1, 1).apply(full) is None  # destination column is full


def test_move_preimage_round_trip():
    s = parse_grid(build_grid([[1, 2], [3], []]))
    t = MoveBlock(2, 1).apply(s)
    assert s in list(MoveBlock(2, 1).preimage(t))
    assert MoveBlock(2, 1).exact_preimage


def test_plan_induction_with_domain_primitives(stack_task):
    rep = plan(stack_task)
    assert rep.solution.program == (MoveBlock(2, 1), MoveBlock(1, 2))
    assert rep.solution.test_traces[0].outcome.towers == ((), (5, 2), (1,))


def test_arc_vocabulary_cannot_express_gravity():
    """Block 2's displacement differs across pairs (landing height depends on
    the destination stack), so no translation-based rule fits — the domain
    mechanism is genuinely necessary. Posed on the GRID serialization, whose
    default primitives are the ARC vocabulary."""
    grid_task = task_from_towers(
        train=[
            ([[1, 2], [], []], [[], [2], [1]]),
            ([[1, 2], [3], []], [[], [3, 2], [1]]),
        ],
        test=[([[1, 2], [5], []], [[], [5, 2], [1]])],
        representation="grid",
    )
    with pytest.raises(UnsolvedTaskError):
        twinworld.model(grid_task)


def test_contrastive_why_here_and_not_there(stack_task):
    rep = plan(stack_task)
    foil = [[], [5, 2, 1], []]  # what if block 1 were ON block 2?
    cfs = twinworld.compute(twinworld.identify(rep, Contrastive(foil, on="test[0]")))
    assert cfs.items
    assert all(item.metrics.sparsity == 1 for item in cfs.items)
    assert cfs.responsibility == {0: 0.0, 1: 1.0}  # only the final placement
    assert any("move block 1 to column 1" in item.narrative for item in cfs.items)


def test_pertinent_negatives_expose_plan_presuppositions(stack_task):
    rep = plan(stack_task)
    pn = twinworld.compute(
        twinworld.identify(
            rep,
            PertinentNegative(max_cells=1, separated=False, colours=(9,), max_witnesses=8),
        )
    )
    texts = [item.narrative for item in pn.items]
    assert any("no longer apply" in t for t in texts)  # clearance: a block on top of 2
    assert any("outcome would change" in t for t in texts)  # landing-height dependence


def test_plan_is_behaviourally_determined(stack_task):
    fits = solve_all(stack_task, candidate_moves(stack_task), max_depth=2)
    assert fits
    report = diagnose(stack_task, fits)
    assert not report.underdetermined
