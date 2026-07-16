"""τ-abstraction cross-check: the native relational blocks world and its grid
serialization must commute (Beckers & Halpern's constructive abstraction, as a
test). τ maps a native state to its grid rendering at a fixed height; applying
a move natively then serializing must equal serializing then applying the same
move — except exactly where the grid frame cannot express the result (column
overflow), which must be the ONLY divergence."""

import twinworld
from twinworld import Contrastive, PertinentNegative, assess
from twinworld.domains.blocks import MoveBlock, build_grid, candidate_moves, task_from_towers
from twinworld.representation import parse_grid

HEIGHT = 4

TRAIN = [
    ([[1, 2], [], []], [[], [2], [1]]),
    ([[1, 2], [3], []], [[], [3, 2], [1]]),
]
TEST = [([[1, 2], [5], []], [[], [5, 2], [1]])]


def _tau(state):
    return parse_grid(build_grid([list(t) for t in state.towers], height=HEIGHT))


def _fits(state) -> bool:
    return all(len(t) <= HEIGHT for t in state.towers)


def _native_task():
    return task_from_towers(train=TRAIN, test=TEST)


def _grid_task():
    return task_from_towers(train=TRAIN, test=TEST, representation="grid")


def _plan(task):
    return twinworld.model(
        task, primitives=candidate_moves(task), induction="never", max_depth=2
    )


def test_moveblock_commutes_with_grid_serialization():
    task = _native_task()
    moves = candidate_moves(task)
    rep = twinworld.representation_of(task)
    frontier = [rep.parse(raw) for raw, _ in [*task.train, *task.test]]
    seen = set()
    checked = 0
    for _ in range(2):  # all states reachable within two moves
        next_frontier = []
        for s in frontier:
            if s.key in seen:
                continue
            seen.add(s.key)
            for m in moves:
                native = m.apply(s)
                serialized = m.apply(_tau(s))
                checked += 1
                if native is None:
                    assert serialized is None, f"{m} applies on the grid but not natively"
                elif _fits(native):
                    assert serialized is not None, f"{m} applies natively but not on the grid"
                    assert serialized.key == _tau(native).key, f"{m} does not commute with τ"
                    next_frontier.append(native)
                else:
                    assert serialized is None, f"{m} overflowed yet the grid accepted it"
        frontier = next_frontier
    assert checked > 50  # the sweep actually exercised the space


def test_overflow_is_the_serialization_boundary():
    from twinworld.backends.relational import state_from_towers

    tall = state_from_towers([[1, 2, 3, 5], [4], []])
    assert MoveBlock(4, 0).apply(tall) is not None  # native: unbounded
    assert MoveBlock(4, 0).apply(_tau(tall)) is None  # grid: column full


def test_native_and_grid_certificates_agree():
    rep_n, rep_g = _plan(_native_task()), _plan(_grid_task())

    # the same plan, as the same mechanism objects (the dual-dispatch payoff)
    assert rep_n.solution.program == rep_g.solution.program

    # contrastive: same certified minimality and the same responsibility profile
    foil_n = [[], [5, 2, 1], []]
    cfs_n = twinworld.compute(twinworld.identify(rep_n, Contrastive(foil_n, on="test[0]")))
    cfs_g = twinworld.compute(
        twinworld.identify(rep_g, Contrastive(build_grid(foil_n), on="test[0]"))
    )
    assert {i.metrics.sparsity for i in cfs_n.items} == {i.metrics.sparsity for i in cfs_g.items}
    assert cfs_n.responsibility == cfs_g.responsibility

    # pertinent negatives: the same presuppositions surface
    query = PertinentNegative(max_cells=1, separated=False, colours=(9,), max_witnesses=8)
    details = {}
    for name, rep in (("native", rep_n), ("grid", rep_g)):
        pn = twinworld.compute(twinworld.identify(rep, query))
        details[name] = {
            marker
            for item in pn.items
            for marker in ("no longer apply", "outcome would change")
            if marker in item.narrative
        }
    assert details["native"] == details["grid"] == {"no longer apply", "outcome would change"}

    # gate and refutation battery agree
    assert assess(rep_n).confidence == assess(rep_g).confidence == "high"
    assert twinworld.refute(rep_n).passed and twinworld.refute(rep_g).passed
