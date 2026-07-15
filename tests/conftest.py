import pytest

from twinworld import Task, as_grid


def T(*rows: str):
    """Compact grid literal: '00300' -> row of ints."""
    return as_grid([[int(ch) for ch in row] for row in rows])


@pytest.fixture
def recolor_task() -> Task:
    """Depth-1 task: every colour-3 object becomes colour 4; colour-5 spectator."""
    return Task(
        train=(
            (
                T("00000", "03300", "03000", "00050", "00000"),
                T("00000", "04400", "04000", "00050", "00000"),
            ),
            (
                T("33300", "00000", "05000", "00003", "00000"),
                T("44400", "00000", "05000", "00004", "00000"),
            ),
        ),
        test=(
            (
                T("00000", "00300", "00305", "00000", "00000"),
                T("00000", "00400", "00405", "00000", "00000"),
            ),
        ),
        task_id="synthetic-recolor",
    )


@pytest.fixture
def move_recolor_task() -> Task:
    """Depth-2 task: colour-2 objects move right by one AND become colour 6."""
    return Task(
        train=(
            (
                T("0000", "2200", "0000", "0007"),
                T("0000", "0660", "0000", "0007"),
            ),
            (
                T("2000", "0000", "0700", "0000"),
                T("0600", "0000", "0700", "0000"),
            ),
        ),
        test=(
            (
                T("0000", "0020", "0700", "0000"),
                T("0000", "0006", "0700", "0000"),
            ),
        ),
        task_id="synthetic-move-recolor",
    )


@pytest.fixture
def three_way_move_task() -> Task:
    """Colour 2 moves right, 3 moves down, 6 moves left — three rules needed.

    Blind enumeration at depth 2 cannot express three translations; the
    analogy path proposes exactly three rules and searches depth 3 trivially.
    """
    return Task(
        train=(
            (
                T("20000", "00300", "00000", "00060", "00000"),
                T("02000", "00000", "00300", "00600", "00000"),
            ),
            (
                T("00020", "30000", "00000", "00060", "00000"),
                T("00002", "00000", "30000", "00600", "00000"),
            ),
        ),
        test=(
            (
                T("00000", "02000", "00030", "06000", "00000"),
                T("00000", "00200", "00000", "60030", "00000"),
            ),
        ),
        task_id="synthetic-three-way",
    )


@pytest.fixture
def denoise_task() -> Task:
    """Delete the 1-cell specks, keep the big block — all the SAME colour.

    Colour-substitution cannot separate specks from block, so the blind path
    is hopeless at any depth; only smallest-objects + delete expresses it.
    """
    return Task(
        train=(
            (
                T("33000", "33000", "00030", "00000", "30000"),
                T("33000", "33000", "00000", "00000", "00000"),
            ),
            (
                T("00030", "00000", "00000", "00033", "00033"),
                T("00000", "00000", "00000", "00033", "00033"),
            ),
        ),
        test=(
            (
                T("00033", "00033", "00000", "30000", "00000"),
                T("00033", "00033", "00000", "00000", "00000"),
            ),
        ),
        task_id="synthetic-denoise",
    )


@pytest.fixture
def unsolvable_task() -> Task:
    """Object duplication: outside the slice vocabulary by construction."""
    return Task(
        train=(((T("30", "00"), T("33", "33"))),),
        test=((T("03", "00"), T("33", "33")),),
        task_id="synthetic-unsolvable",
    )


@pytest.fixture
def ambiguous_task() -> Task:
    """Largest object == the colour-2 object in every train pair: 'recolour the
    largest' and 'recolour colour 2' fit equally. Only a counterfactual probe
    (e.g. deleting the bar) separates them."""
    return Task(
        train=(
            (T("22200", "00000", "00030"), T("55500", "00000", "00030")),
            (T("00000", "02220", "30000"), T("00000", "05550", "30000")),
        ),
        test=((T("00222", "30000", "00000"), T("00555", "30000", "00000")),),
        task_id="synthetic-ambiguous",
    )


@pytest.fixture
def small_ambiguous_task() -> Task:
    """The size-2 colour-2 bar is the largest object in every train pair, so
    'recolour largest to 5' and 'recolour colour-2 to 5' fit equally."""
    return Task(
        train=(
            (
                T("220000", "000000", "000000", "000030", "000000"),
                T("550000", "000000", "000000", "000030", "000000"),
            ),
            (
                T("000000", "022000", "000000", "300000", "000000"),
                T("000000", "055000", "000000", "300000", "000000"),
            ),
        ),
        test=(
            (
                T("000022", "000000", "300000", "000000", "000000"),
                T("000055", "000000", "300000", "000000", "000000"),
            ),
        ),
        task_id="synthetic-small-ambiguous",
    )
