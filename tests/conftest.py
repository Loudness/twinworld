import pytest

from dowhat import Task, as_grid


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
