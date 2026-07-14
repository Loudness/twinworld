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
def unsolvable_task() -> Task:
    """Object duplication: outside the slice vocabulary by construction."""
    return Task(
        train=(((T("30", "00"), T("33", "33"))),),
        test=((T("03", "00"), T("33", "33")),),
        task_id="synthetic-unsolvable",
    )
