"""Render a task's full dowhat process page to a self-contained HTML file.

Run:  python examples/visual_report.py                        # ARC a79310a0
      python examples/visual_report.py 007bbfb7 -o out.html --open
      python examples/visual_report.py --blocks --open        # no arckit needed

(For browsing the whole corpus interactively: python -m dowhat.viz)
"""

import argparse
import sys

from dowhat.viz import full_report, save_report, show


def blocks_demo():
    from dowhat.domains.blocks import build_grid, candidate_moves, task_from_towers

    task = task_from_towers(
        train=[
            ([[1, 2], [], []], [[], [2], [1]]),
            ([[1, 2], [3], []], [[], [3, 2], [1]]),
        ],
        test=[([[1, 2], [5], []], [[], [5, 2], [1]])],
    )
    kwargs = {
        "primitives": candidate_moves(task),
        "induction": "never",
        "max_depth": 2,
        "foil": build_grid([[], [5, 2, 1], []]),  # why not block 1 ON TOP of block 2?
        "foil_on": "test[0]",
    }
    return task, kwargs


def main():
    parser = argparse.ArgumentParser(description="Write a dowhat process page as HTML.")
    parser.add_argument("task_id", nargs="?", default="a79310a0")
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    parser.add_argument("--blocks", action="store_true", help="blocks-world demo task")
    args = parser.parse_args()

    if args.blocks:
        task, kwargs = blocks_demo()
        default_out = "report_blocks.html"
    else:
        try:
            from dowhat.domains.arc import load_task

            task = load_task(args.task_id)
        except ModuleNotFoundError:
            sys.exit("arckit is not installed — pip install 'dowhat[arc]' (or use --blocks)")
        except KeyError as err:
            sys.exit(str(err))
        kwargs = {}
        default_out = f"report_{args.task_id}.html"

    doc = full_report(task, **kwargs)
    path = save_report(doc, args.out or default_out)
    print(f"wrote {path}")
    if args.open_browser:
        show(doc, path)


if __name__ == "__main__":
    main()
