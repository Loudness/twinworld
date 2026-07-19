"""The sequence (letter-string) backend: identity laws, rule induction,
counterfactual alphabets, and the analogy machinery on its home domain."""

import pytest

import twinworld
from twinworld import Interventional, PertinentNegative, UnsolvedTaskError, conformance_battery
from twinworld.backends.sequence import (
    ALPHABET,
    DeleteT,
    Leftmost,
    Rightmost,
    SeqRule,
    SetLetterTo,
    SuccessorT,
    letters_task,
)
from twinworld.discriminate import diagnose
from twinworld.engine import solve_all
from twinworld.mechanisms import All, ByAttr

SEQ = twinworld.get_representation("sequence")

SAMPLES = (tuple("abc"), tuple("ijjk"), tuple("z"))

MECHANISMS = (
    SeqRule(Rightmost(), SuccessorT(ALPHABET)),
    SeqRule(ByAttr("letter", "a"), SetLetterTo("b")),
    SeqRule(All(), DeleteT()),
    SeqRule(Leftmost(), SetLetterTo("q")),
)


def test_parse_canon_key_round_trip():
    for raw in SAMPLES:
        assert SEQ.parse(raw).key == SEQ.canon(raw)
    assert SEQ.canon("abc") == SEQ.canon(("a", "b", "c"))  # strings welcome


def test_key_abstraction_invariance_letters_vs_runs():
    for raw in SAMPLES:
        assert len({SEQ.parse(raw, name).key for name in SEQ.abstractions}) == 1
        assert SEQ.parse(raw, "letters") == SEQ.parse(raw, "runs")


def test_runs_segmentation_groups():
    runs = SEQ.parse(tuple("aabcc"), "runs").objects
    assert [(o.attributes["kind"], o.attributes["length"]) for o in runs] == [
        ("same", 2),
        ("successor", 2),  # "bc" advances by one
        ("single", 1),  # the second "c" starts anew
    ]


def test_conformance_battery_sequence_passes():
    report = conformance_battery(SEQ, SAMPLES, mechanisms=MECHANISMS)
    assert report.passed, str(report)
    by_name = {row.name: row for row in report.rows}
    assert by_name["L5_preimage_sound"].passed is True
    assert by_name["L6_exact_preimage_spot"].passed is True


def test_successor_preimage_is_exact_predecessor():
    rule = SeqRule(Rightmost(), SuccessorT(ALPHABET))
    state = SEQ.parse("abd")
    assert list(rule.preimage(state)) == [SEQ.parse("abc")]
    assert rule.exact_preimage


def test_induce_rules_proposes_rightmost_successor():
    task = letters_task(train=[("abc", "abd"), ("ijk", "ijl")], test=[("mno", "mnp")])
    rules = twinworld.induce_rules(task)
    assert SeqRule(Rightmost(), SuccessorT(ALPHABET)) in rules
    rep = twinworld.model(task)
    assert rep.solution.program == (SeqRule(Rightmost(), SuccessorT(ALPHABET)),)
    assert rep.solution.strategy == "analogy"
    assert rep.solution.test_traces[0].outcome.key == ("sequence", tuple("mnp"))


def test_one_demo_ambiguity_diagnosed_with_probe():
    task = letters_task(train=[("abc", "abd")], test=[("ijk", "ijl")])
    fits = solve_all(task, SEQ.candidate_primitives(task), max_depth=1)
    assert (SeqRule(Rightmost(), SuccessorT(ALPHABET)),) in fits
    assert (SeqRule(Rightmost(), SetLetterTo("d")),) in fits  # the classic rival reading
    report = diagnose(task, fits)
    assert report.underdetermined
    assert report.probe is not None  # the input on which the readings part ways


def test_counterfactual_alphabet_is_interventional_swap():
    permuted = tuple("qwertyuiopasdfghjklzxcvbnm")
    task = letters_task(train=[("abc", "abd"), ("ijk", "ijl")], test=[("mno", "mnp")])
    rep = twinworld.model(task)
    swap = SeqRule(Rightmost(), SuccessorT(permuted))
    cfs = twinworld.compute(twinworld.identify(rep, Interventional(step=0, alternative=swap)))
    assert any("permuted alphabet" in item.narrative for item in cfs.items)
    # and a permuted-successor WORLD is honestly unsolvable by the standard vocabulary
    def succ(sym: str) -> str:
        return permuted[permuted.index(sym) + 1]

    permuted_task = letters_task(
        train=[("abc", "ab" + succ("c")), ("ijk", "ij" + succ("k"))],
        test=[("mno", "mn" + succ("o"))],
    )
    with pytest.raises(UnsolvedTaskError):
        twinworld.model(permuted_task)
    solved = twinworld.model(
        permuted_task,
        primitives=[SeqRule(Rightmost(), SuccessorT(permuted))],
        induction="never",
        max_depth=1,
    )
    assert solved.solution.test_traces[0].outcome.key == ("sequence", tuple("mn" + succ("o")))


def test_backtracking_insertion_is_legal_with_none_frame():
    task = letters_task(train=[("abc", "abd"), ("ijk", "ijl")], test=[("mno", "mnp")])
    rep = twinworld.model(task)
    cfs = twinworld.compute(twinworld.identify(rep, twinworld.Backtracking(tuple("abcx"))))
    assert cfs.items[0].counterfactual.applicable  # a longer input is a fine backtrack


def test_pn_append_witness_exposes_rightmost_dependence():
    task = letters_task(train=[("abc", "abd"), ("ijk", "ijl")], test=[("mno", "mnp")])
    rep = twinworld.model(task)
    pn = twinworld.compute(twinworld.identify(rep, PertinentNegative(max_cells=1)))
    assert any(
        "stood at the end" in item.narrative
        and ("no longer apply" in item.narrative or "outcome would change" in item.narrative)
        for item in pn.items
    )  # a trailing letter steals the Rightmost selection (here: 'z', whose successor
    # does not exist — the plan stops applying)


def test_plausible_rejects_non_alphabet_symbol():
    assert SEQ.plausible(SEQ.parse("abc")) is True
    assert SEQ.plausible(SEQ.parse(("a", "3"))) is False
