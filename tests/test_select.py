"""Phase C: selection among underdetermined programs (failure class 3) —
controlled ambiguous generation, deterministic policies, and active probe
resolution. The abstaining predict() default stays untouched."""

import random

import twinworld
from twinworld import ByColour, Largest, ObjectRule, RecolourTo, induce_rules, select
from twinworld.benchmark import random_ambiguous_task
from twinworld.discriminate import diagnose
from twinworld.engine import ApplyCache, solve_all
from twinworld.representation import as_grid, parse_grid
from twinworld.select import POLICIES, resolve_with_probe


def _instances(treacherous, n=8, bias=0.5):
    rng = random.Random(f"select-{treacherous}")
    out = []
    while len(out) < n:
        inst = random_ambiguous_task(rng, treacherous=treacherous, latent_bias=bias)
        if inst is not None:
            out.append(inst)
    return out


def _fits(task):
    return solve_all(task, induce_rules(task), max_depth=1)


def test_random_ambiguous_task_is_underdetermined():
    for task, latent in _instances(treacherous=True):
        fits = _fits(task)
        selectors = {type(p[0].selector) for p in fits}
        assert ByColour in selectors and Largest in selectors
        assert diagnose(task, fits).underdetermined


def test_treacherous_diverges_benign_agrees():
    cache = ApplyCache()

    def test_outputs(task, fits):
        outs = set()
        for program in fits:
            trace = cache.run(parse_grid(task.test[0][0]), program)
            outs.add(trace.outcome.key if trace else None)
        return outs

    assert all(
        len(test_outputs(task, _fits(task))) >= 2
        for task, _ in _instances(treacherous=True)
    )
    assert all(
        len(test_outputs(task, _fits(task))) == 1
        for task, _ in _instances(treacherous=False)
    )


def test_policies_return_valid_deterministic_index(ambiguous_task):
    rep = twinworld.model(ambiguous_task)
    report = twinworld.assess(rep)
    assert report.discrimination is not None
    n = len(report.discrimination.classes)
    for name, policy in POLICIES.items():
        one = policy(rep, report, random.Random(3))
        two = policy(rep, report, random.Random(3))
        assert one == two, name
        assert 0 <= one < n, name


def test_probe_stability_prefers_the_surviving_class(ambiguous_task):
    """Deleting the bar makes the ByColour reading inapplicable while the
    Largest reading keeps applying — stability must prefer the latter."""
    rep = twinworld.model(ambiguous_task)
    report = twinworld.assess(rep)
    index = POLICIES["probe_stability"](rep, report, random.Random(0))
    chosen = report.discrimination.classes[index]
    assert any(isinstance(p[0].selector, Largest) for p in chosen)


def test_fewest_absences_prefers_the_colour_class(small_ambiguous_task):
    """The Largest reading carries absence dependencies (a bigger added object
    steals selection); the ByColour reading earns a robustness certificate."""
    rep = twinworld.model(small_ambiguous_task)
    report = twinworld.assess(rep)
    index = POLICIES["fewest_absences"](rep, report, random.Random(0))
    chosen = report.discrimination.classes[index]
    assert any(isinstance(p[0].selector, ByColour) for p in chosen)


def test_select_answers_where_predict_abstains():
    task, _ = _instances(treacherous=True, n=1)[0]
    rep = twinworld.model(task)
    prediction, report = twinworld.predict(rep)
    assert prediction is None  # the calibrated default still abstains
    chosen, sreport = select(rep, policy="first")
    assert chosen in sreport.predictions
    assert sreport.confidence == "low"


def test_resolve_with_probe_reaches_the_latent_class(ambiguous_task):
    fits = _fits(ambiguous_task)
    oracle = (ObjectRule(ByColour(2), RecolourTo(5)),)
    chosen, queries = resolve_with_probe(ambiguous_task, fits, oracle)
    assert 1 <= queries <= 2  # one probe per surviving split
    # the survivor is probe-equivalent to the oracle's reading
    assert not diagnose(ambiguous_task, [chosen, oracle]).underdetermined
    cache = ApplyCache()
    for grid_in, expected in ambiguous_task.train:
        got = cache.run(parse_grid(grid_in), chosen).outcome.key
        assert got == as_grid(expected)


def test_resolve_respects_zero_budget(ambiguous_task):
    fits = _fits(ambiguous_task)
    oracle = (ObjectRule(ByColour(2), RecolourTo(5)),)
    chosen, queries = resolve_with_probe(ambiguous_task, fits, oracle, max_queries=0)
    assert queries == 0
    assert chosen == diagnose(ambiguous_task, fits).classes[0][0]
