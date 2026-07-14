from conftest import T

from dowhat import match_objects, parse_grid
from dowhat.representation import infer_background


def test_round_trip_reconstructs_grid():
    for grid in (
        T("00000", "03300", "03000", "00050", "00000"),
        T("000", "000", "000"),  # all background
        T("30", "03"),  # diagonal touch: two objects under cc4
        T("123", "456", "789"),
    ):
        assert parse_grid(grid).grid == grid


def test_cc4_separates_diagonal_and_distinct_colours():
    state = parse_grid(T("330", "305", "005"))
    by_colour = sorted((o.colour, o.size) for o in state.objects)
    # 3s at (0,0),(0,1),(1,0) form one 4-connected object; 5s at (1,2),(2,2) another
    assert by_colour == [(3, 3), (5, 2)]


def test_background_most_frequent_ties_prefer_zero():
    assert infer_background(T("07", "70")) == 0
    assert infer_background(T("77", "70")) == 7


def test_object_properties():
    state = parse_grid(T("00000", "03300", "03000", "00000", "00000"))
    (obj,) = state.objects
    assert obj.colour == 3
    assert obj.location == (1, 1)
    assert obj.size == 3
    assert obj.shape == frozenset({(0, 0), (0, 1), (1, 0)})


def test_shape_is_translation_invariant():
    a = parse_grid(T("330", "300", "000")).objects[0]
    b = parse_grid(T("000", "033", "030")).objects[0]
    assert a.shape == b.shape and a.location != b.location


def test_match_objects_tracks_move_and_recolour():
    before = parse_grid(T("22000", "00000", "00050"))
    after_move = parse_grid(T("02200", "00000", "00050"))
    pairs = {(-1 if x is None else x.location, -1 if y is None else y.location)
             for x, y in match_objects(before, after_move)}
    assert ((0, 0), (0, 1)) in pairs  # the bar, moved
    assert ((2, 3), (2, 3)) in pairs  # the spectator, unmoved

    after_recolour = parse_grid(T("66000", "00000", "00050"))
    matched = {
        (x.colour, y.colour) for x, y in match_objects(before, after_recolour) if x and y
    }
    assert (2, 6) in matched


def test_networkx_view_carries_ontology():
    g = parse_grid(T("33000", "00050", "33050")).to_networkx()
    assert g.graph["abstraction"] == "cc4"
    colours = {d["colour"] for _, d in g.nodes(data=True)}
    assert colours == {3, 5}
    relations = {d["relation"] for _, _, d in g.edges(data=True)}
    assert "same_colour" in relations
