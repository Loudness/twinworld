"""Milestone 2: plural abstractions, pixel objects, new properties, IoU matching."""

from conftest import T

from twinworld import Recolor, match_objects, parse_grid


def test_cc8_joins_diagonals_cc4_does_not():
    grid = T("30", "03")
    assert len(parse_grid(grid, "cc4").objects) == 2
    assert len(parse_grid(grid, "cc8").objects) == 1


def test_mcc_joins_touching_colours_into_composite():
    grid = T("230", "000", "005")
    state = parse_grid(grid, "mcc")
    assert len(state.objects) == 2  # {2,3} composite + the 5
    composite = next(o for o in state.objects if o.size == 2)
    assert composite.colours == frozenset({2, 3})
    assert composite.colour == 2  # tie on pixel count breaks to lowest colour


def test_all_schemes_render_round_trip():
    grid = T("230", "030", "005")
    for scheme in ("cc4", "cc8", "mcc"):
        assert parse_grid(grid, scheme).grid == grid


def test_shape_signature_rotation_invariant():
    l_shape = parse_grid(T("300", "300", "330")).objects[0]
    l_rotated = parse_grid(T("333", "300", "000")).objects[0]
    bar = parse_grid(T("333", "000", "000")).objects[0]
    assert l_shape.shape != l_rotated.shape
    assert l_shape.shape_signature == l_rotated.shape_signature
    assert l_shape.shape_signature != bar.shape_signature


def test_symmetries():
    plus = parse_grid(T("030", "333", "030")).objects[0]
    assert plus.symmetries == frozenset({"horizontal", "vertical", "rot180"})
    t_shape = parse_grid(T("333", "030", "000")).objects[0]
    assert t_shape.symmetries == frozenset({"vertical"})
    l_shape = parse_grid(T("300", "330", "000")).objects[0]
    assert l_shape.symmetries == frozenset()


def test_iou_matching_tracks_grown_object():
    before = parse_grid(T("2200", "0000"))
    after = parse_grid(T("2220", "0000"))  # same object, one cell longer
    (pair,) = [(x, y) for x, y in match_objects(before, after) if x and y]
    assert pair[0].size == 2 and pair[1].size == 3


def test_recolor_is_pixelwise_inside_composites():
    state = parse_grid(T("23", "00"), "mcc")
    out = Recolor(3, 9).apply(state)
    assert out.grid == T("29", "00")
    assert out.abstraction == "mcc"
