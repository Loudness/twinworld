from conftest import T

from dowhat import Recolor, Translate, parse_grid


def test_recolor_applies_to_all_objects_of_colour():
    s = parse_grid(T("30003", "00500", "00000"))
    t = Recolor(3, 4).apply(s)
    assert t.grid == T("40004", "00500", "00000")


def test_recolor_noop_is_inapplicable():
    s = parse_grid(T("500", "000", "000"))
    assert Recolor(3, 4).apply(s) is None
    assert Recolor(4, 4).apply(s) is None


def test_recolor_to_background_deletes():
    s = parse_grid(T("300", "050", "000"))
    t = Recolor(3, 0).apply(s)
    assert t.grid == T("000", "050", "000")


def test_recolor_preimage_contains_original():
    s = parse_grid(T("30003", "00500", "00000"))
    t = Recolor(3, 4).apply(s)
    assert s in list(Recolor(3, 4).preimage(t))


def test_recolor_preimage_enumerates_object_subsets():
    t = parse_grid(T("40004", "00000", "00000"))
    pres = list(Recolor(3, 4).preimage(t))
    grids = {p.grid for p in pres}
    # one or the other or both 4-objects may have been a 3 (empty flip is not a
    # preimage: recolor with no colour-3 object present is inapplicable)
    assert grids == {
        T("30004", "00000", "00000"),
        T("40003", "00000", "00000"),
        T("30003", "00000", "00000"),
    }
    for p in pres:
        assert Recolor(3, 4).apply(p) == t


def test_recolor_preimage_empty_when_src_survives():
    t = parse_grid(T("34000", "00000", "00000"))
    assert list(Recolor(3, 4).preimage(t)) == []


def test_translate_moves_only_target_colour():
    s = parse_grid(T("2050", "0000", "0000"))
    t = Translate(1, 0, colour=2).apply(s)
    assert t.grid == T("0050", "2000", "0000")


def test_translate_out_of_bounds_and_collision_inapplicable():
    s = parse_grid(T("0002", "0000", "0000"))
    assert Translate(0, 1, colour=2).apply(s) is None  # off the right edge
    s2 = parse_grid(T("2050", "0000", "0000"))
    assert Translate(0, 2, colour=2).apply(s2) is None  # lands on the 5


def test_translate_preimage_is_exact_inverse():
    s = parse_grid(T("0000", "0220", "0000"))
    t = Translate(1, 0, colour=2).apply(s)
    assert list(Translate(1, 0, colour=2).preimage(t)) == [s]
    assert Translate(1, 0, colour=2).exact_preimage


def test_apply_preimage_round_trip_property():
    mechs = [Recolor(2, 6), Translate(0, 1, colour=2), Translate(-1, 0, colour=2)]
    s = parse_grid(T("0000", "0220", "0050"))
    for m in mechs:
        t = m.apply(s)
        assert t is not None
        assert s in list(m.preimage(t)), f"{m} lost its own preimage"
