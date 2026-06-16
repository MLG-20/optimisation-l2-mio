"""Tests de l'analyse à deux variables."""

import pytest

from analyse import deux_variables as dv


def nature_en(resultat, x0, y0):
    for pc in resultat.points_critiques:
        if pc.point == (x0, y0):
            return pc.nature
    return None


def test_paraboloide_minimum():
    r = dv.analyser("x**2 + y**2")
    assert len(r.points_critiques) == 1
    pc = r.points_critiques[0]
    assert pc.point == (0, 0)
    assert pc.nature == "minimum local"


def test_paraboloide_maximum():
    r = dv.analyser("-x**2 - y**2")
    assert r.points_critiques[0].nature == "maximum local"


def test_point_col():
    r = dv.analyser("x**2 - y**2")
    assert r.points_critiques[0].nature == "point col (selle)"


def test_monkey_saddle_indetermine():
    # x**3 + y**3 - 3xy : un minimum en (1,1) et un col en (0,0)
    r = dv.analyser("x**3 + y**3 - 3*x*y")
    assert nature_en(r, 0, 0) == "point col (selle)"
    assert nature_en(r, 1, 1) == "minimum local"


def test_det_nul_minimum():
    # x⁴+y⁴ : det(H)=0 en (0,0), mais c'est un minimum (test numérique)
    r = dv.analyser("x**4 + y**4")
    assert nature_en(r, 0, 0).startswith("minimum local")


def test_det_nul_maximum():
    r = dv.analyser("-x**4 - y**4")
    assert nature_en(r, 0, 0).startswith("maximum local")


def test_det_nul_ni_min_ni_max():
    r = dv.analyser("x**3 + y**3")
    assert nature_en(r, 0, 0).startswith("ni minimum ni maximum")


def test_expression_invalide():
    with pytest.raises(ValueError):
        dv.analyser("x +* y")


def test_symbole_inattendu():
    with pytest.raises(ValueError):
        dv.analyser("x + y + z")
