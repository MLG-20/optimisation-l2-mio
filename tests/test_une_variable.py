"""Tests de l'analyse à une variable."""

import sympy as sp
import pytest

from analyse import une_variable as uv


def natures(resultat):
    return {sp.nsimplify(pc.abscisse): pc.nature for pc in resultat.points_critiques}


def test_polynome_cubique():
    r = uv.analyser("x**3 - 3*x")
    n = natures(r)
    assert n[sp.Integer(-1)] == "maximum local"
    assert n[sp.Integer(1)] == "minimum local"


def test_parabole_minimum():
    r = uv.analyser("x**2")
    assert len(r.points_critiques) == 1
    pc = r.points_critiques[0]
    assert pc.abscisse == 0
    assert pc.nature == "minimum local"


def test_point_inflexion_tangente_horizontale():
    # x**3 : dérivée nulle en 0 mais ce n'est pas un extremum
    r = uv.analyser("x**3")
    assert r.points_critiques[0].nature == "point d'inflexion à tangente horizontale"


def test_asymptote_horizontale():
    r = uv.analyser("1/x")
    horizontales = [val for val, _ in r.asymptotes["horizontales"]]
    assert sp.Integer(0) in horizontales
    assert sp.Integer(0) in r.asymptotes["verticales"]


def test_asymptote_oblique():
    r = uv.analyser("(x**2 + 1)/x")  # ~ x quand x -> ±∞
    pentes = [a for a, _, _ in r.asymptotes["obliques"]]
    assert sp.Integer(1) in pentes


def test_optimisation_intervalle_feuille2():
    # f(x) = x^3 - 12x + 9 sur [0, 3] : min = -7 en x=2, max = 9 en x=0
    r = uv.optimiser_sur_intervalle("x**3 - 12*x + 9", 0, 3)
    assert r.minimum.abscisse == 2
    assert r.minimum.valeur == -7
    assert r.maximum.abscisse == 0
    assert r.maximum.valeur == 9


def test_optimisation_intervalle_max_sur_borne():
    # x^2 sur [-1, 3] : min 0 en 0 (intérieur), max 9 en 3 (borne)
    r = uv.optimiser_sur_intervalle("x**2", -1, 3)
    assert r.minimum.valeur == 0
    assert r.maximum.valeur == 9
    assert r.maximum.type == "borne"


def test_expression_invalide():
    with pytest.raises(ValueError):
        uv.analyser("x +* 2")


def test_symbole_inattendu():
    with pytest.raises(ValueError):
        uv.analyser("x + y")
