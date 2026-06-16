"""Tests de la programmation linéaire (résolution graphique, Feuille 4)."""

import sympy as sp

from analyse import lineaire as lp


def test_feuille4_exo1_max():
    # max z = 4x1 + 5x2 -> optimum (300, 200), z* = 2200, C1 et C2 saturées
    pl = lp.programme((4, 5), "max",
                      [((2, 1), "<=", 800), ((1, 2), "<=", 700), ((0, 1), "<=", 300)])
    r = lp.resoudre_graphique(pl)
    assert r.statut == "optimal"
    assert r.optimum.point == (300, 200)
    assert r.optimum.valeur == 2200
    assert "C1" in r.optimum.contraintes_saturees
    assert "C2" in r.optimum.contraintes_saturees


def test_feuille4_exo2_min():
    # min z = 2xA + 3xB -> optimum (3, 0), z* = 6
    pl = lp.programme((2, 3), "min",
                      [((4, 1), ">=", 5), ((3, 2), ">=", 6), ((1, 1), ">=", 3)])
    r = lp.resoudre_graphique(pl)
    assert r.statut == "optimal"
    assert r.optimum.point == (3, 0)
    assert r.optimum.valeur == 6


def test_sommets_realisables():
    pl = lp.programme((4, 5), "max",
                      [((2, 1), "<=", 800), ((1, 2), "<=", 700), ((0, 1), "<=", 300)])
    r = lp.resoudre_graphique(pl)
    points = {s.point for s in r.sommets}
    assert (0, 0) in points
    assert (400, 0) in points
    assert (300, 200) in points


def test_fractions_exactes():
    # Vérifie que les coordonnées rationnelles sont exactes (pas de flottants)
    pl = lp.programme((2, 3), "min",
                      [((4, 1), ">=", 5), ((3, 2), ">=", 6), ((1, 1), ">=", 3)])
    r = lp.resoudre_graphique(pl)
    points = {s.point for s in r.sommets}
    assert (sp.Rational(2, 3), sp.Rational(7, 3)) in points
