"""Tests de l'optimisation sous contrainte (Lagrange)."""

import sympy as sp
import pytest

from analyse import lagrange as lg


def trouver(resultat, point):
    for pc in resultat.points_critiques:
        if pc.point == point:
            return pc
    return None


def test_produit_sous_somme():
    # max de x*y sous x + y = 10  ->  (5, 5), f = 25, maximum
    r = lg.analyser("x*y", "x + y - 10")
    pc = trouver(r, (5, 5))
    assert pc is not None
    assert pc.valeur == 25
    assert pc.nature == "maximum local sous contrainte"


def test_somme_sur_cercle():
    # x + y sur le cercle unité -> +√2 (max) et -√2 (min)
    r = lg.analyser("x + y", "x**2 + y**2 - 1")
    valeurs = sorted(sp.nsimplify(pc.valeur) for pc in r.points_critiques)
    assert valeurs == [-sp.sqrt(2), sp.sqrt(2)]


def test_distance_min():
    # min de x²+y² sous x+y=1 -> (1/2,1/2), f=1/2
    r = lg.analyser("x**2 + y**2", "x + y - 1")
    pc = trouver(r, (sp.Rational(1, 2), sp.Rational(1, 2)))
    assert pc is not None
    assert pc.valeur == sp.Rational(1, 2)
    assert pc.nature == "minimum local sous contrainte"


def test_produit_sur_cercle():
    r = lg.analyser("x*y", "x**2 + y**2 - 8")
    natures = {pc.point: pc.nature for pc in r.points_critiques}
    assert natures[(2, 2)] == "maximum local sous contrainte"
    assert natures[(-2, -2)] == "maximum local sous contrainte"
    assert natures[(2, -2)] == "minimum local sous contrainte"
    assert natures[(-2, 2)] == "minimum local sous contrainte"


def test_substitution_simple():
    # max x*y sous x+y=10 par substitution -> (5,5), f=25
    r = lg.resoudre_par_substitution("x*y", "x + y - 10")
    assert len(r.points_critiques) == 1
    pc = r.points_critiques[0]
    assert pc.point == (5, 5)
    assert pc.valeur == 25
    assert pc.nature == "maximum local sous contrainte"


def test_substitution_egale_lagrange():
    # Feuille 2 Exo 6 : les deux méthodes donnent les mêmes valeurs/natures.
    f, g = "x**2*y + 2*x**2 - 2*x - y + 1", "x - y - 1"
    subst = lg.resoudre_par_substitution(f, g)
    lagr = lg.analyser(f, g)
    cle = lambda pc: (sp.nsimplify(pc.valeur), pc.nature)
    assert {cle(pc) for pc in subst.points_critiques} == \
           {cle(pc) for pc in lagr.points_critiques}


def test_expression_invalide():
    with pytest.raises(ValueError):
        lg.analyser("x*y", "x +* y")


def test_symbole_inattendu():
    with pytest.raises(ValueError):
        lg.analyser("x*y*z", "x + y - 1")
