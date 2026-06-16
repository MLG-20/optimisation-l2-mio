"""Tests du simplexe et de la dualité (Feuille 4)."""

import pytest
import sympy as sp

from analyse import lineaire as lp, simplexe as sx, dualite as du


def test_simplexe_feuille4_exo1():
    pl = lp.programme((4, 5), "max",
                      [((2, 1), "<=", 800), ((1, 2), "<=", 700), ((0, 1), "<=", 300)])
    r = sx.resoudre(pl)
    assert r.statut == "optimal"
    assert r.valeur_optimale == 2200
    assert r.solution["x1"] == 300
    assert r.solution["x2"] == 200


def test_simplexe_refuse_min():
    pl = lp.programme((2, 3), "min", [((1, 1), ">=", 3)])
    with pytest.raises(ValueError):
        sx.resoudre(pl)


def test_dual_de_max():
    pl = lp.programme((4, 5), "max",
                      [((2, 1), "<=", 800), ((1, 2), "<=", 700)])
    d = du.dual(pl)
    assert d.sens == "min"
    assert d.objectif == (800, 700)            # = b du primal
    assert all(c.sens == ">=" for c in d.contraintes)
    # Première contrainte duale = première colonne de A, rhs = c1
    assert d.contraintes[0].coeffs == (2, 1)
    assert d.contraintes[0].rhs == 4


def test_dualite_forte_exo2():
    # min via dual : z* identique, et solution primale déductible
    pl = lp.programme((2, 3), "min",
                      [((4, 1), ">=", 5), ((3, 2), ">=", 6), ((1, 1), ">=", 3)])
    res_dual = sx.resoudre(du.dual(pl))
    assert res_dual.valeur_optimale == 6
    primal = du.solution_primale_depuis_dual(res_dual)
    assert primal["x1"] == 3
    assert primal["x2"] == 0
