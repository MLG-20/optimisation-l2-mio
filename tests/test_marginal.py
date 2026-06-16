"""Tests de l'analyse marginale (Feuille 1)."""

import sympy as sp

from analyse import marginal as mg


def test_cout_moyen_feuille1_exo1():
    # C(x) = x^2 + 16x + 256 -> coût moyen min en x=16, valeur 48,
    # et coût marginal = 48 au minimum (propriété moyen = marginal).
    r = mg.cout_moyen("x**2 + 16*x + 256")
    assert r.quantite_optimale == 16
    assert r.cout_moyen_min == 48
    assert r.cout_marginal_au_min == 48


def test_marginal_evaluation():
    # C(x) = 20 + 500x - x^2 -> C'(x) = 500 - 2x, C'(50) = 400
    r = mg.fonction_marginale("20 + 500*x - x**2", point=50)
    assert r.marginale == 500 - 2 * sp.Symbol("x", real=True)
    assert r.valeur_marginale == 400


def test_profit_maximum():
    # R = 100x - x^2, C = 10x + 20 -> P = -x^2 + 90x - 20, max en x=45
    r = mg.profit("100*x - x**2", "10*x + 20")
    assert len(r.quantites_optimales) == 1
    x_opt, p_opt = r.quantites_optimales[0]
    assert x_opt == 45
    assert p_opt == 2005
