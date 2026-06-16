"""Tests de l'étude de coût (Feuille 1, Exo 1)."""

import sympy as sp

from analyse import etude_cout as ec


def test_feuille1_exo1():
    # C(x) = x² + 16x + 256 sur [5, 50]
    r = ec.etudier("x**2 + 16*x + 256", 5, 50)
    x = r.variable
    assert r.cout_marginal == 2 * x + 16
    # f(x) = x + 16 + 256/x
    assert sp.simplify(r.cout_moyen - (x + 16 + 256 / x)) == 0
    # f'(x) = (x-16)(x+16)/x²
    assert sp.simplify(r.cout_moyen_derivee - (x - 16) * (x + 16) / x**2) == 0
    # minimum en x = 16, coût moyen 48
    assert r.quantite_optimale == 16
    assert r.cout_moyen_min == 48
    # propriété f(16) = C'(16) = 48
    assert r.cout_marginal_au_min == 48
    assert r.egalite_verifiee is True


def test_cout_de_40_paires():
    # Partie B.1 : C(40) = 2496 ; f(40) = 62.4
    r = ec.etudier("x**2 + 16*x + 256", 5, 50)
    x = r.variable
    assert r.cout.subs(x, 40) == 2496
    assert sp.Rational(r.cout_moyen.subs(x, 40)) == sp.Rational(312, 5)


def test_table_valeurs():
    r = ec.etudier("x**2 + 16*x + 256", 5, 50)
    table = ec.table_valeurs(r, pas=5)
    assert len(table) == 10          # x = 5, 10, ..., 50
    assert table[0]["x"] == 5
    assert table[-1]["x"] == 50
    assert table[7]["C(x)"] == 2496  # x = 40
