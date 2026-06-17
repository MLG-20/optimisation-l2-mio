"""Tests de la saisie tolérante (notations naturelles)."""

import sympy as sp

from analyse import parsing
from analyse import une_variable as uv, deux_variables as dv


def test_multiplication_implicite():
    x = sp.Symbol("x")
    assert parsing.lire("2x", {"x": x}) == 2 * x
    assert parsing.lire("3x + 1", {"x": x}) == 3 * x + 1


def test_puissance_chapeau_et_exposant():
    x = sp.Symbol("x")
    assert parsing.lire("x^2", {"x": x}) == x**2
    assert parsing.lire("x²", {"x": x}) == x**2


def test_racine_symbole():
    x = sp.Symbol("x")
    assert parsing.lire("√x", {"x": x}) == sp.sqrt(x)
    assert parsing.lire("√(x)", {"x": x}) == sp.sqrt(x)


def test_produit_point():
    x, y = sp.symbols("x y")
    assert parsing.lire("3·x·y", {"x": x, "y": y}) == 3 * x * y


def test_integration_dans_modules():
    # La saisie naturelle doit fonctionner via les modules d'analyse.
    r = uv.analyser("2x")          # f(x) = 2x
    assert r.derivee_premiere == 2
    r2 = dv.analyser("x² + y²")    # paraboloïde
    assert any(pc.point == (0, 0) and "minimum" in pc.nature
               for pc in r2.points_critiques)


def test_saisie_invalide():
    x = sp.Symbol("x")
    try:
        parsing.lire("x +* ", {"x": x})
    except ValueError:
        return
    raise AssertionError("aurait dû lever ValueError")
