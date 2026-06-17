"""Lecture tolérante des expressions saisies par l'utilisateur.

Accepte les notations « naturelles » d'un étudiant en plus de la syntaxe Python :
  - √x ou √(x)        → sqrt(x)
  - x², x³ …          → x**2, x**3
  - ^                 → ** (puissance)
  - ·  ou  ×          → *
  - multiplication implicite : 2x, 3xy, 2sin(x), 4x² …

Renvoie une expression SymPy, ou lève ValueError si la saisie est invalide.
"""

import re
import tokenize

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
)

_EXPOSANTS = {
    "⁰": "**0", "¹": "**1", "²": "**2", "³": "**3", "⁴": "**4",
    "⁵": "**5", "⁶": "**6", "⁷": "**7", "⁸": "**8", "⁹": "**9",
}


def nettoyer(texte):
    """Convertit la notation mathématique « naturelle » en syntaxe Python."""
    t = texte.strip()
    for symbole, remplacement in _EXPOSANTS.items():
        t = t.replace(symbole, remplacement)
    t = t.replace("·", "*").replace("×", "*").replace("÷", "/")
    t = t.replace("^", "**")
    t = t.replace("√(", "sqrt(")
    # √x, √2, √xy  ->  sqrt(x), sqrt(2), sqrt(xy)
    t = re.sub(r"√\s*([A-Za-z0-9.]+)", r"sqrt(\1)", t)
    return t


def lire(texte, symboles):
    """Lit `texte` comme une expression SymPy avec les `symboles` donnés.

    `symboles` : dict {nom: Symbol}. `texte` peut déjà être une expression
    SymPy (renvoyée telle quelle). Lève ValueError si la saisie est invalide.
    """
    if not isinstance(texte, str):
        return texte
    try:
        return parse_expr(nettoyer(texte), local_dict=dict(symboles),
                          transformations=_TRANSFORMATIONS)
    except (SyntaxError, TypeError, AttributeError, ValueError,
            tokenize.TokenError, sp.SympifyError) as err:
        raise ValueError(f"Expression invalide : {texte!r}") from err
