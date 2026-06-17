"""Analyse complète d'une fonction de deux variables f(x, y).

Calcule : gradient, points critiques (∇f = 0), matrice hessienne et nature
de chaque point critique (minimum, maximum, point col / selle) via le test
du déterminant hessien.

Lorsque det(H) = 0 (test hessien non concluant), on bascule sur un test
numérique de voisinage : on échantillonne f sur de petits cercles autour du
point pour décider entre minimum, maximum, ou ni l'un ni l'autre.
"""

import math
from dataclasses import dataclass, field

import sympy as sp

from . import parsing


@dataclass
class PointCritique2D:
    point: tuple           # (x0, y0)
    valeur: sp.Expr        # f(x0, y0)
    nature: str            # "minimum local", "maximum local", "point col", ...
    determinant: sp.Expr   # det de la hessienne au point
    f_xx: sp.Expr          # dérivée seconde par rapport à x au point


@dataclass
class ResultatDeuxVariables:
    expression: sp.Expr
    variables: tuple                       # (x, y)
    gradient: tuple                        # (f_x, f_y)
    hessienne: sp.Matrix
    points_critiques: list = field(default_factory=list)


def _indice(n):
    """Convertit un entier en chiffres indices (0 -> ₀, 12 -> ₁₂)."""
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _est_reel(val):
    if val.is_real is True:
        return True
    if val.is_real is False:
        return False
    try:
        c = complex(val)
        return abs(c.imag) < 1e-9
    except (TypeError, ValueError):
        return False


def _test_numerique_voisinage(expr, x, y, point, eps=1e-3, n=24):
    """Test de repli quand det(H) = 0 : on échantillonne f autour du point.

    On compare f sur de petits cercles autour du point critique à f(point) :
      - tout ≥ f(point)  → minimum
      - tout ≤ f(point)  → maximum
      - les deux         → ni minimum ni maximum (point col / dégénéré)
    """
    try:
        x0 = float(point[0])
        y0 = float(point[1])
        f0 = float(expr.subs({x: point[0], y: point[1]}))
    except (TypeError, ValueError):
        return "cas indéterminé (det = 0)"

    f_num = sp.lambdify((x, y), expr, "math")
    tol = 1e-12
    positif = negatif = False
    for rayon in (eps, eps * 10, eps * 100):
        for k in range(n):
            theta = 2 * math.pi * k / n
            xv = x0 + rayon * math.cos(theta)
            yv = y0 + rayon * math.sin(theta)
            try:
                ecart = float(f_num(xv, yv)) - f0
            except (TypeError, ValueError, ZeroDivisionError, ArithmeticError):
                continue
            if ecart > tol:
                positif = True
            elif ecart < -tol:
                negatif = True

    if positif and not negatif:
        return "minimum local (confirmé numériquement, det = 0)"
    if negatif and not positif:
        return "maximum local (confirmé numériquement, det = 0)"
    if positif and negatif:
        return "ni minimum ni maximum / point col (det = 0)"
    return "cas indéterminé (det = 0)"


def _classifier(det, fxx, expr, x, y, point):
    """Test de la hessienne ; repli numérique de voisinage si det = 0."""
    if det.is_positive:
        if fxx.is_positive:
            return "minimum local"
        if fxx.is_negative:
            return "maximum local"
    elif det.is_negative:
        return "point col (selle)"
    elif det == 0:
        return _test_numerique_voisinage(expr, x, y, point)

    # Repli numérique si le signe symbolique est inconnu.
    try:
        d = float(det)
        h = float(fxx)
    except (TypeError, ValueError):
        return "nature indéterminée"
    if abs(d) < 1e-12:
        return _test_numerique_voisinage(expr, x, y, point)
    if d > 0:
        return "minimum local" if h > 0 else "maximum local"
    return "point col (selle)"


def analyser(expression, variables=None):
    """Analyse complète de f(x, y).

    `expression` : chaîne ("x**2 + y**2") ou expression SymPy.
    Lève ValueError si l'expression est invalide.
    """
    if variables is None:
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
    else:
        x, y = variables

    expr = parsing.lire(expression, {"x": x, "y": y})

    symboles_libres = expr.free_symbols - {x, y}
    if symboles_libres:
        raise ValueError(
            f"L'expression contient des symboles inattendus : {symboles_libres}. "
            "Utilisez uniquement les variables x et y."
        )

    f_x = sp.simplify(sp.diff(expr, x))
    f_y = sp.simplify(sp.diff(expr, y))

    hessienne = sp.hessian(expr, (x, y))

    # Points critiques : résolution du système ∇f = 0
    try:
        solutions = sp.solve([f_x, f_y], [x, y], dict=True)
    except (NotImplementedError, Exception):  # noqa: BLE001
        solutions = []

    points_critiques = []
    for sol in solutions:
        x0 = sol.get(x, x)
        y0 = sol.get(y, y)
        # On ignore les solutions paramétriques ou complexes.
        if x0.free_symbols or y0.free_symbols:
            continue
        if not (_est_reel(x0) and _est_reel(y0)):
            continue

        h_au_point = hessienne.subs({x: x0, y: y0})
        det = sp.simplify(h_au_point.det())
        fxx = sp.simplify(h_au_point[0, 0])
        valeur = sp.simplify(expr.subs({x: x0, y: y0}))
        nature = _classifier(det, fxx, expr, x, y, (x0, y0))

        points_critiques.append(
            PointCritique2D((x0, y0), valeur, nature, det, fxx)
        )

    return ResultatDeuxVariables(
        expression=expr,
        variables=(x, y),
        gradient=(f_x, f_y),
        hessienne=hessienne,
        points_critiques=points_critiques,
    )


def rapport_texte(resultat):
    r = resultat
    x, y = r.variables
    fxy = sp.simplify(sp.diff(r.expression, x, y))
    fyy = sp.simplify(sp.diff(r.expression, y, 2))
    L = []
    L.append("═" * 64)
    L.append(f"  ÉTUDE DE LA FONCTION  f(x, y) = {r.expression}")
    L.append("═" * 64)

    L.append("\nÉTAPE 1 — Points critiques : on résout ∇f = 0")
    L.append(f"   ∂f/∂x = {r.gradient[0]} = 0")
    L.append(f"   ∂f/∂y = {r.gradient[1]} = 0")
    if r.points_critiques:
        L.append("   Solutions du système :")
        for i, pc in enumerate(r.points_critiques):
            L.append(f"      • P{_indice(i)} = ({pc.point[0]}, {pc.point[1]})")
    else:
        L.append("   (aucun point critique réel)")

    L.append("\nÉTAPE 2 — Dérivées secondes (notation r, s, t)")
    L.append(f"   r = ∂²f/∂x²  = {sp.simplify(sp.diff(r.expression, x, 2))}")
    L.append(f"   s = ∂²f/∂x∂y = {fxy}")
    L.append(f"   t = ∂²f/∂y²  = {fyy}")

    L.append("\nÉTAPE 3 — Nature : condition suffisante du 2nd ordre")
    L.append("   On évalue rt − s² (= det de la hessienne) en chaque point :")
    L.append("     • rt − s² > 0 et r > 0 ⟹ MINIMUM local")
    L.append("     • rt − s² > 0 et r < 0 ⟹ MAXIMUM local")
    L.append("     • rt − s² < 0          ⟹ POINT COL (selle)")
    L.append("     • rt − s² = 0          ⟹ on ne peut pas conclure (test poussé)")
    if r.points_critiques:
        for i, pc in enumerate(r.points_critiques):
            x0, y0 = pc.point
            r0 = pc.f_xx
            s0 = sp.simplify(fxy.subs({x: x0, y: y0}))
            t0 = sp.simplify(fyy.subs({x: x0, y: y0}))
            L.append(
                f"      • P{_indice(i)} = ({x0}, {y0}) : r={r0}, s={s0}, t={t0}, "
                f"rt−s² = {pc.determinant} ⟹ {pc.nature}"
            )
    else:
        L.append("      (rien à classer)")

    if r.points_critiques:
        L.append("\nCONCLUSION")
        for i, pc in enumerate(r.points_critiques):
            L.append(f"   P{_indice(i)} = {pc.point} : {pc.nature}  (f = {pc.valeur})")
    L.append("═" * 64)
    return "\n".join(L)
