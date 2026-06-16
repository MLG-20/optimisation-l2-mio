"""Optimisation sous contrainte par les multiplicateurs de Lagrange.

On cherche les extrema de f(x, y) sous la contrainte g(x, y) = 0.

Méthode :
  - lagrangien  L(x, y, λ) = f(x, y) - λ · g(x, y)
  - système     ∂L/∂x = 0, ∂L/∂y = 0, g(x, y) = 0
  - nature      test de la HESSIENNE BORDÉE en chaque point.

Pour 2 variables et 1 contrainte, le déterminant de la hessienne bordée donne :
  det > 0  →  maximum local sous contrainte
  det < 0  →  minimum local sous contrainte
"""

from dataclasses import dataclass, field

import sympy as sp

from . import une_variable


@dataclass
class PointLagrange:
    point: tuple        # (x0, y0)
    multiplicateur: sp.Expr
    valeur: sp.Expr     # f(x0, y0)
    nature: str
    det_bordee: sp.Expr


@dataclass
class ResultatLagrange:
    objectif: sp.Expr
    contrainte: sp.Expr            # g, la contrainte est g = 0
    variables: tuple              # (x, y)
    multiplicateur: sp.Symbol     # λ
    lagrangien: sp.Expr
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


def _sympify(expression, locals_):
    if isinstance(expression, str):
        try:
            return sp.sympify(expression, locals=locals_)
        except (sp.SympifyError, SyntaxError, TypeError) as err:
            raise ValueError(f"Expression invalide : {expression!r}") from err
    return expression


def _hessienne_bordee(L, contrainte, x, y, lam):
    """Construit la hessienne bordée 3×3 (cas 2 variables, 1 contrainte)."""
    gx = sp.diff(contrainte, x)
    gy = sp.diff(contrainte, y)
    Lxx = sp.diff(L, x, 2)
    Lyy = sp.diff(L, y, 2)
    Lxy = sp.diff(L, x, y)
    return sp.Matrix([
        [0,  gx,  gy],
        [gx, Lxx, Lxy],
        [gy, Lxy, Lyy],
    ])


def _classifier(det):
    if det.is_positive:
        return "maximum local sous contrainte"
    if det.is_negative:
        return "minimum local sous contrainte"
    if det == 0:
        return "cas indéterminé (det bordée = 0)"
    try:
        d = float(det)
    except (TypeError, ValueError):
        return "nature indéterminée"
    if abs(d) < 1e-12:
        return "cas indéterminé (det bordée ≈ 0)"
    return "maximum local sous contrainte" if d > 0 else "minimum local sous contrainte"


def analyser(objectif, contrainte, variables=None):
    """Optimise `objectif` f(x, y) sous la contrainte `contrainte` = 0.

    `objectif` et `contrainte` : chaînes ou expressions SymPy.
    Exemple : analyser("x*y", "x + y - 10")  → max de x·y sous x+y=10.
    Lève ValueError si une expression est invalide.
    """
    if variables is None:
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
    else:
        x, y = variables
    lam = sp.Symbol("lambda", real=True)

    locals_ = {"x": x, "y": y}
    f = _sympify(objectif, locals_)
    g = _sympify(contrainte, locals_)

    symboles = (f.free_symbols | g.free_symbols) - {x, y}
    if symboles:
        raise ValueError(
            f"Symboles inattendus : {symboles}. Utilisez uniquement x et y."
        )

    L = f - lam * g

    equations = [sp.diff(L, x), sp.diff(L, y), g]
    try:
        solutions = sp.solve(equations, [x, y, lam], dict=True)
    except (NotImplementedError, Exception):  # noqa: BLE001
        solutions = []

    H = _hessienne_bordee(L, g, x, y, lam)

    points = []
    for sol in solutions:
        x0 = sol.get(x, x)
        y0 = sol.get(y, y)
        lam0 = sol.get(lam, lam)
        if x0.free_symbols or y0.free_symbols:
            continue
        if not (_est_reel(x0) and _est_reel(y0)):
            continue

        subs = {x: x0, y: y0, lam: lam0}
        det = sp.simplify(H.subs(subs).det())
        valeur = sp.simplify(f.subs({x: x0, y: y0}))
        nature = _classifier(det)
        points.append(PointLagrange((x0, y0), sp.simplify(lam0), valeur, nature, det))

    return ResultatLagrange(
        objectif=f,
        contrainte=g,
        variables=(x, y),
        multiplicateur=lam,
        lagrangien=L,
        points_critiques=points,
    )


def rapport_texte(resultat):
    r = resultat
    x, y = r.variables
    lam = r.multiplicateur
    L = []
    L.append("═" * 64)
    L.append("  OPTIMISATION SOUS CONTRAINTE — Méthode de Lagrange")
    L.append("═" * 64)
    L.append(f"  Objectif   : f(x, y) = {r.objectif}")
    L.append(f"  Contrainte : g(x, y) = {r.contrainte} = 0")

    L.append("\nÉTAPE 1 — Lagrangien L = f − λ·g")
    L.append(f"   L(x, y, λ) = {r.lagrangien}")

    L.append("\nÉTAPE 2 — Conditions du 1er ordre : ∇L = 0")
    L.append(f"   ∂L/∂x = {sp.diff(r.lagrangien, x)} = 0")
    L.append(f"   ∂L/∂y = {sp.diff(r.lagrangien, y)} = 0")
    L.append(f"   ∂L/∂λ = {-r.contrainte} = 0   (c'est la contrainte g = 0)")

    L.append("\nÉTAPE 3 — Résolution du système ⟹ points candidats")
    if r.points_critiques:
        for i, pc in enumerate(r.points_critiques):
            L.append(f"      • P{_indice(i)} = ({pc.point[0]}, {pc.point[1]})"
                     f"   avec λ = {pc.multiplicateur}")
    else:
        L.append("      (aucun point trouvé)")

    L.append("\nÉTAPE 4 — Nature : test de la HESSIENNE BORDÉE")
    L.append("   det(H̄) > 0 ⟹ MAXIMUM sous contrainte ; det(H̄) < 0 ⟹ MINIMUM.")
    if r.points_critiques:
        for i, pc in enumerate(r.points_critiques):
            L.append(
                f"      • P{_indice(i)} = ({pc.point[0]}, {pc.point[1]}) : "
                f"det(H̄) = {pc.det_bordee} ⟹ {pc.nature}   (f = {pc.valeur})"
            )

    numeriques = []
    for pc in r.points_critiques:
        try:
            numeriques.append((float(pc.valeur), pc))
        except (TypeError, ValueError):
            pass
    if numeriques:
        vmax = max(numeriques, key=lambda t: t[0])[1]
        vmin = min(numeriques, key=lambda t: t[0])[1]
        L.append("\nCONCLUSION (comparaison des valeurs)")
        L.append(f"   MAX : f = {vmax.valeur} en {vmax.point}")
        L.append(f"   MIN : f = {vmin.valeur} en {vmin.point}")
    L.append("═" * 64)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Méthode par SUBSTITUTION
# --------------------------------------------------------------------------- #
@dataclass
class PointSubstitution:
    point: tuple        # (x0, y0)
    valeur: sp.Expr     # f(x0, y0)
    nature: str


@dataclass
class ResultatSubstitution:
    objectif: sp.Expr
    contrainte: sp.Expr
    variables: tuple                 # (x, y)
    variable_isolee: sp.Symbol       # variable tirée de la contrainte
    expression_isolee: sp.Expr       # ex. y = x - 1
    variable_libre: sp.Symbol
    fonction_reduite: sp.Expr        # F(variable_libre)
    points_critiques: list = field(default_factory=list)


def resoudre_par_substitution(objectif, contrainte, variables=None):
    """Optimise f(x, y) sous g(x, y) = 0 par SUBSTITUTION.

    On isole une variable depuis la contrainte, on l'injecte dans f pour obtenir
    une fonction d'une seule variable F, puis on optimise F (F' = 0, test de F'').
    Donne les mêmes points critiques que la méthode de Lagrange.
    """
    if variables is None:
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
    else:
        x, y = variables

    locals_ = {"x": x, "y": y}
    f = _sympify(objectif, locals_)
    g = _sympify(contrainte, locals_)

    symboles = (f.free_symbols | g.free_symbols) - {x, y}
    if symboles:
        raise ValueError(
            f"Symboles inattendus : {symboles}. Utilisez uniquement x et y."
        )

    # On tente d'isoler y, sinon x.
    isolee = libre = expr_isolee = None
    sols_y = sp.solve(g, y)
    if sols_y:
        isolee, libre, expr_isolee = y, x, sols_y[0]
    else:
        sols_x = sp.solve(g, x)
        if not sols_x:
            raise ValueError(
                "Impossible d'isoler une variable depuis la contrainte. "
                "Utilisez plutôt la méthode de Lagrange."
            )
        isolee, libre, expr_isolee = x, y, sols_x[0]

    F = sp.simplify(f.subs(isolee, expr_isolee))
    derivee = sp.diff(F, libre)

    points = []
    for t in une_variable.solutions_reelles(derivee, libre):
        autre = sp.simplify(expr_isolee.subs(libre, t))
        nature = une_variable._nature_point_critique(F, libre, t)
        nature = nature.replace("local", "local sous contrainte")
        valeur = sp.simplify(f.subs({libre: t, isolee: autre}))
        point = (t, autre) if libre == x else (autre, t)
        points.append(PointSubstitution(point, valeur, nature))

    return ResultatSubstitution(
        objectif=f, contrainte=g, variables=(x, y),
        variable_isolee=isolee, expression_isolee=expr_isolee,
        variable_libre=libre, fonction_reduite=F, points_critiques=points,
    )


def rapport_substitution(resultat):
    r = resultat
    t = r.variable_libre
    F = r.fonction_reduite
    dF = sp.simplify(sp.diff(F, t))
    d2F = sp.simplify(sp.diff(F, t, 2))
    L = []
    L.append("═" * 64)
    L.append("  OPTIMISATION SOUS CONTRAINTE — Méthode par substitution")
    L.append("═" * 64)
    L.append(f"  Objectif   : f(x, y) = {r.objectif}")
    L.append(f"  Contrainte : {r.contrainte} = 0")

    L.append("\nÉTAPE 1 — On isole une variable dans la contrainte")
    L.append(f"   {r.variable_isolee} = {r.expression_isolee}")

    L.append("\nÉTAPE 2 — On l'injecte dans f ⟹ fonction à UNE variable")
    L.append(f"   F({t}) = {F}")

    L.append("\nÉTAPE 3 — On optimise F : F'({t}) = 0")
    L.append(f"   F'({t}) = {dF}")
    if r.points_critiques:
        for i, pc in enumerate(r.points_critiques):
            L.append(f"      • P{_indice(i)} = ({pc.point[0]}, {pc.point[1]})")
    else:
        L.append("      (aucune solution réelle)")

    L.append("\nÉTAPE 4 — Nature : signe de F''")
    L.append(f"   F''({t}) = {d2F}")
    L.append("   F'' > 0 ⟹ minimum ; F'' < 0 ⟹ maximum.")
    if r.points_critiques:
        # On retrouve la valeur de la variable libre pour évaluer F''.
        for i, pc in enumerate(r.points_critiques):
            tv = pc.point[0] if t == r.variables[0] else pc.point[1]
            f2 = sp.simplify(d2F.subs(t, tv))
            L.append(
                f"      • P{_indice(i)} = ({pc.point[0]}, {pc.point[1]}) : "
                f"F''({tv}) = {f2} ⟹ {pc.nature}   (f = {pc.valeur})"
            )
    L.append("═" * 64)
    return "\n".join(L)
