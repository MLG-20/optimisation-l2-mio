"""Analyse complète d'une fonction d'une variable réelle f(x).

Calcule : domaine de définition, dérivées, points critiques et leur nature,
points d'inflexion, limites et asymptotes (horizontales, verticales, obliques).
"""

from dataclasses import dataclass, field

import sympy as sp
from sympy.calculus.util import continuous_domain

from . import parsing


# --------------------------------------------------------------------------- #
# Structures de données
# --------------------------------------------------------------------------- #
@dataclass
class PointCritique:
    abscisse: sp.Expr        # x du point critique
    ordonnee: sp.Expr        # f(x) correspondant
    nature: str              # "minimum local", "maximum local", ...


@dataclass
class ResultatUneVariable:
    expression: sp.Expr
    variable: sp.Symbol
    domaine: object
    derivee_premiere: sp.Expr
    derivee_seconde: sp.Expr
    points_critiques: list = field(default_factory=list)
    points_inflexion: list = field(default_factory=list)
    asymptotes: dict = field(default_factory=dict)
    limites: dict = field(default_factory=dict)


@dataclass
class Candidat:
    abscisse: sp.Expr
    valeur: sp.Expr     # f(abscisse)
    type: str           # "point critique intérieur" | "borne"


@dataclass
class ResultatIntervalle:
    expression: sp.Expr
    variable: sp.Symbol
    a: sp.Expr
    b: sp.Expr
    candidats: list = field(default_factory=list)
    minimum: Candidat = None    # minimum absolu sur [a, b]
    maximum: Candidat = None    # maximum absolu sur [a, b]


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def _est_reel(val):
    """Vrai si la valeur symbolique est (probablement) un réel."""
    if val.is_real is True:
        return True
    if val.is_real is False:
        return False
    # Indéterminé symboliquement : on tente une évaluation numérique.
    try:
        c = complex(val)
        return abs(c.imag) < 1e-9
    except (TypeError, ValueError):
        return False


def solutions_reelles(equation, variable):
    """Résout equation = 0 et ne garde que les solutions réelles, triées."""
    try:
        solutions = sp.solve(equation, variable)
    except (NotImplementedError, Exception):  # noqa: BLE001
        return []
    reelles = [s for s in solutions if _est_reel(s)]

    def cle_tri(s):
        try:
            return (0, float(s))
        except (TypeError, ValueError):
            return (1, 0.0)

    return sorted(reelles, key=cle_tri)


def _nature_point_critique(f, x, point):
    """Nature d'un point critique via le test des dérivées successives."""
    k = 2
    while k <= 8:
        valeur = sp.simplify(sp.diff(f, x, k).subs(x, point))
        if valeur.is_positive:
            signe = 1
        elif valeur.is_negative:
            signe = -1
        elif valeur == 0:
            k += 1
            continue
        else:
            try:
                fv = float(valeur)
            except (TypeError, ValueError):
                return "nature indéterminée"
            if abs(fv) < 1e-12:
                k += 1
                continue
            signe = 1 if fv > 0 else -1

        if k % 2 == 0:
            return "minimum local" if signe > 0 else "maximum local"
        return "point d'inflexion à tangente horizontale"
    return "nature indéterminée"


def _asymptotes(expr, x):
    res = {"horizontales": [], "verticales": [], "obliques": []}

    for direction in (sp.oo, -sp.oo):
        nom = "+∞" if direction == sp.oo else "-∞"
        try:
            limite = sp.limit(expr, x, direction)
        except Exception:  # noqa: BLE001
            continue
        if limite.is_finite:
            if limite not in [h[0] for h in res["horizontales"]]:
                res["horizontales"].append((limite, nom))
        else:
            try:
                a = sp.limit(expr / x, x, direction)
                if a.is_finite and a != 0:
                    b = sp.limit(expr - a * x, x, direction)
                    if b.is_finite:
                        res["obliques"].append((a, b, nom))
            except Exception:  # noqa: BLE001
                pass

    try:
        singularites = sp.singularities(expr, x, sp.S.Reals)
        if isinstance(singularites, sp.FiniteSet):
            for s in singularites:
                res["verticales"].append(s)
    except Exception:  # noqa: BLE001
        pass

    return res


# --------------------------------------------------------------------------- #
# Analyse principale
# --------------------------------------------------------------------------- #
def analyser(expression, variable=None):
    """Analyse complète de la fonction donnée.

    `expression` peut être une chaîne ("x**3 - 3*x") ou une expression SymPy.
    Lève ValueError si l'expression est invalide.
    """
    x = variable or sp.Symbol("x", real=True)

    expr = parsing.lire(expression, {"x": x})

    symboles_libres = expr.free_symbols - {x}
    if symboles_libres:
        raise ValueError(
            f"L'expression contient des symboles inattendus : {symboles_libres}. "
            "Utilisez uniquement la variable x."
        )

    try:
        domaine = continuous_domain(expr, x, sp.S.Reals)
    except Exception:  # noqa: BLE001
        domaine = sp.S.Reals

    derivee_1 = sp.simplify(sp.diff(expr, x))
    derivee_2 = sp.simplify(sp.diff(derivee_1, x))

    # Points critiques
    points_critiques = []
    for point in solutions_reelles(derivee_1, x):
        ordonnee = sp.simplify(expr.subs(x, point))
        nature = _nature_point_critique(expr, x, point)
        points_critiques.append(PointCritique(point, ordonnee, nature))

    # Points d'inflexion (f'' = 0 avec changement de signe)
    points_inflexion = []
    for point in solutions_reelles(derivee_2, x):
        try:
            avant = sp.diff(expr, x, 3).subs(x, point)
            change = avant != 0
        except Exception:  # noqa: BLE001
            change = True
        if change:
            ordonnee = sp.simplify(expr.subs(x, point))
            points_inflexion.append((point, ordonnee))

    limites = {}
    for direction, nom in ((sp.oo, "+∞"), (-sp.oo, "-∞")):
        try:
            limites[nom] = sp.limit(expr, x, direction)
        except Exception:  # noqa: BLE001
            limites[nom] = sp.nan

    return ResultatUneVariable(
        expression=expr,
        variable=x,
        domaine=domaine,
        derivee_premiere=derivee_1,
        derivee_seconde=derivee_2,
        points_critiques=points_critiques,
        points_inflexion=points_inflexion,
        asymptotes=_asymptotes(expr, x),
        limites=limites,
    )


# --------------------------------------------------------------------------- #
# Rapport texte
# --------------------------------------------------------------------------- #
def rapport_texte(resultat):
    """Rapport détaillé et pédagogique de l'étude de fonction (avec démarche)."""
    r = resultat
    x = r.variable
    L = []
    L.append("═" * 64)
    L.append(f"  ÉTUDE DE LA FONCTION  f(x) = {r.expression}")
    L.append("═" * 64)

    L.append("\nÉTAPE 1 — Domaine de définition")
    L.append(f"   D_f = {r.domaine}")
    L.append("   (on exclut les valeurs annulant un dénominateur ou hors d'un log/racine)")

    L.append("\nÉTAPE 2 — Dérivée première et points critiques")
    L.append(f"   f'(x) = {r.derivee_premiere}")
    L.append("   On résout f'(x) = 0 (la tangente est horizontale en ces points) :")
    if r.points_critiques:
        for pc in r.points_critiques:
            L.append(f"      • x = {pc.abscisse}   →   f(x) = {pc.ordonnee}")
    else:
        L.append("      (aucune solution réelle : pas de point critique)")

    L.append("\nÉTAPE 3 — Nature des points critiques (test de la dérivée seconde)")
    L.append(f"   f''(x) = {r.derivee_seconde}")
    L.append("   Règle : f''(x0) > 0 ⟹ minimum,  f''(x0) < 0 ⟹ maximum.")
    if r.points_critiques:
        for pc in r.points_critiques:
            f2 = sp.simplify(r.derivee_seconde.subs(x, pc.abscisse))
            signe = ">0" if f2.is_positive else ("<0" if f2.is_negative else "=0")
            L.append(
                f"      • en x = {pc.abscisse} : f''({pc.abscisse}) = {f2} ({signe}) "
                f"⟹ {pc.nature}"
            )
        L.append("   (si f'' = 0, on regarde les dérivées d'ordre supérieur)")
    else:
        L.append("      (rien à classer)")

    L.append("\nÉTAPE 4 — Points d'inflexion (changement de concavité, f''(x) = 0)")
    if r.points_inflexion:
        for point, ordonnee in r.points_inflexion:
            L.append(f"      • x = {point}   →   f(x) = {ordonnee}")
    else:
        L.append("      (aucun)")

    L.append("\nÉTAPE 5 — Limites aux infinis et asymptotes")
    for nom, val in r.limites.items():
        L.append(f"   lim f(x) quand x → {nom} = {val}")
    a = r.asymptotes
    if a.get("horizontales"):
        for val, sens in a["horizontales"]:
            L.append(f"   ⟹ asymptote HORIZONTALE : y = {val}  (en {sens})")
    if a.get("verticales"):
        for val in a["verticales"]:
            L.append(f"   ⟹ asymptote VERTICALE : x = {val}  (f y diverge)")
    if a.get("obliques"):
        for pente, ordo, sens in a["obliques"]:
            L.append(f"   ⟹ asymptote OBLIQUE : y = {pente}·x + {ordo}  (en {sens})")
    if not any(a.values()):
        L.append("   ⟹ aucune asymptote")

    # Conclusion : extrema comparés.
    if r.points_critiques:
        L.append("\nCONCLUSION")
        for pc in r.points_critiques:
            L.append(f"   {pc.nature} en x = {pc.abscisse}, valeur f = {pc.ordonnee}")
    L.append("═" * 64)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Optimisation sur un intervalle fermé [a, b]
# --------------------------------------------------------------------------- #
def optimiser_sur_intervalle(expression, a, b, variable=None):
    """Cherche le minimum et le maximum ABSOLUS de f sur l'intervalle [a, b].

    Méthode (théorème des bornes atteintes sur un fermé borné) :
      - candidats = points critiques INTÉRIEURS à [a, b]  +  les deux bornes ;
      - on évalue f sur tous les candidats et on compare.

    `expression` : chaîne ou expression SymPy. `a`, `b` : bornes (nombres/exprs).
    """
    x = variable or sp.Symbol("x", real=True)

    expr = parsing.lire(expression, {"x": x})

    a, b = sp.sympify(a), sp.sympify(b)
    if a > b:
        a, b = b, a

    derivee = sp.diff(expr, x)

    candidats = []
    # Points critiques strictement intérieurs à l'intervalle.
    for point in solutions_reelles(derivee, x):
        try:
            dans_intervalle = bool(a < point) and bool(point < b)
        except TypeError:
            dans_intervalle = False
        if dans_intervalle:
            valeur = sp.simplify(expr.subs(x, point))
            candidats.append(Candidat(point, valeur, "point critique intérieur"))

    # Les deux bornes.
    for borne in (a, b):
        valeur = sp.simplify(expr.subs(x, borne))
        candidats.append(Candidat(borne, valeur, "borne"))

    def cle(candidat):
        return float(candidat.valeur)

    try:
        minimum = min(candidats, key=cle)
        maximum = max(candidats, key=cle)
    except (TypeError, ValueError):
        minimum = maximum = None

    return ResultatIntervalle(
        expression=expr, variable=x, a=a, b=b,
        candidats=candidats, minimum=minimum, maximum=maximum,
    )


def rapport_intervalle(resultat):
    r = resultat
    x = r.variable
    L = []
    L.append("═" * 64)
    L.append(f"  OPTIMISATION DE f(x) = {r.expression}  SUR [{r.a}, {r.b}]")
    L.append("═" * 64)
    L.append("\nMÉTHODE — Théorème des bornes atteintes :")
    L.append("   f continue sur un intervalle fermé borné [a, b] atteint son")
    L.append("   minimum et son maximum. Les candidats sont :")
    L.append("     (1) les points critiques INTÉRIEURS (où f'(x) = 0)")
    L.append("     (2) les deux BORNES a et b.")
    L.append("   On évalue f sur tous les candidats, puis on compare.")

    L.append(f"\nÉTAPE 1 — Dérivée : f'(x) = {sp.diff(r.expression, x)}")
    interieurs = [c for c in r.candidats if c.type != "borne"]
    L.append("ÉTAPE 2 — Points critiques intérieurs à [a, b] :")
    if interieurs:
        for c in interieurs:
            L.append(f"      • x = {c.abscisse}   →   f(x) = {c.valeur}")
    else:
        L.append("      (aucun point critique dans l'intervalle)")

    L.append("\nÉTAPE 3 — Évaluation de f sur TOUS les candidats :")
    for c in r.candidats:
        L.append(f"      f({c.abscisse}) = {c.valeur}    [{c.type}]")

    if r.minimum and r.maximum:
        L.append("\nÉTAPE 4 — Comparaison ⟹ RÉSULTAT :")
        L.append(f"   MINIMUM absolu : f({r.minimum.abscisse}) = {r.minimum.valeur}"
                 f"   (atteint sur : {r.minimum.type})")
        L.append(f"   MAXIMUM absolu : f({r.maximum.abscisse}) = {r.maximum.valeur}"
                 f"   (atteint sur : {r.maximum.type})")
    else:
        L.append("\n• Comparaison impossible (valeurs non numériques).")
    L.append("═" * 64)
    return "\n".join(L)
