"""Analyse marginale (économie) — Feuille 1 du cours d'optimisation.

Outils :
  - fonction marginale (coût / recette / profit marginal) = dérivée ;
  - coût unitaire moyen C(x)/x et sa minimisation ;
  - profit P = R - C, profit marginal, quantité optimale (R' = C').

Interprétation : la valeur marginale en x₀, m(x₀) = f'(x₀), approxime la
variation de f pour une unité supplémentaire, c.-à-d. ≈ f(x₀+1) - f(x₀)
(coût / recette / profit de la (x₀+1)-ème unité).
"""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import sympy as sp


@dataclass
class ResultatMarginal:
    fonction: sp.Expr
    variable: sp.Symbol
    marginale: sp.Expr               # f'(x)
    point: sp.Expr = None            # x₀ éventuel
    valeur_marginale: sp.Expr = None  # f'(x₀)
    variation_exacte: sp.Expr = None  # f(x₀+1) - f(x₀)


@dataclass
class ResultatCoutMoyen:
    cout: sp.Expr
    variable: sp.Symbol
    cout_moyen: sp.Expr              # C(x)/x
    derivee: sp.Expr
    quantite_optimale: sp.Expr = None  # x>0 minimisant C(x)/x
    cout_moyen_min: sp.Expr = None
    cout_marginal_au_min: sp.Expr = None  # C'(x*) — doit égaler le coût moyen min


@dataclass
class ResultatProfit:
    revenu: sp.Expr
    cout: sp.Expr
    variable: sp.Symbol
    profit: sp.Expr                  # R - C
    profit_marginal: sp.Expr         # P'
    quantites_optimales: list = field(default_factory=list)  # [(x*, P(x*))]


def _sympify(expression, x):
    if isinstance(expression, str):
        try:
            return sp.sympify(expression, locals={"x": x})
        except (sp.SympifyError, SyntaxError, TypeError) as err:
            raise ValueError(f"Expression invalide : {expression!r}") from err
    return expression


def _positifs_reels(equation, x):
    try:
        solutions = sp.solve(equation, x)
    except (NotImplementedError, Exception):  # noqa: BLE001
        return []
    bons = []
    for s in solutions:
        if s.is_real is False:
            continue
        try:
            if float(s) > 0:
                bons.append(s)
        except (TypeError, ValueError):
            if s.is_positive:
                bons.append(s)
    return bons


def fonction_marginale(fonction, point=None, variable=None):
    """Renvoie la fonction marginale f'(x) (coût/recette/profit marginal).

    Si `point` est fourni, calcule aussi f'(x₀) et la variation exacte
    f(x₀+1) - f(x₀) (utile pour « coût de la (x₀+1)-ème unité »).
    """
    x = variable or sp.Symbol("x", real=True)
    f = _sympify(fonction, x)
    marginale = sp.simplify(sp.diff(f, x))

    valeur = variation = None
    if point is not None:
        point = sp.sympify(point)
        valeur = sp.simplify(marginale.subs(x, point))
        variation = sp.simplify(f.subs(x, point + 1) - f.subs(x, point))

    return ResultatMarginal(
        fonction=f, variable=x, marginale=marginale,
        point=point, valeur_marginale=valeur, variation_exacte=variation,
    )


def cout_moyen(cout, variable=None):
    """Coût unitaire moyen C(x)/x, sa dérivée et le x>0 qui le minimise.

    Au minimum du coût moyen, on a la propriété : coût moyen = coût marginal.
    """
    x = variable or sp.Symbol("x", real=True, positive=True)
    C = _sympify(cout, x)
    CM = sp.simplify(C / x)
    derivee = sp.simplify(sp.diff(CM, x))

    quantite = cm_min = marginal_min = None
    candidats = _positifs_reels(sp.numer(sp.together(derivee)), x)
    if candidats:
        quantite = min(candidats, key=lambda s: float(CM.subs(x, s)))
        cm_min = sp.simplify(CM.subs(x, quantite))
        marginal_min = sp.simplify(sp.diff(C, x).subs(x, quantite))

    return ResultatCoutMoyen(
        cout=C, variable=x, cout_moyen=CM, derivee=derivee,
        quantite_optimale=quantite, cout_moyen_min=cm_min,
        cout_marginal_au_min=marginal_min,
    )


def profit(revenu, cout, variable=None):
    """Profit P = R - C, profit marginal P', et quantités optimales (P' = 0)."""
    x = variable or sp.Symbol("x", real=True, positive=True)
    R = _sympify(revenu, x)
    C = _sympify(cout, x)
    P = sp.simplify(R - C)
    Pm = sp.simplify(sp.diff(P, x))

    quantites = []
    for s in _positifs_reels(sp.numer(sp.together(Pm)), x):
        # On ne garde que les maxima (P'' < 0).
        seconde = sp.diff(P, x, 2).subs(x, s)
        try:
            est_max = float(seconde) < 0
        except (TypeError, ValueError):
            est_max = seconde.is_negative is True
        if est_max:
            quantites.append((s, sp.simplify(P.subs(x, s))))

    return ResultatProfit(
        revenu=R, cout=C, variable=x, profit=P,
        profit_marginal=Pm, quantites_optimales=quantites,
    )


def rapport_marginal(resultat):
    r = resultat
    L = []
    L.append("═" * 64)
    L.append("  ANALYSE MARGINALE — fonction marginale")
    L.append("═" * 64)
    L.append("\nRAPPEL — La fonction marginale est la DÉRIVÉE.")
    L.append("   Elle approxime la variation pour 1 unité de plus :")
    L.append("   f'(x0) ≈ f(x0+1) − f(x0)  (coût/recette/profit de l'unité suivante).")
    L.append(f"\nÉTAPE 1 — Fonction : f(x) = {r.fonction}")
    L.append(f"ÉTAPE 2 — Marginale : f'(x) = {r.marginale}")
    if r.point is not None:
        L.append(f"\nÉTAPE 3 — Évaluation en x = {r.point} :")
        L.append(f"   f'({r.point}) = {r.valeur_marginale}   (valeur marginale)")
        L.append(f"   Vérification — variation exacte f({r.point}+1) − f({r.point}) "
                 f"= {r.variation_exacte}")
        L.append("   (les deux sont proches : le marginal approxime bien l'unité suivante)")
    L.append("═" * 64)
    return "\n".join(L)


def rapport_cout_moyen(resultat):
    r = resultat
    L = []
    L.append("═" * 64)
    L.append("  ANALYSE MARGINALE — coût unitaire moyen")
    L.append("═" * 64)
    L.append(f"\nÉTAPE 1 — Coût total : C(x) = {r.cout}")
    L.append("ÉTAPE 2 — Coût moyen = coût total / quantité :")
    L.append(f"   CM(x) = C(x)/x = {r.cout_moyen}")
    L.append("ÉTAPE 3 — Minimisation : on résout (CM)'(x) = 0")
    L.append(f"   (CM)'(x) = {r.derivee}")
    if r.quantite_optimale is not None:
        L.append(f"\nRÉSULTAT")
        L.append(f"   Coût moyen MINIMAL en x = {r.quantite_optimale}")
        L.append(f"   valeur minimale du coût moyen = {r.cout_moyen_min}")
        L.append(f"   coût marginal C'(x) en ce point = {r.cout_marginal_au_min}")
        L.append("   PROPRIÉTÉ : au minimum, coût moyen = coût marginal ✓")
    else:
        L.append("\n   (pas de minimum sur x > 0)")
    L.append("═" * 64)
    return "\n".join(L)


def rapport_profit(resultat):
    r = resultat
    L = []
    L.append("═" * 64)
    L.append("  ANALYSE MARGINALE — profit maximal")
    L.append("═" * 64)
    L.append(f"\nÉTAPE 1 — Recette : R(x) = {r.revenu}")
    L.append(f"ÉTAPE 2 — Coût    : C(x) = {r.cout}")
    L.append(f"ÉTAPE 3 — Profit  : P(x) = R(x) − C(x) = {r.profit}")
    L.append(f"ÉTAPE 4 — Profit marginal : P'(x) = {r.profit_marginal}")
    L.append("   On résout P'(x) = 0 (⟺ recette marginale = coût marginal),")
    L.append("   en gardant les maxima (P''(x) < 0) :")
    if r.quantites_optimales:
        for q, val in r.quantites_optimales:
            L.append(f"      • PROFIT MAXIMAL en x = {q}   →   P = {val}")
    else:
        L.append("      (aucun maximum de profit trouvé)")
    L.append("═" * 64)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Graphiques
# --------------------------------------------------------------------------- #
def _courbe(fig, expr, x, xs, nom, couleur):
    """Ajoute la courbe de `expr` (évaluée sur xs) à la figure."""
    f_num = sp.lambdify(x, expr, "numpy")
    ys = np.full_like(xs, np.nan, dtype=float)
    with np.errstate(all="ignore"):
        try:
            ys[:] = f_num(xs)
        except Exception:  # noqa: BLE001
            for i, xv in enumerate(xs):
                try:
                    ys[i] = float(expr.subs(x, xv))
                except Exception:  # noqa: BLE001
                    ys[i] = np.nan
    ys[np.abs(ys) > 1e7] = np.nan
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=nom,
                             line=dict(color=couleur, width=2)))


def _bornes(point, defaut=20.0):
    """Choisit un intervalle de tracé [lo, hi] autour d'un point repère."""
    try:
        p = float(point)
        if p > 0:
            return 0.0, 2.0 * p
    except (TypeError, ValueError):
        pass
    return 0.0, defaut


def figure_marginale(resultat, nb_points=400):
    """Trace f(x) et sa fonction marginale f'(x) ; marque le point x0."""
    r = resultat
    x = r.variable
    lo, hi = _bornes(r.point)
    if hi <= lo:
        lo, hi = 0.0, 20.0
    xs = np.linspace(lo, hi, nb_points)

    fig = go.Figure()
    _courbe(fig, r.fonction, x, xs, "f(x)", "royalblue")
    _courbe(fig, r.marginale, x, xs, "f'(x) (marginale)", "darkorange")

    if r.point is not None and r.valeur_marginale is not None:
        xv = float(r.point)
        yv = float(r.valeur_marginale)
        fig.add_trace(go.Scatter(
            x=[xv], y=[yv], mode="markers+text",
            marker=dict(size=12, color="green", symbol="star"),
            text=[f"f'({xv:g}) = {yv:g}"], textposition="top center",
            name="valeur marginale",
        ))
        fig.add_vline(x=xv, line_dash="dot", line_color="gray")

    fig.update_layout(title=f"Fonction et sa marginale — f(x) = {r.fonction}",
                      xaxis_title="x", yaxis_title="valeur",
                      template="plotly_white", hovermode="x unified")
    return fig


def figure_cout_moyen(resultat, nb_points=400):
    """Trace le coût moyen C(x)/x et le coût marginal C'(x) ; marque le minimum."""
    r = resultat
    x = r.variable
    Cm = sp.diff(r.cout, x)
    lo, hi = _bornes(r.quantite_optimale, defaut=50.0)
    lo = max(lo, hi / 100)   # éviter x = 0 (division)
    xs = np.linspace(lo, hi, nb_points)

    fig = go.Figure()
    _courbe(fig, r.cout_moyen, x, xs, "coût moyen C(x)/x", "royalblue")
    _courbe(fig, Cm, x, xs, "coût marginal C'(x)", "darkorange")

    if r.quantite_optimale is not None:
        xv = float(r.quantite_optimale)
        yv = float(r.cout_moyen_min)
        fig.add_trace(go.Scatter(
            x=[xv], y=[yv], mode="markers+text",
            marker=dict(size=13, color="green", symbol="star"),
            text=[f"min ({xv:g}; {yv:g})"], textposition="top center",
            name="coût moyen minimal",
        ))
        fig.add_vline(x=xv, line_dash="dot", line_color="gray",
                      annotation_text=f"x* = {xv:g}")

    fig.update_layout(title=f"Coût moyen & coût marginal — C(x) = {r.cout}",
                      xaxis_title="x (quantité)", yaxis_title="coût",
                      template="plotly_white", hovermode="x unified")
    return fig


def figure_profit(resultat, nb_points=400):
    """Trace R(x), C(x) et le profit P(x) = R − C ; marque le profit maximal."""
    r = resultat
    x = r.variable
    point = r.quantites_optimales[0][0] if r.quantites_optimales else None
    lo, hi = _bornes(point, defaut=50.0)
    if hi <= lo:
        lo, hi = 0.0, 50.0
    xs = np.linspace(lo, hi, nb_points)

    fig = go.Figure()
    _courbe(fig, r.revenu, x, xs, "recette R(x)", "seagreen")
    _courbe(fig, r.cout, x, xs, "coût C(x)", "crimson")
    _courbe(fig, r.profit, x, xs, "profit P(x) = R − C", "royalblue")

    for q, val in r.quantites_optimales:
        try:
            xv, yv = float(q), float(val)
        except (TypeError, ValueError):
            continue
        fig.add_trace(go.Scatter(
            x=[xv], y=[yv], mode="markers+text",
            marker=dict(size=13, color="darkorange", symbol="star"),
            text=[f"profit max ({xv:g}; {yv:g})"], textposition="top center",
            name="profit maximal",
        ))
        fig.add_vline(x=xv, line_dash="dot", line_color="gray")

    fig.update_layout(title="Recette, coût et profit",
                      xaxis_title="x (quantité)", yaxis_title="euros",
                      template="plotly_white", hovermode="x unified")
    return fig
