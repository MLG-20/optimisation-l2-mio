"""Étude complète d'une fonction de COÛT sur un intervalle (Feuille 1, Exo 1).

À partir d'un coût total C(x) défini sur [a, b], on produit :
  - le coût marginal      C'(x)
  - le coût unitaire moyen f(x) = C(x)/x  et sa dérivée f'(x)
  - la quantité qui MINIMISE le coût moyen, et ce coût minimal
  - le tableau de valeurs et le tableau de variations
  - la vérification de la propriété  f(x*) = C'(x*)  au minimum.

Tout est exact (SymPy) ; le graphique superpose f(x) et C'(x).
"""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import sympy as sp


@dataclass
class ResultatCout:
    cout: sp.Expr                 # C(x)
    variable: sp.Symbol
    a: sp.Expr
    b: sp.Expr
    cout_marginal: sp.Expr        # C'(x)
    cout_moyen: sp.Expr           # f(x) = C(x)/x
    cout_moyen_derivee: sp.Expr   # f'(x)
    quantite_optimale: sp.Expr = None
    cout_moyen_min: sp.Expr = None
    cout_marginal_au_min: sp.Expr = None
    egalite_verifiee: bool = False


def _sympify(expression, x):
    if isinstance(expression, str):
        try:
            return sp.sympify(expression, locals={"x": x})
        except (sp.SympifyError, SyntaxError, TypeError) as err:
            raise ValueError(f"Expression invalide : {expression!r}") from err
    return expression


def etudier(cout, a, b, variable=None):
    """Étudie le coût C(x) sur [a, b]. Renvoie un ResultatCout."""
    x = variable or sp.Symbol("x", positive=True)
    C = _sympify(cout, x)
    a, b = sp.sympify(a), sp.sympify(b)
    if a > b:
        a, b = b, a

    Cm = sp.simplify(sp.diff(C, x))            # coût marginal
    f = sp.simplify(C / x)                      # coût moyen
    fp = sp.simplify(sp.diff(f, x))            # f'

    # Points stationnaires de f intérieurs à [a, b].
    candidats = []
    try:
        for s in sp.solve(sp.numer(sp.together(fp)), x):
            if s.is_real is False:
                continue
            try:
                if a <= s <= b:
                    candidats.append(s)
            except TypeError:
                pass
    except (NotImplementedError, Exception):  # noqa: BLE001
        pass
    # On ajoute les bornes pour le minimum absolu.
    points = list(candidats) + [a, b]

    quantite = cm_min = marg_min = None
    egalite = False
    if points:
        quantite = min(points, key=lambda s: float(f.subs(x, s)))
        cm_min = sp.simplify(f.subs(x, quantite))
        marg_min = sp.simplify(Cm.subs(x, quantite))
        egalite = bool(sp.simplify(cm_min - marg_min) == 0)

    return ResultatCout(
        cout=C, variable=x, a=a, b=b,
        cout_marginal=Cm, cout_moyen=f, cout_moyen_derivee=fp,
        quantite_optimale=quantite, cout_moyen_min=cm_min,
        cout_marginal_au_min=marg_min, egalite_verifiee=egalite,
    )


def table_valeurs(resultat, pas=None, decimales=2):
    """Tableau de valeurs : liste de dicts {x, C(x), C'(x), coût moyen f(x)}."""
    r = resultat
    x = r.variable
    a, b = float(r.a), float(r.b)
    if pas is None:
        pas = (b - a) / 9 if b > a else 1
    pas = float(pas)

    def arrondi(expr, v):
        try:
            return round(float(expr.subs(x, v)), decimales)
        except (TypeError, ValueError):
            return None

    lignes = []
    v = a
    while v <= b + 1e-9:
        lignes.append({
            "x": round(v, decimales),
            "C(x)": arrondi(r.cout, v),
            "C'(x) (marginal)": arrondi(r.cout_marginal, v),
            "f(x) (coût moyen)": arrondi(r.cout_moyen, v),
        })
        v += pas
    return lignes


def tableau_variations(resultat, decimales=2):
    """Construit un tableau de variations ASCII de f (coût moyen) avec ↗ ↘."""
    r = resultat
    x = r.variable
    a, b = float(r.a), float(r.b)
    fp, f = r.cout_moyen_derivee, r.cout_moyen

    # Points stationnaires intérieurs (où f' = 0), triés.
    interieurs = []
    try:
        for s in sp.solve(sp.numer(sp.together(fp)), x):
            if s.is_real is False:
                continue
            try:
                sv = float(s)
                if a < sv < b:
                    interieurs.append((sv, s))
            except (TypeError, ValueError):
                pass
    except (NotImplementedError, Exception):  # noqa: BLE001
        pass
    interieurs.sort()
    noeuds = [(a, r.a)] + interieurs + [(b, r.b)]

    # Signe de f' sur chaque sous-intervalle (échantillon au milieu).
    signes = []
    for i in range(len(noeuds) - 1):
        milieu = (noeuds[i][0] + noeuds[i + 1][0]) / 2
        try:
            v = float(fp.subs(x, milieu))
        except (TypeError, ValueError):
            v = 0
        signes.append("+" if v > 0 else "−" if v < 0 else "0")

    def val_f(noeud):
        try:
            return round(float(f.subs(x, noeud[1])), decimales)
        except (TypeError, ValueError):
            return "?"

    LN, LI = 11, 7   # largeurs des colonnes (nœud, intervalle)
    ligne_x = "  x     "
    ligne_fp = "  f'(x) "
    ligne_f = "  f(x)  "
    for i, noeud in enumerate(noeuds):
        est_stationnaire = 0 < i < len(noeuds) - 1
        ligne_x += f"{noeud[0]:^{LN}g}" if isinstance(noeud[0], float) else f"{noeud[0]:^{LN}}"
        ligne_fp += f"{('0' if est_stationnaire else '|'):^{LN}}"
        ligne_f += f"{str(val_f(noeud)):^{LN}}"
        if i < len(noeuds) - 1:
            ligne_x += " " * LI
            ligne_fp += f"{signes[i]:^{LI}}"
            ligne_f += f"{('↗' if signes[i] == '+' else '↘' if signes[i] == '−' else '→'):^{LI}}"
    return "\n".join([ligne_x, ligne_fp, ligne_f])


def rapport_texte(resultat):
    r = resultat
    x = r.variable
    f_apart = sp.apart(r.cout_moyen, x)            # forme x + 16 + 256/x
    fp_factor = sp.factor(r.cout_moyen_derivee)    # forme (x-16)(x+16)/x²
    L = []
    L.append("═" * 64)
    L.append(f"  ÉTUDE DU COÛT  C(x) = {r.cout}   sur  [{r.a}, {r.b}]")
    L.append("═" * 64)

    L.append("\nÉTAPE 1 — Coût marginal (dérivée du coût total)")
    L.append(f"   C'(x) = {r.cout_marginal}")

    L.append("\nÉTAPE 2 — Coût unitaire moyen f(x) = C(x) / x")
    L.append(f"   f(x) = {r.cout_moyen}")
    L.append(f"        = {f_apart}   (forme simplifiée)")

    L.append("\nÉTAPE 3 — Dérivée du coût moyen")
    L.append(f"   f'(x) = {r.cout_moyen_derivee}")
    L.append(f"         = {fp_factor}   (forme factorisée)")

    if r.quantite_optimale is not None:
        L.append("\nÉTAPE 4 — Signe de f' et tableau de variations sur [a, b]")
        L.append(f"   f'(x) = 0  en  x = {r.quantite_optimale}")
        L.append(f"   f'(x) < 0 avant (f décroît), f'(x) > 0 après (f croît)")
        L.append("")
        L.append(tableau_variations(r))
        L.append(f"   ⟹ f admet un MINIMUM en x = {r.quantite_optimale}")

        L.append("\nÉTAPE 5 — Quantité optimale et coût minimal")
        L.append(f"   Pour MINIMISER le coût moyen, produire x = {r.quantite_optimale} unités.")
        L.append(f"   Coût moyen minimal f({r.quantite_optimale}) = {r.cout_moyen_min}")

        L.append("\nÉTAPE 6 — Vérification de la propriété  f(x*) = C'(x*)")
        L.append(f"   f({r.quantite_optimale})  = {r.cout_moyen_min}")
        L.append(f"   C'({r.quantite_optimale}) = {r.cout_marginal_au_min}")
        L.append("   ⟹ égalité VÉRIFIÉE ✓ (au minimum, coût moyen = coût marginal)"
                 if r.egalite_verifiee else
                 "   (minimum atteint sur une borne : l'égalité ne s'applique pas ici)")
    L.append("═" * 64)
    return "\n".join(L)


def figure(resultat, nb_points=400):
    """Superpose f(x) (coût moyen) et C'(x) (coût marginal) sur [a, b]."""
    r = resultat
    x = r.variable
    a, b = float(r.a), float(r.b)
    xs = np.linspace(a, b, nb_points)
    f_num = sp.lambdify(x, r.cout_moyen, "numpy")
    cm_num = sp.lambdify(x, r.cout_marginal, "numpy")
    with np.errstate(all="ignore"):
        yf = np.broadcast_to(np.asarray(f_num(xs), dtype=float), xs.shape)
        yc = np.broadcast_to(np.asarray(cm_num(xs), dtype=float), xs.shape)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=yf, mode="lines", name="coût moyen f(x)",
                             line=dict(color="royalblue", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=yc, mode="lines", name="coût marginal C'(x)",
                             line=dict(color="darkorange", width=2)))

    if r.quantite_optimale is not None:
        xv = float(r.quantite_optimale)
        yv = float(r.cout_moyen_min)
        fig.add_trace(go.Scatter(
            x=[xv], y=[yv], mode="markers+text",
            marker=dict(size=13, color="green", symbol="star"),
            text=[f"min coût moyen ({xv:g}; {yv:g})"], textposition="top center",
            name="coût moyen minimal",
        ))
        if r.egalite_verifiee:
            fig.add_vline(x=xv, line_dash="dot", line_color="gray",
                          annotation_text=f"f({xv:g}) = C'({xv:g}) = {yv:g}")

    fig.update_layout(
        title=f"Coût moyen et coût marginal — C(x) = {r.cout}",
        xaxis_title="x (quantité)", yaxis_title="euros",
        template="plotly_white", hovermode="x unified",
    )
    return fig
