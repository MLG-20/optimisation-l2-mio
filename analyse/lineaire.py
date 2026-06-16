"""Programmation linéaire à 2 variables — résolution graphique (Feuille 4).

Principe : l'optimum d'un programme linéaire (s'il existe) est atteint en un
SOMMET du domaine réalisable (polygone convexe). On énumère donc tous les
sommets (intersections de paires de droites de contraintes qui sont
réalisables), puis on évalue la fonction objectif sur chacun.

Tout est calculé en exact (fractions SymPy).
"""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import sympy as sp


@dataclass
class Contrainte:
    coeffs: tuple        # (a1, a2) pour a1·x1 + a2·x2
    sens: str            # "<=", ">=", "="
    rhs: sp.Expr         # second membre b
    nom: str = ""


@dataclass
class ProgrammeLineaire:
    objectif: tuple              # (c1, c2) pour z = c1·x1 + c2·x2
    sens: str                    # "max" ou "min"
    contraintes: list            # liste de Contrainte
    positives: bool = True       # x1 >= 0 et x2 >= 0


@dataclass
class Sommet:
    point: tuple         # (x1, x2)
    valeur: sp.Expr      # z au sommet
    contraintes_saturees: list = field(default_factory=list)


@dataclass
class ResultatLineaire:
    programme: ProgrammeLineaire
    sommets: list = field(default_factory=list)   # tous les sommets réalisables
    optimum: Sommet = None
    statut: str = "optimal"   # "optimal" | "irréalisable" | "non borné"


def programme(objectif, sens, contraintes, positives=True):
    """Construit un ProgrammeLineaire.

    `contraintes` : liste de tuples ((a1, a2), sens, b[, nom]).
    Exemple :
        programme((4, 5), "max",
                  [((2, 1), "<=", 800), ((1, 2), "<=", 700), ((0, 1), "<=", 300)])
    """
    objets = []
    for i, c in enumerate(contraintes, start=1):
        coeffs, sgn, b = c[0], c[1], c[2]
        nom = c[3] if len(c) > 3 else f"C{i}"
        objets.append(Contrainte(
            tuple(sp.sympify(v) for v in coeffs), sgn, sp.sympify(b), nom
        ))
    return ProgrammeLineaire(
        objectif=tuple(sp.sympify(v) for v in objectif),
        sens=sens.lower(), contraintes=objets, positives=positives,
    )


def _toutes_les_contraintes(pl):
    """Contraintes explicites + non-négativité, comme liste de Contrainte."""
    liste = list(pl.contraintes)
    if pl.positives:
        liste.append(Contrainte((1, 0), ">=", 0, "x1>=0"))
        liste.append(Contrainte((0, 1), ">=", 0, "x2>=0"))
    return liste


def _satisfait(point, contrainte, tol=sp.Rational(1, 10**9)):
    x1, x2 = point
    a1, a2 = contrainte.coeffs
    g = a1 * x1 + a2 * x2 - contrainte.rhs
    if contrainte.sens == "<=":
        return g <= tol
    if contrainte.sens == ">=":
        return g >= -tol
    return abs(g) <= tol


def _intersection(c1, c2):
    """Point d'intersection de deux droites a·x = b, ou None si parallèles."""
    a1, a2 = c1.coeffs
    b1 = c1.rhs
    a3, a4 = c2.coeffs
    b2 = c2.rhs
    M = sp.Matrix([[a1, a2], [a3, a4]])
    if M.det() == 0:
        return None
    sol = M.LUsolve(sp.Matrix([b1, b2]))
    return (sp.nsimplify(sol[0]), sp.nsimplify(sol[1]))


def resoudre_graphique(pl):
    """Résout le PL par énumération des sommets du domaine réalisable."""
    contraintes = _toutes_les_contraintes(pl)
    c1, c2 = pl.objectif

    sommets = []
    vus = set()
    n = len(contraintes)
    for i in range(n):
        for j in range(i + 1, n):
            pt = _intersection(contraintes[i], contraintes[j])
            if pt is None:
                continue
            if not all(_satisfait(pt, c) for c in contraintes):
                continue
            cle = (sp.nsimplify(pt[0]), sp.nsimplify(pt[1]))
            if cle in vus:
                continue
            vus.add(cle)
            satures = [c.nom for c in contraintes
                       if abs(c.coeffs[0] * pt[0] + c.coeffs[1] * pt[1] - c.rhs) == 0]
            z = sp.simplify(c1 * pt[0] + c2 * pt[1])
            sommets.append(Sommet(pt, z, satures))

    if not sommets:
        return ResultatLineaire(pl, [], None, "irréalisable")

    # Tri des sommets pour un affichage cohérent (angle autour du centre).
    cx = sum(s.point[0] for s in sommets) / len(sommets)
    cy = sum(s.point[1] for s in sommets) / len(sommets)
    sommets.sort(key=lambda s: sp.atan2(s.point[1] - cy, s.point[0] - cx))

    if pl.sens == "max":
        optimum = max(sommets, key=lambda s: float(s.valeur))
    else:
        optimum = min(sommets, key=lambda s: float(s.valeur))

    return ResultatLineaire(pl, sommets, optimum, "optimal")


def rapport_texte(resultat):
    r = resultat
    pl = r.programme
    c1, c2 = pl.objectif
    lignes = []
    lignes.append("═" * 64)
    lignes.append(f"  PROGRAMME LINÉAIRE — résolution graphique  ({pl.sens.upper()})")
    lignes.append("═" * 64)
    lignes.append(f"  Objectif : z = {c1}·x1 + {c2}·x2   →   {pl.sens.upper()}")
    lignes.append("  Contraintes :")
    for c in pl.contraintes:
        lignes.append(f"    {c.coeffs[0]}·x1 + {c.coeffs[1]}·x2 {c.sens} {c.rhs}   ({c.nom})")
    if pl.positives:
        lignes.append("    x1 ≥ 0,  x2 ≥ 0")

    lignes.append("\nMÉTHODE — L'optimum d'un PL est atteint en un SOMMET du")
    lignes.append("domaine réalisable. On trace les contraintes, on liste les")
    lignes.append("sommets (intersections réalisables), puis on évalue z en chacun.")

    if r.statut != "optimal":
        lignes.append(f"\n• Statut : {r.statut.upper()}")
        lignes.append("═" * 64)
        return "\n".join(lignes)

    lignes.append("\nÉTAPE 1 — Sommets du domaine réalisable et valeur de z :")
    for s in r.sommets:
        lignes.append(f"    ({s.point[0]}, {s.point[1]})   →   z = {s.valeur}")

    o = r.optimum
    sens_txt = "la plus GRANDE" if pl.sens == "max" else "la plus PETITE"
    lignes.append(f"\nÉTAPE 2 — On retient la valeur {sens_txt} de z :")
    lignes.append(f"   SOLUTION OPTIMALE : x1 = {o.point[0]},  x2 = {o.point[1]}")
    lignes.append(f"   z* = {o.valeur}")
    lignes.append(f"   contraintes saturées (actives à l'optimum) : "
                  f"{', '.join(o.contraintes_saturees)}")
    lignes.append("   (saturées = vérifiées avec égalité ; elles définissent le sommet)")
    lignes.append("═" * 64)
    return "\n".join(lignes)


def figure_domaine(resultat):
    """Construit et renvoie la figure Plotly du domaine réalisable + optimum."""
    r = resultat
    pl = r.programme
    if not r.sommets:
        raise ValueError("Aucun domaine réalisable à tracer.")

    xs = [float(s.point[0]) for s in r.sommets]
    ys = [float(s.point[1]) for s in r.sommets]
    x_max = max(xs + [0]) * 1.3 + 1
    y_max = max(ys + [0]) * 1.3 + 1

    fig = go.Figure()

    # Domaine réalisable (polygone des sommets, déjà triés par angle).
    fig.add_trace(go.Scatter(
        x=xs + [xs[0]], y=ys + [ys[0]],
        fill="toself", fillcolor="rgba(65,105,225,0.20)",
        line=dict(color="royalblue"), name="domaine réalisable",
        hoverinfo="skip",
    ))

    # Droites de contraintes.
    t = np.linspace(0, x_max, 200)
    for c in pl.contraintes:
        a1, a2 = float(c.coeffs[0]), float(c.coeffs[1])
        b = float(c.rhs)
        if a2 != 0:
            fig.add_trace(go.Scatter(
                x=t, y=(b - a1 * t) / a2, mode="lines",
                line=dict(dash="dash"), name=f"{c.nom}: {c.coeffs[0]}x1+{c.coeffs[1]}x2{c.sens}{c.rhs}",
            ))
        elif a1 != 0:  # droite verticale x1 = b/a1
            xv = b / a1
            fig.add_trace(go.Scatter(
                x=[xv, xv], y=[0, y_max], mode="lines",
                line=dict(dash="dash"), name=f"{c.nom}: x1{c.sens}{xv:g}",
            ))

    # Sommets.
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=8, color="navy"),
        text=[f"({s.point[0]},{s.point[1]})" for s in r.sommets],
        textposition="top right", name="sommets",
    ))

    # Optimum.
    if r.optimum is not None:
        o = r.optimum
        fig.add_trace(go.Scatter(
            x=[float(o.point[0])], y=[float(o.point[1])],
            mode="markers+text", marker=dict(size=14, color="red", symbol="star"),
            text=[f"OPTIMUM z*={o.valeur}"], textposition="bottom center",
            name="optimum",
        ))

    c1, c2 = pl.objectif
    fig.update_layout(
        title=f"PL ({pl.sens.upper()})  z = {c1}·x1 + {c2}·x2",
        xaxis_title="x1", yaxis_title="x2",
        xaxis=dict(range=[0, x_max]), yaxis=dict(range=[0, y_max]),
        template="plotly_white",
    )
    return fig


def tracer_domaine(resultat, fichier="domaine_lineaire.html"):
    """Trace le domaine réalisable et exporte en HTML. Renvoie le chemin."""
    fig = figure_domaine(resultat)
    fig.write_html(fichier, include_plotlyjs="cdn")
    return fichier
