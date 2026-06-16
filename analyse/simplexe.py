"""Méthode du simplexe (tableaux) pour un programme linéaire — Feuille 4.

Implémentation de la forme « standard facile » :
    max  z = c·x
    s.c. A·x ≤ b,  x ≥ 0,  avec b ≥ 0.

On ajoute une variable d'écart par contrainte, puis on effectue les pivotages
successifs. Chaque tableau intermédiaire est conservé (comme en TD).

Pour un problème de minimisation ou avec contraintes ≥, passer par le dual
(voir module `dualite`) qui se ramène à ce cas.

Tous les calculs sont exacts (fractions SymPy).
"""

from dataclasses import dataclass, field

import sympy as sp


@dataclass
class EtapeSimplexe:
    tableau: sp.Matrix
    base: list           # noms des variables de base (une par contrainte)
    entrante: str = ""   # variable entrante choisie pour ce pivot
    sortante: str = ""   # variable sortante


@dataclass
class ResultatSimplexe:
    noms_variables: list             # ['x1', .., 'xn', 's1', .., 'sm']
    etapes: list = field(default_factory=list)
    solution: dict = field(default_factory=dict)   # nom -> valeur
    valeur_optimale: sp.Expr = None
    statut: str = "optimal"          # "optimal" | "non borné"


def _verifier_forme(pl):
    if pl.sens != "max":
        raise ValueError(
            "Le simplexe direct exige une MAXIMISATION. "
            "Pour un min, résolvez le dual (module dualite)."
        )
    for c in pl.contraintes:
        if c.sens != "<=":
            raise ValueError(
                f"Contrainte '{c.nom}' de type '{c.sens}' : le simplexe direct "
                "exige des contraintes '≤'. Passez par le dual sinon."
            )
        if c.rhs < 0:
            raise ValueError(
                f"Second membre négatif sur '{c.nom}'. Cas non géré "
                "(nécessiterait la méthode des deux phases / Big-M)."
            )


def resoudre(pl):
    """Résout le PL (max, ≤, b≥0) par la méthode du simplexe."""
    _verifier_forme(pl)

    n = len(pl.objectif)            # variables de décision
    m = len(pl.contraintes)         # contraintes (= variables d'écart)
    noms = [f"x{j+1}" for j in range(n)] + [f"s{i+1}" for i in range(m)]
    N = n + m

    # Construction du tableau initial : m lignes de contraintes + 1 ligne objectif.
    lignes = []
    for i, c in enumerate(pl.contraintes):
        ligne = [sp.sympify(c.coeffs[j]) for j in range(n)]      # coeffs x
        ligne += [sp.Integer(1) if k == i else sp.Integer(0) for k in range(m)]  # écarts
        ligne.append(sp.sympify(c.rhs))                          # rhs
        lignes.append(ligne)
    # Ligne objectif : z - c·x = 0  ->  coefficients -c_j sur les x.
    ligne_obj = [-sp.sympify(pl.objectif[j]) for j in range(n)] + [sp.Integer(0)] * m + [sp.Integer(0)]
    lignes.append(ligne_obj)

    tableau = sp.Matrix(lignes)
    base = [n + i for i in range(m)]   # indices : les variables d'écart au départ

    etapes = [EtapeSimplexe(tableau.copy(), [noms[b] for b in base])]

    while True:
        # Variable entrante : coefficient le plus négatif de la ligne objectif.
        ligne_objectif = tableau.row(m)
        col_entrante, valeur_min = None, 0
        for j in range(N):
            if ligne_objectif[j] < valeur_min:
                valeur_min = ligne_objectif[j]
                col_entrante = j
        if col_entrante is None:
            break  # optimal : plus aucun coefficient négatif

        # Test du ratio (variable sortante).
        ligne_sortante, meilleur_ratio = None, None
        for i in range(m):
            pivot = tableau[i, col_entrante]
            if pivot > 0:
                ratio = tableau[i, N] / pivot
                if meilleur_ratio is None or ratio < meilleur_ratio:
                    meilleur_ratio = ratio
                    ligne_sortante = i
        if ligne_sortante is None:
            etapes[-1].entrante = noms[col_entrante]
            return ResultatSimplexe(noms, etapes, {}, None, "non borné")

        # Pivotage.
        pivot = tableau[ligne_sortante, col_entrante]
        tableau[ligne_sortante, :] = tableau[ligne_sortante, :] / pivot
        for i in range(m + 1):
            if i != ligne_sortante and tableau[i, col_entrante] != 0:
                tableau[i, :] = tableau[i, :] - tableau[i, col_entrante] * tableau[ligne_sortante, :]

        etapes[-1].entrante = noms[col_entrante]
        etapes[-1].sortante = noms[base[ligne_sortante]]
        base[ligne_sortante] = col_entrante
        etapes.append(EtapeSimplexe(tableau.copy(), [noms[b] for b in base]))

    # Lecture de la solution.
    solution = {nom: sp.Integer(0) for nom in noms}
    for i in range(m):
        solution[noms[base[i]]] = tableau[i, N]
    valeur = tableau[m, N]

    return ResultatSimplexe(noms, etapes, solution, sp.simplify(valeur), "optimal")


def _formater_tableau(etape, noms):
    N = len(noms)
    entetes = ["base"] + noms + ["b"]
    largeur = max(6, max(len(e) for e in entetes) + 1)

    def cell(v):
        return f"{str(v):>{largeur}}"

    lignes = ["".join(cell(e) for e in entetes)]
    m = etape.tableau.rows - 1
    for i in range(m):
        base_nom = etape.base[i]
        ligne = [base_nom] + [etape.tableau[i, j] for j in range(N + 1)]
        lignes.append("".join(cell(v) for v in ligne))
    ligne_z = ["z"] + [etape.tableau[m, j] for j in range(N + 1)]
    lignes.append("".join(cell(v) for v in ligne_z))
    return "\n".join(lignes)


def rapport_texte(resultat):
    r = resultat
    lignes = []
    lignes.append("═" * 64)
    lignes.append("  MÉTHODE DU SIMPLEXE")
    lignes.append("═" * 64)
    lignes.append("MÉTHODE — On ajoute une variable d'écart (s1, s2, …) par")
    lignes.append("contrainte ≤ pour passer aux égalités, puis on pivote :")
    lignes.append("  • VARIABLE ENTRANTE : colonne au coefficient le plus négatif")
    lignes.append("    de la ligne z (celle qui fait le plus monter z).")
    lignes.append("  • VARIABLE SORTANTE : règle du rapport mini (b / coeff > 0).")
    lignes.append("  • On s'arrête quand la ligne z n'a plus de coefficient négatif.")
    for k, etape in enumerate(r.etapes):
        titre = f"Tableau {k}"
        if etape.entrante:
            titre += f"   (entrante : {etape.entrante}, sortante : {etape.sortante or '—'})"
        lignes.append("\n" + titre)
        lignes.append(_formater_tableau(etape, r.noms_variables))

    if r.statut != "optimal":
        lignes.append(f"\n• Statut : {r.statut.upper()}")
        lignes.append("═" * 64)
        return "\n".join(lignes)

    lignes.append("\n• SOLUTION OPTIMALE :")
    for nom, val in r.solution.items():
        if val != 0:
            lignes.append(f"    {nom} = {val}")
    lignes.append(f"    z* = {r.valeur_optimale}")
    lignes.append("═" * 64)
    return "\n".join(lignes)
