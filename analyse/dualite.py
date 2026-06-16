"""Dualité en programmation linéaire — Feuille 4.

Construit le programme DUAL à partir d'un programme primal :

  Primal (max) :  max c·x,  A·x ≤ b,  x ≥ 0
  Dual   (min) :  min b·y,  Aᵀ·y ≥ c,  y ≥ 0

  Primal (min) :  min c·x,  A·x ≥ b,  x ≥ 0
  Dual   (max) :  max b·y,  Aᵀ·y ≤ c,  y ≥ 0

Le théorème de dualité forte garantit : z*(primal) = z*(dual).
Pratique : un min avec contraintes ≥ devient un max avec contraintes ≤,
directement résoluble par le simplexe.
"""

import sympy as sp

from .lineaire import programme


def dual(pl):
    """Renvoie le programme dual (un ProgrammeLineaire) du programme `pl`.

    Hypothèses : primal 'max' avec toutes ses contraintes '≤', ou primal 'min'
    avec toutes ses contraintes '≥'.
    """
    n = len(pl.objectif)          # variables primales
    m = len(pl.contraintes)       # contraintes primales -> variables duales

    if pl.sens == "max":
        if any(c.sens != "<=" for c in pl.contraintes):
            raise ValueError("Primal 'max' attendu avec toutes les contraintes '≤'.")
        sens_dual = "min"
        sens_contrainte = ">="
    elif pl.sens == "min":
        if any(c.sens != ">=" for c in pl.contraintes):
            raise ValueError("Primal 'min' attendu avec toutes les contraintes '≥'.")
        sens_dual = "max"
        sens_contrainte = "<="
    else:
        raise ValueError("Le sens du programme doit être 'max' ou 'min'.")

    # Objectif dual : les seconds membres b du primal.
    objectif_dual = tuple(c.rhs for c in pl.contraintes)

    # Contraintes duales : une par variable primale (colonnes de A = Aᵀ).
    contraintes_duales = []
    for j in range(n):
        colonne = tuple(pl.contraintes[i].coeffs[j] for i in range(m))
        contraintes_duales.append((colonne, sens_contrainte, pl.objectif[j], f"D{j+1}"))

    return programme(objectif_dual, sens_dual, contraintes_duales)


def solution_primale_depuis_dual(resultat_simplexe_dual):
    """Lit la solution optimale du PRIMAL dans le tableau final du dual.

    Les valeurs optimales des variables primales se lisent dans la ligne z
    du dernier tableau du simplexe, sous les colonnes des variables d'écart.
    """
    derniere = resultat_simplexe_dual.etapes[-1].tableau
    m = derniere.rows - 1
    ligne_z = derniere.row(m)
    primal = {}
    j = 0
    for idx, nom in enumerate(resultat_simplexe_dual.noms_variables):
        if nom.startswith("s"):
            j += 1
            primal[f"x{j}"] = ligne_z[idx]
    return primal


def rapport_texte(primal, dual_pl):
    p, d = primal, dual_pl
    lignes = []
    lignes.append("═" * 64)
    lignes.append("  DUALITÉ  (primal  →  dual)")
    lignes.append("═" * 64)

    def bloc(titre, pl, var):
        sous = [f"  {titre} :", f"    {pl.sens.upper()}  "
                + " + ".join(f"{pl.objectif[k]}·{var}{k+1}" for k in range(len(pl.objectif)))]
        for c in pl.contraintes:
            sous.append("      "
                        + " + ".join(f"{c.coeffs[k]}·{var}{k+1}" for k in range(len(c.coeffs)))
                        + f" {c.sens} {c.rhs}")
        sous.append(f"      {var}i ≥ 0")
        return "\n".join(sous)

    lignes.append("RÈGLE DE CONSTRUCTION DU DUAL :")
    lignes.append("  • une variable duale yᵢ par contrainte du primal ;")
    lignes.append("  • max ⟷ min ; les sens des contraintes s'inversent (≤ ⟷ ≥) ;")
    lignes.append("  • on TRANSPOSE la matrice A (les colonnes deviennent les lignes) ;")
    lignes.append("  • l'objectif du dual = seconds membres b du primal, et")
    lignes.append("    les seconds membres du dual = coefficients c de l'objectif primal.")
    lignes.append("  • Dualité forte : z*(primal) = z*(dual).")
    lignes.append("")
    lignes.append(bloc("PRIMAL", p, "x"))
    lignes.append("")
    lignes.append(bloc("DUAL", d, "y"))
    lignes.append("═" * 64)
    return "\n".join(lignes)
