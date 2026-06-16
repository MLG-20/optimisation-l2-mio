"""Étude de fonctions & optimisation — programme interactif.

Menu en ligne de commande couvrant le cours d'optimisation (L2 MIO) :
analyse de fonctions 1 et 2 variables, optimisation sur intervalle borné,
optimisation sous contrainte (Lagrange), et analyse marginale (économie).
Rapports détaillés + tracés interactifs (Plotly / HTML).
"""

import webbrowser
from pathlib import Path

from analyse import (
    affichage,
    deux_variables,
    dualite,
    lagrange,
    lineaire,
    marginal,
    simplexe,
    une_variable,
)


def _ouvrir_dans_navigateur(chemin):
    try:
        webbrowser.open(Path(chemin).resolve().as_uri())
    except Exception:  # noqa: BLE001
        pass


def _demander(invite):
    """Lit une saisie non vide ; renvoie None si vide."""
    valeur = input(invite).strip()
    return valeur or None


def etude_une_variable():
    print("\n--- Fonction à une variable f(x) ---")
    print("Exemples : x**3 - 3*x  |  (x**2 - 1)/(x - 2)  |  sin(x)  |  exp(-x**2)")
    expression = _demander("Entrez f(x) : ")
    if not expression:
        return
    try:
        resultat = une_variable.analyser(expression)
    except ValueError as err:
        print(f"\n⚠️  {err}")
        return
    print("\n" + une_variable.rapport_texte(resultat))
    if (input("\nTracer la courbe ? (o/n) ").strip().lower().startswith("o")):
        chemin = affichage.tracer_1d(resultat)
        print(f"✓ Graphique : {chemin}")
        _ouvrir_dans_navigateur(chemin)


def etude_intervalle():
    print("\n--- Optimisation sur un intervalle fermé [a, b] ---")
    print("Exemple : f(x) = x**3 - 12*x + 9 sur [0, 3]")
    expression = _demander("Entrez f(x) : ")
    if not expression:
        return
    a = _demander("Borne a : ")
    b = _demander("Borne b : ")
    if a is None or b is None:
        print("Bornes manquantes.")
        return
    try:
        resultat = une_variable.optimiser_sur_intervalle(expression, a, b)
    except (ValueError, TypeError) as err:
        print(f"\n⚠️  {err}")
        return
    print("\n" + une_variable.rapport_intervalle(resultat))


def etude_deux_variables():
    print("\n--- Fonction à deux variables f(x, y) ---")
    print("Exemples : x**2 + y**2  |  x**2 - y**2  |  x**3 + y**3 - 3*x*y")
    expression = _demander("Entrez f(x, y) : ")
    if not expression:
        return
    try:
        resultat = deux_variables.analyser(expression)
    except ValueError as err:
        print(f"\n⚠️  {err}")
        return
    print("\n" + deux_variables.rapport_texte(resultat))
    if (input("\nTracer la surface 3D ? (o/n) ").strip().lower().startswith("o")):
        chemin = affichage.tracer_2d(resultat)
        print(f"✓ Graphique : {chemin}")
        _ouvrir_dans_navigateur(chemin)


def etude_lagrange():
    print("\n--- Optimisation sous contrainte (2 variables) ---")
    print("Exemple : maximiser x*y sous la contrainte x + y = 10")
    print("On écrit la contrainte sous la forme g(x, y) = 0, ici : x + y - 10")
    objectif = _demander("Objectif f(x, y) : ")
    contrainte = _demander("Contrainte g(x, y) = 0, entrez g : ")
    if not objectif or not contrainte:
        return
    print("\nMéthode : 1=Lagrange  2=Substitution  3=Les deux (comparer)")
    methode = _demander("Votre choix : ") or "3"
    try:
        if methode in ("1", "3"):
            print("\n" + lagrange.rapport_texte(
                lagrange.analyser(objectif, contrainte)))
        if methode in ("2", "3"):
            print("\n" + lagrange.rapport_substitution(
                lagrange.resoudre_par_substitution(objectif, contrainte)))
    except ValueError as err:
        print(f"\n⚠️  {err}")


def analyse_marginale():
    print("\n--- Analyse marginale (économie) ---")
    print("  1. Coût/recette/profit marginal d'une fonction")
    print("  2. Coût unitaire moyen et sa minimisation")
    print("  3. Profit maximal (à partir de R et C)")
    choix = _demander("Votre choix : ")
    try:
        if choix == "1":
            f = _demander("Fonction (ex. coût C(x)) : ")
            x0 = _demander("Évaluer en x0 (laisser vide si non) : ")
            res = marginal.fonction_marginale(f, point=x0)
            print("\n" + marginal.rapport_marginal(res))
        elif choix == "2":
            C = _demander("Coût C(x) : ")
            print("\n" + marginal.rapport_cout_moyen(marginal.cout_moyen(C)))
        elif choix == "3":
            R = _demander("Recette R(x) : ")
            C = _demander("Coût C(x) : ")
            print("\n" + marginal.rapport_profit(marginal.profit(R, C)))
        else:
            print("Choix invalide.")
    except ValueError as err:
        print(f"\n⚠️  {err}")


def _saisir_pl():
    """Saisie interactive d'un programme linéaire. Renvoie un PL ou None."""
    print("Exemple Feuille 4 Exo 1 :")
    print("  objectif : 4 5      sens : max")
    print("  contrainte : 2 1 <= 800   (pour 2·x1 + 1·x2 <= 800)")
    coeffs = _demander("Coefficients objectif (c1 c2) : ")
    sens = _demander("max ou min : ")
    if not coeffs or not sens:
        return None
    morceaux_obj = coeffs.split()
    if len(morceaux_obj) != 2:
        print("Coefficients invalides (attendu : deux nombres, ex. '4 5').")
        return None

    print("Entrez les contraintes une par une (ex. '2 1 <= 800').")
    print("Ligne vide pour terminer. (x1 ≥ 0, x2 ≥ 0 sont automatiques.)")
    contraintes = []
    i = 1
    while True:
        ligne = input(f"  Contrainte {i} : ").strip()
        if not ligne:
            break
        morceaux = ligne.split()
        if len(morceaux) != 4 or morceaux[2] not in ("<=", ">=", "="):
            print("    Format attendu : a1 a2 <= b   (sens parmi <= >= =)")
            continue
        contraintes.append(((morceaux[0], morceaux[1]), morceaux[2], morceaux[3]))
        i += 1

    if not contraintes:
        print("Aucune contrainte saisie.")
        return None
    return lineaire.programme(tuple(morceaux_obj), sens, contraintes)


def programme_lineaire():
    print("\n--- Programmation linéaire (2 variables) ---")
    print("  1. Résolution graphique (domaine réalisable + optimum)")
    print("  2. Méthode du simplexe (tableaux)")
    print("  3. Dualité (primal → dual, résolution du dual)")
    methode = _demander("Méthode : ")
    if methode not in ("1", "2", "3"):
        print("Choix invalide.")
        return

    pl = _saisir_pl()
    if pl is None:
        return

    try:
        if methode == "1":
            resultat = lineaire.resoudre_graphique(pl)
            print("\n" + lineaire.rapport_texte(resultat))
            if (resultat.statut == "optimal"
                    and input("\nTracer le domaine réalisable ? (o/n) ")
                    .strip().lower().startswith("o")):
                chemin = lineaire.tracer_domaine(resultat)
                print(f"✓ Graphique : {chemin}")
                _ouvrir_dans_navigateur(chemin)

        elif methode == "2":
            print("\n" + simplexe.rapport_texte(simplexe.resoudre(pl)))

        else:  # dualité
            d = dualite.dual(pl)
            print("\n" + dualite.rapport_texte(pl, d))
            print("\n--- Résolution du dual par le simplexe ---")
            res_dual = simplexe.resoudre(d)
            print(simplexe.rapport_texte(res_dual))
            if res_dual.statut == "optimal":
                primal = dualite.solution_primale_depuis_dual(res_dual)
                print("\n• Solution optimale du PRIMAL (lue dans le dual) :")
                for nom, val in primal.items():
                    print(f"    {nom} = {val}")
                print(f"    z* = {res_dual.valeur_optimale}")
    except ValueError as err:
        print(f"\n⚠️  {err}")


def menu():
    actions = {
        "1": etude_une_variable,
        "2": etude_intervalle,
        "3": etude_deux_variables,
        "4": etude_lagrange,
        "5": analyse_marginale,
        "6": programme_lineaire,
    }
    while True:
        print("\n" + "=" * 52)
        print("  ÉTUDE DE FONCTIONS & OPTIMISATION")
        print("=" * 52)
        print("  1. Fonction à une variable        f(x)")
        print("  2. Optimisation sur intervalle    [a, b]")
        print("  3. Fonction à deux variables      f(x, y)")
        print("  4. Optimisation sous contrainte   (Lagrange / substitution)")
        print("  5. Analyse marginale              (économie)")
        print("  6. Programmation linéaire         (graphique / simplexe / dualité)")
        print("  7. Quitter")
        choix = input("Votre choix : ").strip()
        if choix == "7":
            print("Au revoir !")
            break
        action = actions.get(choix)
        if action:
            action()
        else:
            print("Choix invalide, réessayez.")


if __name__ == "__main__":
    try:
        menu()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrompu. Au revoir !")
