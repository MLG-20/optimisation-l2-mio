# Optimisation — Boîte à outils (L2 MIO, Université de Thiès)

Outils de calcul et de visualisation couvrant le cours d'optimisation :
analyse de fonctions, optimisation libre / sous contrainte / bornée,
analyse marginale, et programmation linéaire (graphique, simplexe, dualité).

> ⚠️ **Outil pédagogique** : pensez à toujours vérifier vos résultats. Il aide à
> comprendre la démarche, il ne remplace pas votre raisonnement.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### Application web (recommandé)

```bash
source .venv/bin/activate
streamlit run app.py
```
Puis ouvrir l'adresse affichée (par défaut http://localhost:8501).

### Menu en ligne de commande

```bash
source .venv/bin/activate
python main.py
```

## Fonctionnalités

| Outil | Contenu | Cours |
|-------|---------|-------|
| Une variable f(x) | domaine, dérivées, extrema, asymptotes | Feuille 2 |
| Intervalle [a, b] | min/max absolus (points critiques + bornes) | Feuille 2 |
| Deux variables f(x, y) | gradient, hessienne, nature (+ repli si det = 0) | Feuille 2, Examen |
| Sous contrainte | Lagrange **et** substitution | Feuille 2, Examen |
| Analyse marginale | coût/recette/profit marginal, coût moyen | Feuille 1 |
| Programmation linéaire | graphique, simplexe, dualité | Feuilles 3 & 4 |

## Structure

```
analyse/            modules de calcul (réutilisables)
  une_variable.py   deux_variables.py   lagrange.py
  marginal.py       lineaire.py         simplexe.py   dualite.py
  affichage.py      tracés Plotly (figure_* et tracer_* HTML)
app.py              application web Streamlit
main.py             menu en ligne de commande
tests/              tests pytest
```

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Syntaxe des fonctions

Notation Python / SymPy : `**` puissance, `*` produit, `sin(x)`, `cos(x)`,
`exp(x)`, `log(x)`, `sqrt(x)`. Exemples : `x**3 - 3*x`, `(x**2-1)/(x-2)`,
`x**2 + y**2 - 3*x*y`. La saisie « naturelle » est aussi acceptée :
`√x`, `x²`, `2x`, `3xy`.

## Auteur

**Mamadou Lamine Gueye** — développeur web. Étudiant en L2 MIO à l'Université
de Thiès, j'ai créé cet outil pour aider mes cadets à **comprendre concrètement**
l'optimisation.

## Licence

Projet **open source** sous licence **MIT** (voir [LICENSE](LICENSE)) : libre
d'utilisation, de modification et de partage. 💚
