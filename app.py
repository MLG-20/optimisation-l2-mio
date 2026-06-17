"""Application web (Streamlit) — Étude de fonctions & Optimisation (L2 MIO).

Lancement :  streamlit run app.py
Réutilise les modules du package `analyse/`.
"""

import re

import streamlit as st
import streamlit.components.v1 as components

from analyse import (
    affichage,
    deux_variables,
    dualite,
    etude_cout,
    lagrange,
    lineaire,
    marginal,
    simplexe,
    une_variable,
)

st.set_page_config(page_title="Optimisation — Dr Gueye", page_icon="📐",
                   layout="wide")


def declarer_langue_francaise():
    """Empêche la traduction automatique de Chrome (qui déforme les libellés
    français en croyant que la page est en anglais)."""
    components.html(
        "<script>"
        "var d = window.parent.document;"
        "d.documentElement.setAttribute('lang', 'fr');"
        "d.documentElement.setAttribute('translate', 'no');"
        "if (d.body) { d.body.classList.add('notranslate'); "
        "             d.body.setAttribute('translate', 'no'); }"
        "var m = d.querySelector('meta[name=google]') || d.createElement('meta');"
        "m.name = 'google'; m.content = 'notranslate';"
        "d.head.appendChild(m);"
        "var c = d.querySelector('meta[http-equiv=\"Content-Language\"]') "
        "|| d.createElement('meta');"
        "c.setAttribute('http-equiv', 'Content-Language'); c.content = 'fr';"
        "d.head.appendChild(c);"
        "</script>",
        height=0,
    )


# --------------------------------------------------------------------------- #
# Utilitaires d'affichage
# --------------------------------------------------------------------------- #
_EXPOSANTS = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
              "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}


def jolifier(texte):
    """Convertit la notation informatique en notation mathématique lisible :
    sqrt → √, x**2 → x², a*b → a·b. (Affichage uniquement.)"""
    texte = texte.replace("sqrt", "√")
    # puissances entières : x**2 -> x², x**10 -> x¹⁰
    texte = re.sub(r"\*\*(\d+)",
                   lambda m: "".join(_EXPOSANTS[d] for d in m.group(1)), texte)
    texte = texte.replace("**", "^")          # puissances restantes (ex. **(-1))
    texte = texte.replace("*", "·")           # produit
    texte = re.sub(r"√\((\w+)\)", r"√\1", texte)   # √(2) -> √2, √(x) -> √x
    return texte


def afficher_rapport(texte):
    st.code(jolifier(texte), language="text")


def afficher_graphe(fig, width="stretch"):
    """Embellit le titre du graphique (sqrt → √, ** → exposant) puis l'affiche."""
    try:
        if fig.layout.title and fig.layout.title.text:
            fig.update_layout(title=jolifier(fig.layout.title.text))
    except Exception:  # noqa: BLE001
        pass
    st.plotly_chart(fig, width=width)


def erreur(msg):
    st.error(f"⚠️ {msg}")


# --------------------------------------------------------------------------- #
# Explications dynamiques des graphiques (lecture du résultat calculé)
# --------------------------------------------------------------------------- #
def _j(expr):
    """Embellit une expression pour l'insérer dans un texte (sqrt → √, etc.)."""
    return jolifier(str(expr))


def expliquer(texte):
    st.info("📖 **Comment lire ce graphique.** " + texte)


def expl_une_variable(res):
    parts = []
    if res.points_critiques:
        for pc in res.points_critiques:
            parts.append(
                f"En **x = {_j(pc.abscisse)}** la courbe atteint un **{pc.nature}** "
                f"(f = {_j(pc.ordonnee)}), marqué par un point coloré.")
    else:
        parts.append("La courbe n'a aucun point critique réel : elle est monotone "
                     "(toujours croissante ou décroissante) sur son domaine.")
    a = res.asymptotes
    asy = []
    if a.get("verticales"):
        asy.append("verticale(s) en " + ", ".join(f"x = {_j(v)}" for v in a["verticales"]))
    if a.get("horizontales"):
        asy.append("horizontale(s) " + ", ".join(f"y = {_j(v)}" for v, _ in a["horizontales"]))
    if a.get("obliques"):
        asy.append("oblique(s)")
    if asy:
        parts.append("Les pointillés gris sont les asymptotes : " + " ; ".join(asy) + ".")
    expliquer(" ".join(parts))


def expl_intervalle(res):
    if not (res.minimum and res.maximum):
        return
    expliquer(
        f"Sur l'intervalle, le **minimum absolu** vaut {_j(res.minimum.valeur)} "
        f"en x = {_j(res.minimum.abscisse)} ({res.minimum.type}), et le "
        f"**maximum absolu** {_j(res.maximum.valeur)} en x = {_j(res.maximum.abscisse)} "
        f"({res.maximum.type}). Les ★ verte et rouge marquent ces deux points ; "
        "compare bien les valeurs aux bornes et aux points critiques.")


def expl_deux_variables(res):
    if not res.points_critiques:
        expliquer("Aucun point critique : la surface n'a ni creux ni sommet visible.")
        return
    lignes = []
    for i, pc in enumerate(res.points_critiques):
        indice = str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
        if "minimum" in pc.nature:
            lecture = "un **creux** (les couleurs s'éclaircissent autour)"
        elif "maximum" in pc.nature:
            lecture = "un **sommet** (les couleurs foncent autour)"
        else:
            lecture = "un **col** (ça monte dans une direction, descend dans l'autre)"
        lignes.append(f"P{indice} = ({_j(pc.point[0])}, {_j(pc.point[1])}) : "
                      f"{pc.nature} → {lecture}.")
    expliquer("Sur la surface 3D et la carte de niveaux 2D :\n\n- " + "\n- ".join(lignes))


def expl_contrainte(res):
    pts = res.points_critiques
    if not pts:
        return
    nums = []
    for pc in pts:
        try:
            nums.append((float(pc.valeur), pc))
        except (TypeError, ValueError):
            pass
    detail = ""
    if nums:
        vmax = max(nums, key=lambda t: t[0])[1]
        vmin = min(nums, key=lambda t: t[0])[1]
        detail = (f" Le **maximum** ({_j(vmax.valeur)}) est en "
                  f"({_j(vmax.point[0])}, {_j(vmax.point[1])}) et le **minimum** "
                  f"({_j(vmin.valeur)}) en ({_j(vmin.point[0])}, {_j(vmin.point[1])}).")
    expliquer(
        "Les points optimaux (★) sont exactement là où une **courbe de niveau de f** "
        "touche (est tangente à) la **contrainte g = 0** (ligne rouge). En ces points, "
        "les flèches **∇f (bleue)** et **∇g (rouge)** sont **parallèles** : c'est la "
        "condition ∇f = λ·∇g." + detail)


def expl_marginale(res):
    if res.point is None:
        expliquer("La courbe orange est la **fonction marginale** f'(x) : sa hauteur "
                  "donne la pente de f(x) (bleue) en chaque point.")
        return
    expliquer(
        f"La courbe orange est la **marginale** f'(x). En x = {_j(res.point)}, elle "
        f"vaut {_j(res.valeur_marginale)} (★) : c'est la pente de f en ce point, donc "
        "≈ l'effet d'**une unité supplémentaire**.")


def expl_cout_moyen(res):
    if res.quantite_optimale is None:
        expliquer("Le coût moyen (bleu) et le coût marginal (orange) sont tracés ; "
                  "le coût moyen n'a pas de minimum intérieur ici.")
        return
    expliquer(
        f"Le **coût moyen** (bleu) est **minimal** en x = {_j(res.quantite_optimale)} "
        f"(★, valeur {_j(res.cout_moyen_min)}). Remarque qu'à ce point précis il "
        "**croise le coût marginal** (orange) : c'est la propriété coût moyen = coût "
        "marginal au minimum.")


def expl_profit(res):
    if not res.quantites_optimales:
        expliquer("On trace recette R(x), coût C(x) et profit P(x) = R − C ; "
                  "aucun maximum de profit n'a été trouvé.")
        return
    q, val = res.quantites_optimales[0]
    expliquer(
        f"Le **profit** (bleu) est **maximal** en x = {_j(q)} (★, P = {_j(val)}). "
        "C'est là où l'**écart entre la recette (vert) et le coût (rouge) est le plus "
        "grand**, c'est-à-dire où recette marginale = coût marginal.")


def expl_etude_cout(res):
    if res.quantite_optimale is None:
        return
    extra = (" À ce point, les deux courbes **se croisent** : f(x*) = C'(x*)."
             if res.egalite_verifiee else "")
    expliquer(
        f"Le **coût moyen** (bleu) est minimal en x = {_j(res.quantite_optimale)} "
        f"(★, valeur {_j(res.cout_moyen_min)}), comparé au **coût marginal** "
        f"(orange).{extra}")


def expl_lineaire(res):
    if res.statut != "optimal" or res.optimum is None:
        return
    o = res.optimum
    expliquer(
        f"La **zone bleue** est le domaine réalisable (toutes les contraintes "
        f"respectées). L'**optimum** (★) est au sommet ({_j(o.point[0])}, "
        f"{_j(o.point[1])}) avec z* = {_j(o.valeur)} : c'est le coin du polygone qui "
        f"pousse l'objectif le plus loin. Contraintes saturées : "
        f"{', '.join(o.contraintes_saturees)}.")


def champ_exemples(label, cle, exemples, defaut, help=None):
    """Champ texte précédé d'un menu d'exemples qui le remplit dynamiquement.

    `exemples` : dict {nom affiché: valeur à insérer}. L'utilisateur garde la
    possibilité de taper librement (option « Saisie libre »).
    """
    st.session_state.setdefault(cle, defaut)
    choix = st.selectbox("📋 Charger un exemple", ["✏️ Saisie libre"] + list(exemples),
                         key=cle + "_ex")
    if choix != "✏️ Saisie libre" and st.session_state.get(cle + "_last") != choix:
        st.session_state[cle] = exemples[choix]
        st.session_state[cle + "_last"] = choix
    return st.text_input(label, key=cle, help=help)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_une_variable():
    st.header("📈 Fonction à une variable  f(x)")
    st.caption("Domaine, dérivées, points critiques, asymptotes, nature des extrema.")
    exemples = {
        "Examen Exo I — f(x) = x³(x²−1)": "x**3*(x**2 - 1)",
        "Examen Exo I — g(x) = x³−1+(x−1)²": "x**3 - 1 + (x - 1)**2",
        "Polynôme x³ − 3x": "x**3 - 3*x",
        "Rationnelle (x²−1)/(x−2)": "(x**2 - 1)/(x - 2)",
        "Gaussienne exp(−x²)": "exp(-x**2)",
        "Trigonométrique sin(x)": "sin(x)",
    }
    expr = champ_exemples("f(x) =", "f_1v", exemples, "x**3 - 3*x",
                          help="Syntaxe : ** puissance, * produit, sin(x), exp(x)…")
    if not expr.strip():
        return
    try:
        res = une_variable.analyser(expr)
    except ValueError as e:
        erreur(e)
        return
    st.subheader("Démarche détaillée")
    afficher_rapport(une_variable.rapport_texte(res))
    st.subheader("Graphique")
    try:
        afficher_graphe(affichage.figure_1d(res), width="stretch")
        expl_une_variable(res)
    except Exception as e:  # noqa: BLE001
        erreur(f"Tracé impossible : {e}")


def page_intervalle():
    st.header("📏 Optimisation sur un intervalle fermé [a, b]")
    st.caption("Points critiques intérieurs + bornes → minimum et maximum absolus.")
    expr = st.text_input("f(x) =", value="x**3 - 12*x + 9")
    c1, c2 = st.columns(2)
    a = c1.text_input("Valeur de a (extrémité gauche)", value="0")
    b = c2.text_input("Valeur de b (extrémité droite)", value="3")
    if not expr.strip():
        return
    try:
        res = une_variable.optimiser_sur_intervalle(expr, a, b)
    except (ValueError, TypeError) as e:
        erreur(e)
        return
    st.subheader("Démarche détaillée")
    afficher_rapport(une_variable.rapport_intervalle(res))
    if res.minimum and res.maximum:
        m1, m2 = st.columns(2)
        m1.success(f"MIN absolu : f({res.minimum.abscisse}) = {res.minimum.valeur}")
        m2.info(f"MAX absolu : f({res.maximum.abscisse}) = {res.maximum.valeur}")
    st.subheader("Graphique sur [a, b]")
    try:
        afficher_graphe(affichage.figure_intervalle(res), width="stretch")
        expl_intervalle(res)
    except Exception as e:  # noqa: BLE001
        erreur(f"Tracé impossible : {e}")


def page_deux_variables():
    st.header("🧊 Fonction à deux variables  f(x, y)")
    st.caption("Gradient, hessienne, nature des points critiques (test du déterminant, "
               "repli numérique si det = 0).")
    exemples = {
        "Examen Exo II.I — x²+2y²+3xy−y+x": "x**2 + 2*y**2 + 3*x*y - y + x",
        "Examen Exo II.II — x⁴+y⁴−x²y²−y² (7 pts)": "x**4 + y**4 - x**2*y**2 - y**2",
        "Selle x³+y³−3xy": "x**3 + y**3 - 3*x*y",
        "Paraboloïde x²+y² (min)": "x**2 + y**2",
        "Selle x²−y²": "x**2 - y**2",
    }
    expr = champ_exemples("f(x, y) =", "f_2v", exemples, "x**3 + y**3 - 3*x*y")
    if not expr.strip():
        return
    try:
        res = deux_variables.analyser(expr)
    except ValueError as e:
        erreur(e)
        return
    st.subheader("Démarche détaillée")
    afficher_rapport(deux_variables.rapport_texte(res))
    st.subheader("Graphique (surface 3D)")
    try:
        afficher_graphe(affichage.figure_2d(res), width="stretch")
    except Exception as e:  # noqa: BLE001
        erreur(f"Tracé impossible : {e}")
    st.subheader("Vue 2D — courbes de niveau")
    st.caption("Minimum = boucles qui se resserrent vers un creux ; maximum = vers "
               "un sommet ; point col = courbes en forme de selle qui se croisent.")
    try:
        afficher_graphe(affichage.figure_contour_2d(res), width="stretch")
        expl_deux_variables(res)
    except Exception as e:  # noqa: BLE001
        erreur(f"Tracé impossible : {e}")


def page_contrainte():
    st.header("🔗 Optimisation sous contrainte  (2 variables)")
    st.caption("Contrainte écrite g(x, y) = 0. Deux méthodes : Lagrange et substitution.")
    exemples = {
        "Examen Exo III — xy sous x²+y²=4": ("x*y", "x**2 + y**2 - 4"),
        "Max xy sous x+y=10": ("x*y", "x + y - 10"),
        "Min x²+y² sous x+y=1": ("x**2 + y**2", "x + y - 1"),
        "Feuille 2 — x²y+2x²−2x−y+1 sous x−y=1": ("x**2*y + 2*x**2 - 2*x - y + 1",
                                                  "x - y - 1"),
    }
    st.session_state.setdefault("c_obj", "x*y")
    st.session_state.setdefault("c_contr", "x + y - 10")
    choix = st.selectbox("📋 Charger un exemple", ["✏️ Saisie libre"] + list(exemples),
                         key="c_ex")
    if choix != "✏️ Saisie libre" and st.session_state.get("c_last") != choix:
        st.session_state["c_obj"], st.session_state["c_contr"] = exemples[choix]
        st.session_state["c_last"] = choix
    c1, c2 = st.columns(2)
    objectif = c1.text_input("Objectif f(x, y) =", key="c_obj")
    contrainte = c2.text_input("Contrainte g(x, y) = 0, entrez g :", key="c_contr")
    methode = st.radio("Méthode", ["Lagrange", "Substitution", "Les deux (comparer)"],
                       horizontal=True)
    if not objectif.strip() or not contrainte.strip():
        return
    try:
        res_lagr = lagrange.analyser(objectif, contrainte)
        if methode in ("Lagrange", "Les deux (comparer)"):
            st.subheader("Méthode de Lagrange")
            afficher_rapport(lagrange.rapport_texte(res_lagr))
        if methode in ("Substitution", "Les deux (comparer)"):
            st.subheader("Méthode par substitution")
            afficher_rapport(lagrange.rapport_substitution(
                lagrange.resoudre_par_substitution(objectif, contrainte)))
    except ValueError as e:
        erreur(e)
        return

    st.subheader("Graphique — interprétation géométrique")
    st.caption("À l'optimum, une courbe de niveau de f est TANGENTE à la contrainte "
               "g = 0 (rouge), et les gradients ∇f (bleu) et ∇g (rouge) sont "
               "PARALLÈLES : c'est exactement la condition ∇f = λ·∇g de Lagrange.")
    try:
        afficher_graphe(affichage.figure_contrainte(res_lagr), width="stretch")
        expl_contrainte(res_lagr)
    except Exception as e:  # noqa: BLE001
        erreur(f"Tracé impossible : {e}")


def page_marginale():
    st.header("💰 Analyse marginale (économie)")
    sous = st.radio("Outil", ["Fonction marginale", "Coût moyen", "Profit maximal"],
                    horizontal=True)
    try:
        if sous == "Fonction marginale":
            f = st.text_input("Fonction (ex. coût C(x))", value="20 + 500*x - x**2")
            x0 = st.text_input("Évaluer en x0 (vide = non)", value="50")
            if f.strip():
                res = marginal.fonction_marginale(f, point=(x0 or None))
                st.subheader("Démarche détaillée")
                afficher_rapport(marginal.rapport_marginal(res))
                st.subheader("Graphique — f(x) et sa marginale f'(x)")
                afficher_graphe(marginal.figure_marginale(res), width="stretch")
                expl_marginale(res)
        elif sous == "Coût moyen":
            C = st.text_input("Coût C(x)", value="x**2 + 16*x + 256")
            if C.strip():
                res = marginal.cout_moyen(C)
                st.subheader("Démarche détaillée")
                afficher_rapport(marginal.rapport_cout_moyen(res))
                st.subheader("Graphique — coût moyen et coût marginal")
                afficher_graphe(marginal.figure_cout_moyen(res), width="stretch")
                expl_cout_moyen(res)
        else:
            c1, c2 = st.columns(2)
            R = c1.text_input("Recette R(x)", value="100*x - x**2")
            C = c2.text_input("Coût C(x)", value="10*x + 20")
            if R.strip() and C.strip():
                res = marginal.profit(R, C)
                st.subheader("Démarche détaillée")
                afficher_rapport(marginal.rapport_profit(res))
                st.subheader("Graphique — recette, coût et profit")
                afficher_graphe(marginal.figure_profit(res), width="stretch")
                expl_profit(res)
    except ValueError as e:
        erreur(e)
    except Exception as e:  # noqa: BLE001
        erreur(f"Tracé impossible : {e}")


def _saisir_pl_web():
    """Widgets de saisie d'un PL. Renvoie un ProgrammeLineaire ou None."""
    c1, c2 = st.columns([2, 1])
    objectif = c1.text_input("Objectif (c1 c2)", value="4 5",
                             help="Deux coefficients séparés par un espace, ex. '4 5'")
    sens = c2.selectbox("Sens", ["max", "min"])
    st.markdown("**Contraintes** (une par ligne, ex. `2 1 <= 800`). "
                "`x1 ≥ 0, x2 ≥ 0` sont automatiques.")
    txt = st.text_area("Contraintes", value="2 1 <= 800\n1 2 <= 700\n0 1 <= 300",
                       height=120, label_visibility="collapsed")

    morceaux_obj = objectif.split()
    if len(morceaux_obj) != 2:
        erreur("Objectif invalide : attendu deux nombres (ex. '4 5').")
        return None
    contraintes = []
    for i, ligne in enumerate(txt.splitlines(), start=1):
        ligne = ligne.strip()
        if not ligne:
            continue
        m = ligne.split()
        if len(m) != 4 or m[2] not in ("<=", ">=", "="):
            erreur(f"Contrainte {i} invalide : « {ligne} » (format : a1 a2 <= b).")
            return None
        contraintes.append(((m[0], m[1]), m[2], m[3]))
    if not contraintes:
        erreur("Aucune contrainte saisie.")
        return None
    return lineaire.programme(tuple(morceaux_obj), sens, contraintes)


def page_lineaire():
    st.header("📊 Programmation linéaire  (2 variables)")
    methode = st.radio("Méthode", ["Résolution graphique", "Simplexe", "Dualité"],
                       horizontal=True)
    pl = _saisir_pl_web()
    if pl is None:
        return
    try:
        if methode == "Résolution graphique":
            res = lineaire.resoudre_graphique(pl)
            st.subheader("Démarche détaillée")
            afficher_rapport(lineaire.rapport_texte(res))
            st.subheader("Graphique (domaine réalisable)")
            if res.statut == "optimal":
                afficher_graphe(lineaire.figure_domaine(res),
                                width="stretch")
                expl_lineaire(res)
            else:
                erreur(f"Statut : {res.statut}")
        elif methode == "Simplexe":
            afficher_rapport(simplexe.rapport_texte(simplexe.resoudre(pl)))
        else:
            d = dualite.dual(pl)
            afficher_rapport(dualite.rapport_texte(pl, d))
            st.subheader("Résolution du dual par le simplexe")
            res_dual = simplexe.resoudre(d)
            afficher_rapport(simplexe.rapport_texte(res_dual))
            if res_dual.statut == "optimal":
                primal = dualite.solution_primale_depuis_dual(res_dual)
                txt = "  ".join(f"{k} = {v}" for k, v in primal.items())
                st.success(f"Solution du PRIMAL (lue dans le dual) : {txt}   "
                           f"|   z* = {res_dual.valeur_optimale}")
    except ValueError as e:
        erreur(e)


def page_etude_cout():
    st.header("🏭 Étude de coût (Feuille 1)")
    st.caption("Programme de représentation : coût marginal C'(x), coût moyen "
               "f(x) = C(x)/x, minimum, tableau de valeurs et graphique.")
    C = st.text_input("Coût total C(x) =", value="x**2 + 16*x + 256")
    c1, c2, c3 = st.columns(3)
    a = c1.text_input("Valeur de a (extrémité gauche)", value="5")
    b = c2.text_input("Valeur de b (extrémité droite)", value="50")
    pas = c3.text_input("Pas du tableau", value="5")
    if not C.strip():
        return
    try:
        res = etude_cout.etudier(C, a, b)
    except (ValueError, TypeError) as e:
        erreur(e)
        return

    st.subheader("Démarche détaillée")
    afficher_rapport(etude_cout.rapport_texte(res))

    st.subheader("Tableau de valeurs")
    try:
        pas_val = float(pas) if pas.strip() else None
        st.table(etude_cout.table_valeurs(res, pas=pas_val))
    except (ValueError, TypeError) as e:
        erreur(f"Tableau impossible : {e}")

    st.subheader("Graphique — coût moyen f(x) et coût marginal C'(x)")
    st.caption("Les deux courbes se croisent au minimum du coût moyen : "
               "c'est là que f(x*) = C'(x*).")
    try:
        afficher_graphe(etude_cout.figure(res), width="stretch")
        expl_etude_cout(res)
    except Exception as e:  # noqa: BLE001
        erreur(f"Tracé impossible : {e}")


def page_aide():
    st.header("❓ Aide — mode d'emploi détaillé")
    st.markdown(
        "Bienvenue ! Cette application est une **boîte à outils du cours "
        "d'Optimisation (L2 MIO, Université de Thiès)**. Elle calcule *et* "
        "explique la démarche, avec des graphiques interactifs."
    )

    st.subheader("1. Comment ça marche, en général")
    st.markdown(
        "- Choisis un **outil** dans la barre latérale (à gauche).\n"
        "- **Saisis** ta/tes fonction(s) dans les champs.\n"
        "- Le résultat s'affiche en deux temps : une **« Démarche détaillée »** "
        "(les étapes, comme à la main) puis un **graphique** interactif.\n"
        "- Les graphiques sont **interactifs** : molette = zoom, glisser = déplacer, "
        "survol = valeurs, clic sur la légende = masquer/afficher une courbe."
    )

    st.subheader("2. Syntaxe des fonctions (très important)")
    st.markdown(
        "On écrit les formules en **notation Python/SymPy** :"
    )
    st.table([
        {"Tu veux écrire": "x²", "Tu tapes": "x**2"},
        {"Tu veux écrire": "x³ − 3x", "Tu tapes": "x**3 - 3*x"},
        {"Tu veux écrire": "produit 2 fois x", "Tu tapes": "2*x  (jamais 2x)"},
        {"Tu veux écrire": "(x²−1)/(x−2)", "Tu tapes": "(x**2 - 1)/(x - 2)"},
        {"Tu veux écrire": "√x", "Tu tapes": "sqrt(x)"},
        {"Tu veux écrire": "e^x", "Tu tapes": "exp(x)"},
        {"Tu veux écrire": "ln(x)", "Tu tapes": "log(x)"},
        {"Tu veux écrire": "sin x, cos x", "Tu tapes": "sin(x), cos(x)"},
        {"Tu veux écrire": "fonction de 2 variables", "Tu tapes": "x**2 + y**2 - 3*x*y"},
    ])
    st.warning("⚠️ Toujours mettre le `*` pour multiplier : `2*x` et non `2x`, "
               "`3*x*y` et non `3xy`. Et la puissance c'est `**`, pas `^`.")

    st.subheader("3. Chaque outil, en détail")

    with st.expander("📈 Une variable f(x) — étude de fonction"):
        st.markdown(
            "**Ce que ça fait** : domaine de définition, dérivée première et "
            "points critiques (f'(x)=0), dérivée seconde et **nature** des extrema, "
            "points d'inflexion, limites et **asymptotes**.\n\n"
            "**Tu saisis** : une fonction f(x).\n\n"
            "**Exemple** : `x**3 - 3*x` → minimum en x=1, maximum en x=−1.")

    with st.expander("📏 Intervalle [a, b] — optimisation bornée"):
        st.markdown(
            "**Ce que ça fait** : trouve le **minimum et le maximum absolus** de f "
            "sur un intervalle fermé. Méthode : comparer les points critiques "
            "intérieurs **et** les deux bornes.\n\n"
            "**Tu saisis** : f(x), la valeur de a, la valeur de b.\n\n"
            "**Exemple (Feuille 2)** : `x**3 - 12*x + 9` sur [0, 3] → min=−7 (x=2), "
            "max=9 (x=0).")

    with st.expander("🧊 Deux variables f(x, y)"):
        st.markdown(
            "**Ce que ça fait** : gradient, points critiques (∇f=0), dérivées "
            "secondes **r, s, t** et nature par le test **rt−s²** (min / max / col). "
            "Affiche la **surface 3D** et les **courbes de niveau 2D**.\n\n"
            "**Tu saisis** : une fonction f(x, y).\n\n"
            "**Exemple (Examen)** : `x**3 + y**3 - 3*x*y` → col en (0,0), min en (1,1).")

    with st.expander("🔗 Sous contrainte — Lagrange & substitution"):
        st.markdown(
            "**Ce que ça fait** : optimise f(x,y) sous une contrainte. **Deux "
            "méthodes** : Lagrange (multiplicateur λ, hessienne bordée) et "
            "substitution. Le graphique montre les **courbes de niveau**, la "
            "**contrainte** et les **gradients ∇f, ∇g** (parallèles à l'optimum).\n\n"
            "**Tu saisis** : l'objectif f(x,y), et la contrainte sous la forme "
            "**g(x,y) = 0**.\n\n"
            "**Astuce contrainte** : pour « x + y = 10 », tu entres `x + y - 10` "
            "(tout passé à gauche).\n\n"
            "**Exemple** : objectif `x*y`, contrainte `x + y - 10` → max 25 en (5,5).")

    with st.expander("💰 Analyse marginale (économie)"):
        st.markdown(
            "**Ce que ça fait** : trois sous-outils — **fonction marginale** "
            "(dérivée, ex. coût de la unité suivante), **coût moyen** C(x)/x et sa "
            "minimisation, **profit** P=R−C et sa maximisation.\n\n"
            "**Exemple (Feuille 1)** : coût marginal de `20 + 500*x - x**2` en x=50.")

    with st.expander("🏭 Étude de coût (Feuille 1) — programme de représentation"):
        st.markdown(
            "**Ce que ça fait** : l'étude complète d'un coût C(x) sur [a,b] : coût "
            "marginal C'(x), coût moyen f(x)=C(x)/x, f'(x), **tableau de "
            "variations** (avec ↗↘), **tableau de valeurs**, et le **graphique** "
            "superposant f et C' avec la vérification f(x*)=C'(x*).\n\n"
            "**Tu saisis** : C(x), a, b, et le pas du tableau.\n\n"
            "**Exemple (Feuille 1, Exo 1)** : `x**2 + 16*x + 256` sur [5, 50], pas 5 "
            "→ coût moyen minimal 48 en x=16, et f(16)=C'(16)=48.")

    with st.expander("📊 Programmation linéaire (Feuilles 3 & 4)"):
        st.markdown(
            "**Ce que ça fait** : résout un programme linéaire à 2 variables par "
            "**3 méthodes** : résolution **graphique** (domaine réalisable + "
            "sommets), méthode du **simplexe** (tableaux), et **dualité** "
            "(primal → dual + solution).\n\n"
            "**Tu saisis** : l'objectif (deux coefficients, ex. `4 5`), le sens "
            "(max/min), et les contraintes **une par ligne** (ex. `2 1 <= 800`). "
            "Les conditions x1 ≥ 0 et x2 ≥ 0 sont automatiques.\n\n"
            "**Exemple (Feuille 4, Exo 1)** : objectif `4 5`, max, contraintes "
            "`2 1 <= 800`, `1 2 <= 700`, `0 1 <= 300` → optimum (300, 200), z*=2200.")

    st.subheader("4. Astuces")
    st.markdown(
        "- **Traduction Chrome** : si Chrome propose de « traduire en français », "
        "clique sur ⋮ → *Ne jamais traduire ce site* (sinon il déforme les "
        "libellés).\n"
        "- **Résultats exacts** : les valeurs sont en fractions exactes "
        "(ex. `1/2`, `sqrt(2)`), pas en décimaux approximatifs.\n"
        "- **Erreur de saisie** : si tu vois un message ⚠️ rouge, vérifie le `*` "
        "de multiplication et les parenthèses."
    )
    st.info("Bon travail ! 🎓  Chaque outil suit la même logique que ton cours, "
            "tu peux donc vérifier tes exercices étape par étape.")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
PAGES = {
    "📈 Une variable  f(x)": page_une_variable,
    "📏 Intervalle [a, b]": page_intervalle,
    "🧊 Deux variables  f(x, y)": page_deux_variables,
    "🔗 Sous contrainte": page_contrainte,
    "💰 Analyse marginale": page_marginale,
    "🏭 Étude de coût (Feuille 1)": page_etude_cout,
    "📊 Programmation linéaire": page_lineaire,
    "❓ Aide / Mode d'emploi": page_aide,
}


def main():
    # On déclare la langue UNE seule fois (sinon l'iframe recréé à chaque
    # interaction désynchronise le rendu et fige parfois l'ancien titre).
    if not st.session_state.get("_langue_ok"):
        declarer_langue_francaise()
        st.session_state["_langue_ok"] = True
    st.sidebar.title("📐 Optimisation : cours, TD et tp de Dr Gueye")
    st.sidebar.caption("Université de Thiès — L2 MIO — boîte à outils du cours")
    choix = st.sidebar.radio("Outils", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Syntaxe des fonctions : `**` puissance, `*` produit, "
        "`sin(x)`, `cos(x)`, `exp(x)`, `log(x)`, `sqrt(x)`."
    )
    PAGES[choix]()


if __name__ == "__main__":
    main()
