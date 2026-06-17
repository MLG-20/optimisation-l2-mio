"""Tracés interactifs avec Plotly (export HTML ouvrable dans le navigateur)."""

import numpy as np
import plotly.graph_objects as go
import sympy as sp


def _en_float(valeur):
    """Convertit une expression SymPy en float, ou None si impossible."""
    try:
        return float(valeur)
    except (TypeError, ValueError):
        return None


def _format_droite(pente, ordonnee):
    """Formate une droite « a·x + b » proprement (x au lieu de 1x, etc.)."""
    if pente == 1:
        terme = "x"
    elif pente == -1:
        terme = "-x"
    else:
        terme = f"{pente}·x"
    if ordonnee == 0:
        return terme
    try:
        negatif = bool(ordonnee < 0)
    except TypeError:
        negatif = False
    if negatif:
        return f"{terme} - {-ordonnee}"
    return f"{terme} + {ordonnee}"


def figure_1d(resultat, marge=4.0, nb_points=1500):
    """Construit et renvoie la figure Plotly de f(x) (extrema + asymptotes)."""
    x = resultat.variable
    expr = resultat.expression

    abscisses_crit = [
        v for v in (_en_float(pc.abscisse) for pc in resultat.points_critiques)
        if v is not None
    ]
    if abscisses_crit:
        x_min = min(abscisses_crit) - marge
        x_max = max(abscisses_crit) + marge
    else:
        x_min, x_max = -10.0, 10.0
    if x_max - x_min < 1e-6:
        x_min, x_max = x_min - 5, x_max + 5

    xs = np.linspace(x_min, x_max, nb_points)
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
    # On masque les valeurs énormes autour des asymptotes verticales.
    ys[np.abs(ys) > 1e6] = np.nan

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)",
                             line=dict(color="royalblue", width=2)))

    # Extrema
    for pc in resultat.points_critiques:
        xv = _en_float(pc.abscisse)
        yv = _en_float(pc.ordonnee)
        if xv is None or yv is None:
            continue
        couleur = "green" if "minimum" in pc.nature else (
            "red" if "maximum" in pc.nature else "orange")
        fig.add_trace(go.Scatter(
            x=[xv], y=[yv], mode="markers+text",
            marker=dict(size=10, color=couleur, symbol="circle"),
            text=[pc.nature], textposition="top center",
            name=f"{pc.nature} ({xv:.3g}; {yv:.3g})",
        ))

    # Asymptotes
    for val, _sens in resultat.asymptotes.get("horizontales", []):
        yv = _en_float(val)
        if yv is not None:
            fig.add_hline(y=yv, line_dash="dash", line_color="gray",
                          annotation_text=f"y = {val}")
    for val in resultat.asymptotes.get("verticales", []):
        xv = _en_float(val)
        if xv is not None:
            fig.add_vline(x=xv, line_dash="dot", line_color="gray",
                          annotation_text=f"x = {val}")
    for pente, ordo, _sens in resultat.asymptotes.get("obliques", []):
        a, b = _en_float(pente), _en_float(ordo)
        if a is not None and b is not None:
            fig.add_trace(go.Scatter(
                x=xs, y=a * xs + b, mode="lines",
                line=dict(dash="dash", color="gray"),
                name=f"asymptote oblique : y = {_format_droite(pente, ordo)}",
            ))

    fig.update_layout(
        title=f"f(x) = {expr}",
        xaxis_title="x", yaxis_title="f(x)",
        template="plotly_white", hovermode="x unified",
    )
    return fig


def tracer_1d(resultat, fichier="courbe_1d.html", marge=4.0, nb_points=1500):
    """Trace f(x) et exporte en HTML. Renvoie le chemin du fichier."""
    fig = figure_1d(resultat, marge=marge, nb_points=nb_points)
    fig.write_html(fichier, include_plotlyjs="cdn")
    return fichier


def figure_intervalle(resultat, nb_points=800):
    """Figure Plotly de f sur [a, b] : courbe + candidats + min/max marqués.

    Attend un ResultatIntervalle (champs : expression, variable, a, b,
    candidats, minimum, maximum).
    """
    x = resultat.variable
    expr = resultat.expression
    a, b = float(resultat.a), float(resultat.b)
    largeur = b - a
    marge = 0.05 * largeur if largeur else 1.0

    xs = np.linspace(a, b, nb_points)
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x) sur [a, b]",
                             line=dict(color="royalblue", width=2)))

    # Candidats (points critiques intérieurs + bornes), hors min/max global.
    opt = {id(resultat.minimum), id(resultat.maximum)}
    for c in resultat.candidats:
        if id(c) in opt:
            continue
        xv, yv = _en_float(c.abscisse), _en_float(c.valeur)
        if xv is None or yv is None:
            continue
        fig.add_trace(go.Scatter(
            x=[xv], y=[yv], mode="markers", marker=dict(size=9, color="gray"),
            name=f"{c.type} ({xv:.3g}; {yv:.3g})",
        ))

    # Minimum et maximum absolus mis en évidence.
    for extr, couleur, etiq in ((resultat.minimum, "green", "MIN absolu"),
                                (resultat.maximum, "red", "MAX absolu")):
        if extr is None:
            continue
        xv, yv = _en_float(extr.abscisse), _en_float(extr.valeur)
        if xv is None or yv is None:
            continue
        fig.add_trace(go.Scatter(
            x=[xv], y=[yv], mode="markers+text",
            marker=dict(size=13, color=couleur, symbol="star"),
            text=[etiq], textposition="top center",
            name=f"{etiq} ({xv:.3g}; {yv:.3g})",
        ))

    # Lignes verticales aux bornes.
    for borne in (a, b):
        fig.add_vline(x=borne, line_dash="dot", line_color="lightgray")

    fig.update_layout(
        title=f"f(x) = {expr}  sur  [{resultat.a}, {resultat.b}]",
        xaxis_title="x", yaxis_title="f(x)",
        xaxis=dict(range=[a - marge, b + marge]),
        template="plotly_white", hovermode="x unified",
    )
    return fig


def tracer_intervalle(resultat, fichier="intervalle.html", nb_points=800):
    """Trace f sur [a, b] et exporte en HTML. Renvoie le chemin du fichier."""
    fig = figure_intervalle(resultat, nb_points=nb_points)
    fig.write_html(fichier, include_plotlyjs="cdn")
    return fichier


def figure_2d(resultat, marge=3.0, nb_points=80):
    """Construit et renvoie la figure Plotly de la surface z = f(x, y)."""
    x, y = resultat.variables
    expr = resultat.expression

    xs_c = [v for v in (_en_float(pc.point[0]) for pc in resultat.points_critiques)
            if v is not None]
    ys_c = [v for v in (_en_float(pc.point[1]) for pc in resultat.points_critiques)
            if v is not None]
    x_min = (min(xs_c) - marge) if xs_c else -5.0
    x_max = (max(xs_c) + marge) if xs_c else 5.0
    y_min = (min(ys_c) - marge) if ys_c else -5.0
    y_max = (max(ys_c) + marge) if ys_c else 5.0

    xs = np.linspace(x_min, x_max, nb_points)
    ys = np.linspace(y_min, y_max, nb_points)
    X, Y = np.meshgrid(xs, ys)
    f_num = sp.lambdify((x, y), expr, "numpy")
    with np.errstate(all="ignore"):
        Z = f_num(X, Y)
    Z = np.broadcast_to(np.asarray(Z, dtype=float), X.shape)

    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z, colorscale="Viridis", opacity=0.85,
        showscale=True, name="f(x,y)",
    )])

    # Points critiques
    couleurs = {"minimum local": "green", "maximum local": "red",
                "point col (selle)": "orange"}
    for pc in resultat.points_critiques:
        xv, yv = _en_float(pc.point[0]), _en_float(pc.point[1])
        zv = _en_float(pc.valeur)
        if None in (xv, yv, zv):
            continue
        fig.add_trace(go.Scatter3d(
            x=[xv], y=[yv], z=[zv], mode="markers+text",
            marker=dict(size=6, color=couleurs.get(pc.nature, "black")),
            text=[pc.nature], name=f"{pc.nature} ({xv:.3g}; {yv:.3g})",
        ))

    fig.update_layout(
        title=f"f(x, y) = {expr}",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
        template="plotly_white",
    )
    return fig


def tracer_2d(resultat, fichier="surface_2d.html", marge=3.0, nb_points=80):
    """Trace la surface z = f(x, y) et exporte en HTML. Renvoie le chemin."""
    fig = figure_2d(resultat, marge=marge, nb_points=nb_points)
    fig.write_html(fichier, include_plotlyjs="cdn")
    return fichier


def figure_contour_2d(resultat, marge=3.0, nb_points=240):
    """Vue 2D en COURBES DE NIVEAU de f(x, y) + points critiques.

    Complément de la surface 3D : lire les niveaux aide à reconnaître
    visuellement minimum, maximum et point col.
    """
    x, y = resultat.variables
    expr = resultat.expression

    pts = []
    for pc in resultat.points_critiques:
        px, py = _en_float(pc.point[0]), _en_float(pc.point[1])
        if px is not None and py is not None:
            pts.append((px, py, pc))
    if pts:
        xs_p = [p[0] for p in pts]
        ys_p = [p[1] for p in pts]
        x_min, x_max = min(xs_p) - marge, max(xs_p) + marge
        y_min, y_max = min(ys_p) - marge, max(ys_p) + marge
    else:
        x_min, x_max, y_min, y_max = -5, 5, -5, 5
    if x_max - x_min < 1e-6:
        x_min, x_max = x_min - 3, x_max + 3
    if y_max - y_min < 1e-6:
        y_min, y_max = y_min - 3, y_max + 3

    xs = np.linspace(x_min, x_max, nb_points)
    ys = np.linspace(y_min, y_max, nb_points)
    X, Y = np.meshgrid(xs, ys)
    f_num = sp.lambdify((x, y), expr, "numpy")
    with np.errstate(all="ignore"):
        Z = np.broadcast_to(np.asarray(f_num(X, Y), dtype=float), X.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, contours_coloring="heatmap", ncontours=24,
        colorscale="Viridis", colorbar=dict(title="f(x,y)"),
        hovertemplate="x=%{x:.2f}, y=%{y:.2f}<br>f=%{z:.2f}<extra></extra>",
    ))

    for px, py, pc in pts:
        couleur = ("lime" if "minimum" in pc.nature
                   else "red" if "maximum" in pc.nature else "white")
        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers+text",
            marker=dict(size=13, color=couleur, symbol="star",
                        line=dict(color="black", width=1.5)),
            text=[pc.nature], textposition="top center",
            name=f"{pc.nature} ({px:.3g}; {py:.3g})",
        ))

    fig.update_layout(
        title=f"Courbes de niveau de f(x, y) = {expr}",
        xaxis_title="x", yaxis_title="y", template="plotly_white",
    )
    return fig


def figure_contrainte(resultat, marge=2.5, nb_points=240):
    """Figure de l'optimisation sous contrainte (vision géométrique de Lagrange).

    Affiche les COURBES DE NIVEAU de f, la CONTRAINTE g = 0, et les points
    critiques. À l'optimum, une courbe de niveau de f est tangente à g = 0.
    Attend un résultat ayant : objectif, contrainte, variables, points_critiques
    (chaque point ayant .point, .nature, .valeur).
    """
    x, y = resultat.variables
    f = resultat.objectif
    g = resultat.contrainte

    pts = []
    for pc in resultat.points_critiques:
        px, py = _en_float(pc.point[0]), _en_float(pc.point[1])
        if px is not None and py is not None:
            pts.append((px, py, pc))
    if pts:
        xs_p = [p[0] for p in pts]
        ys_p = [p[1] for p in pts]
        x_min, x_max = min(xs_p) - marge, max(xs_p) + marge
        y_min, y_max = min(ys_p) - marge, max(ys_p) + marge
    else:
        x_min, x_max, y_min, y_max = -5, 5, -5, 5
    if x_max - x_min < 1e-6:
        x_min, x_max = x_min - 3, x_max + 3
    if y_max - y_min < 1e-6:
        y_min, y_max = y_min - 3, y_max + 3

    xs = np.linspace(x_min, x_max, nb_points)
    ys = np.linspace(y_min, y_max, nb_points)
    X, Y = np.meshgrid(xs, ys)
    f_num = sp.lambdify((x, y), f, "numpy")
    g_num = sp.lambdify((x, y), g, "numpy")
    with np.errstate(all="ignore"):
        Zf = np.broadcast_to(np.asarray(f_num(X, Y), dtype=float), X.shape)
        Zg = np.broadcast_to(np.asarray(g_num(X, Y), dtype=float), X.shape)

    fig = go.Figure()

    # Courbes de niveau de f.
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Zf, contours_coloring="lines", ncontours=18,
        colorscale="Viridis", showscale=True, line_width=1,
        name="courbes de niveau de f",
        colorbar=dict(title="f(x,y)"),
        hovertemplate="x=%{x:.2f}, y=%{y:.2f}<br>f=%{z:.2f}<extra></extra>",
    ))

    # Contrainte g = 0 (ligne rouge épaisse).
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Zg, contours=dict(start=0, end=0, size=1, coloring="lines"),
        line=dict(width=4), colorscale=[[0, "red"], [1, "red"]], showscale=False,
        name="contrainte g = 0", hoverinfo="skip",
    ))

    # Points critiques.
    for px, py, pc in pts:
        couleur = ("green" if "minimum" in pc.nature
                   else "red" if "maximum" in pc.nature else "orange")
        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers+text",
            marker=dict(size=13, color=couleur, symbol="star",
                        line=dict(color="black", width=1)),
            text=[f"{pc.nature} (f={pc.valeur})"], textposition="top center",
            name=f"({px:.3g}; {py:.3g})",
        ))

    # Flèches des gradients ∇f (bleu) et ∇g (rouge) : à l'optimum ils sont
    # colinéaires (∇f = λ·∇g) — c'est la condition de Lagrange, rendue visible.
    fx, fy = sp.diff(f, x), sp.diff(f, y)
    gx, gy = sp.diff(g, x), sp.diff(g, y)
    echelle = 0.18 * min(x_max - x_min, y_max - y_min)

    def fleche(px, py, vx, vy, couleur):
        norme = (vx ** 2 + vy ** 2) ** 0.5
        if norme < 1e-12:
            return
        ux, uy = vx / norme * echelle, vy / norme * echelle
        fig.add_annotation(
            x=px + ux, y=py + uy, ax=px, ay=py,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.5,
            arrowcolor=couleur,
        )

    for px, py, pc in pts:
        subs = {x: pc.point[0], y: pc.point[1]}
        vfx, vfy = _en_float(fx.subs(subs)), _en_float(fy.subs(subs))
        vgx, vgy = _en_float(gx.subs(subs)), _en_float(gy.subs(subs))
        if None not in (vfx, vfy):
            fleche(px, py, vfx, vfy, "blue")
        if None not in (vgx, vgy):
            fleche(px, py, vgx, vgy, "crimson")

    # Entrées de légende pour les flèches.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="blue", width=3), name="∇f (gradient de f)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="crimson", width=3),
                             name="∇g (gradient de la contrainte)"))

    fig.update_layout(
        title=f"Niveaux de f = {f}   &   contrainte  {g} = 0",
        xaxis_title="x", yaxis_title="y",
        template="plotly_white",
    )
    return fig


def tracer_contrainte(resultat, fichier="contrainte.html"):
    """Trace la vue géométrique de Lagrange et exporte en HTML."""
    fig = figure_contrainte(resultat)
    fig.write_html(fichier, include_plotlyjs="cdn")
    return fichier
