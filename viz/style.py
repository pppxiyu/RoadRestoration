"""
Shared, journal-quality figure styling for every plot the project produces
(toy-data experiments, traffic-equilibrium validation, and every solver's figures).
Centralizing the fonts, colors, and sizing here is the single source of truth:
because all figures pull from the same settings, they read as one coherent set
rather than a patchwork of matplotlib defaults.

Typical use: apply the shared style once at the start, build figures as usual,
then save each one through the helper so every file lands at the same resolution
and format.

Usage:
    from viz.style import use_pub, save_pub, panel_label, C, severity_color, roadclass_color
    use_pub()                         # apply the shared style once, up front
    ...
    save_pub(fig, dir / "01_name")    # write a 600-dpi PNG (optionally SVG/PDF too)
"""

import matplotlib as mpl

# Global matplotlib settings (rcParams is matplotlib's runtime configuration
# dictionary) that every figure inherits. The goal is a compact, restrained
# journal look: small type, thin axis lines, no top/right borders, and vector
# text that stays editable after export rather than being flattened to curves.
PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "svg.fonttype": "none",   # store labels as real text nodes, so they can be re-typed in a vector editor
    "pdf.fonttype": 42,       # embed a TrueType font in the PDF so its text stays selectable and editable
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "figure.dpi": 130,
}

# A single named color palette shared across the project, so a given meaning
# always maps to the same color. Greys carry non-focal context; the blues are
# the main emphasis colors; the warm "signal" red marks the key result (an
# optimum, a highlighted item, or a drop in a metric); green marks improvements.
C = {
    "neutral_light": "#CFCECE",
    "neutral_mid": "#8A8A8A",
    "neutral_dark": "#4D4D4D",
    "accent": "#0F4D92",     # primary emphasis (dark blue)
    "accent2": "#3775BA",    # secondary emphasis (lighter blue)
    "teal": "#42949E",       # a third, distinct accent
    "signal": "#B64342",     # the standout result: optimum, highlight, or a drop
    "good": "#2E9E44",       # improvements or gains
    "purple": "#7B5FA3",     # a fourth distinct accent (used by the stochastically-trained RL)
}

# Damage severity levels 1/2/3 map to an increasingly dark warm ramp, so a
# darker color reads immediately as more severe damage.
SEVERITY = {1: "#F6CFCB", 2: "#E59A93", 3: "#B64342"}
# Road classes ordered local < major < highway map to a darkening blue ramp, so
# a darker color reads as a higher-capacity, higher-class road.
ROADCLASS = {"local": "#B4C0E4", "major": "#6B7BB0", "highway": "#2E3A6E"}

# Continuous colormaps for quantities plotted as a shaded field. Each is fixed to
# one kind of quantity so the same meaning always uses the same color scale:
CMAP_SEQ = "cividis"   # road capacity or traffic-flow magnitude
CMAP_DEMAND = "magma"  # origin-destination travel demand (trips between location pairs), as a heatmap
CMAP_ERR = "Reds"      # magnitude of error between two quantities


def use_pub(slide=False):
    """Push the shared publication settings into matplotlib's global config.

    Call this once before building any figures; every figure created afterward
    inherits the consistent fonts, sizes, and spine styling.

    slide=True scales every font size up (~1.7x) for figures whose primary home
    is a slide rather than a manuscript page: same family, same palette, same
    spine styling, just legible from the back of a room. Figures built at slide
    scale should also use a proportionally larger figsize, so the type-to-canvas
    ratio stays similar to the print version.
    """
    rc = dict(PUB_RC)
    if slide:
        k = 1.7
        for key in ("font.size", "axes.titlesize", "axes.labelsize",
                    "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
            rc[key] = round(PUB_RC[key] * k, 1)
        rc["axes.linewidth"] = 1.1
        rc["xtick.major.width"] = 1.1
        rc["ytick.major.width"] = 1.1
        rc["lines.linewidth"] = 2.0
    mpl.rcParams.update(rc)


def severity_color(s):
    # Look up the palette color for a damage-severity level; the int() coercion
    # lets callers pass the level as a float or string (1, 2, or 3).
    return SEVERITY[int(s)]


def roadclass_color(rc):
    # Look up the palette color for a road class; the str() coercion lets callers
    # pass a non-string key ("local", "major", or "highway").
    return ROADCLASS[str(rc)]


def panel_label(ax, label, x=-0.12, y=1.04, size=9):
    """Draw a bold subplot letter (e.g. "a", "b") just outside an axes' top-left.

    Multi-panel figures label each subplot with a letter for reference in the
    text. Positions are given in axes-fraction coordinates (0-1 spans the axes),
    so the letter stays put regardless of the data range.
    """
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size,
            fontweight="bold", ha="left", va="bottom")


def save_pub(fig, path_stem, dpi=600, svg=False, pdf=False):
    """Save a figure as a 600-dpi PNG, and optionally as editable vector files.

    Set svg=True or pdf=True to also emit those formats for later editing in a
    vector tool; by default only the PNG is written. `path_stem` is the file path
    without an extension, which each format's suffix is appended to. Tight bounding
    boxes trim surrounding whitespace so the saved image is cropped to the content.
    """
    path_stem = str(path_stem)
    fig.savefig(path_stem + ".png", dpi=dpi, bbox_inches="tight")
    if svg:
        fig.savefig(path_stem + ".svg", bbox_inches="tight")
    if pdf:
        fig.savefig(path_stem + ".pdf", bbox_inches="tight")
