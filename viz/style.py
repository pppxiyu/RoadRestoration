"""
Shared Nature-style publication styling for ALL project figures (toy-data, UE
validation, and oracle). One source of truth so every figure looks like one set.

Usage:
    from viz.style import use_pub, save_pub, panel_label, C, severity_color, roadclass_color
    use_pub()                         # call once before creating figures
    ...
    save_pub(fig, dir / "01_name")    # writes .png(600) + .svg + .pdf
"""

import matplotlib as mpl

# --- rcParams: editable vector text, restrained spines, compact journal sizing ---
PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "svg.fonttype": "none",   # keep text as <text> nodes (editable in Illustrator/Inkscape)
    "pdf.fonttype": 42,       # editable TrueType in PDF
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

# --- one restrained palette for the whole project ---
C = {
    "neutral_light": "#CFCECE",
    "neutral_mid": "#8A8A8A",
    "neutral_dark": "#4D4D4D",
    "accent": "#0F4D92",     # primary (blue)
    "accent2": "#3775BA",    # secondary blue
    "teal": "#42949E",       # secondary accent
    "signal": "#B64342",     # optimum / highlight / drop
    "good": "#2E9E44",       # gains
}

# severity 1/2/3 → sequential warm ramp (darker = worse)
SEVERITY = {1: "#F6CFCB", 2: "#E59A93", 3: "#B64342"}
# road class local<major<highway → sequential blue ramp (darker = higher class)
ROADCLASS = {"local": "#B4C0E4", "major": "#6B7BB0", "highway": "#2E3A6E"}

# sequential colormaps
CMAP_SEQ = "cividis"   # capacity / flow magnitude
CMAP_DEMAND = "magma"  # OD demand heatmap
CMAP_ERR = "Reds"      # error magnitude


def use_pub():
    """Apply the publication rcParams globally (call once per process)."""
    mpl.rcParams.update(PUB_RC)


def severity_color(s):
    return SEVERITY[int(s)]


def roadclass_color(rc):
    return ROADCLASS[str(rc)]


def panel_label(ax, label, x=-0.12, y=1.04, size=9):
    """Nature-style bold panel letter at the top-left of an axes."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size,
            fontweight="bold", ha="left", va="bottom")


def save_pub(fig, path_stem, dpi=600, svg=False, pdf=False):
    """Save a figure as PNG (600 dpi). Pass svg=True / pdf=True to also write editable vector
    versions; default is PNG-only. `path_stem` has no extension."""
    path_stem = str(path_stem)
    fig.savefig(path_stem + ".png", dpi=dpi, bbox_inches="tight")
    if svg:
        fig.savefig(path_stem + ".svg", bbox_inches="tight")
    if pdf:
        fig.savefig(path_stem + ".pdf", bbox_inches="tight")
