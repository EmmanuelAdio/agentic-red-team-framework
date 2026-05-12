"""Colour palette and shared matplotlib style for Chapter-6 figures.

The dissertation uses the Okabe-Ito 8-colour palette throughout. It was
designed to be distinguishable for the most common forms of colour-blindness
(deuteranopia, protanopia, and tritanopia) and prints legibly in greyscale
- two properties the default ``matplotlib`` palette does not guarantee.
The palette is reproduced here verbatim from Okabe and Ito's original
specification, with hex codes used as the source of truth so the same
swatches drive both ``matplotlib`` and any pandas Styler highlighting in
the notebook.

Cell-colour mapping
-------------------

The four experiment cells get fixed colours so every Chapter-6 figure can
be read against the same legend without the reader retraining their eye:

================ ======================== ==============
cell label       Okabe-Ito name           hex
================ ======================== ==============
``ipi``          orange                   #E69F00
``poiA``         sky blue                 #56B4E9
``poiJ``         vermillion               #D55E00
``qInj``         bluish green             #009E73
``clean``        grey (neutral baseline)  #999999
================ ======================== ==============

Why these specific cells got these specific hues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distinctness of *adjacent* colours matters more than absolute pleasing-ness,
so the assignment puts the two corpus-poisoning cells (``poiA`` and
``poiJ``) on the opposite ends of the warm/cool axis (sky-blue vs
vermillion) - they show up together most often in side-by-side bars and
need to be the *easiest* pair to tell apart at a glance. ``ipi`` and
``qInj`` are the two cells we want to compare against in pairwise plots
(both are integrity attacks, one corpus-channel, one query-channel) and
they get the second-most-distinct pair (orange vs bluish-green).

References
----------

- Okabe, M., & Ito, K. (2008). *Color Universal Design (CUD): How to make
  figures and presentations that are friendly to colorblind people.*
  Tokyo University of Agriculture and Technology.
- https://jfly.uni-koeln.de/color/ (canonical mirror).
"""

from __future__ import annotations

from typing import Final

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Okabe-Ito 8-colour palette
# ---------------------------------------------------------------------------

OKABE_ITO: Final[dict[str, str]] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# ---------------------------------------------------------------------------
# Cell-colour mapping
# ---------------------------------------------------------------------------

CELL_COLOURS: Final[dict[str, str]] = {
    "ipi":   OKABE_ITO["orange"],
    "poiA":  OKABE_ITO["sky_blue"],
    "poiJ":  OKABE_ITO["vermillion"],
    "qInj":  OKABE_ITO["bluish_green"],
    "clean": "#999999",  # neutral grey for the baseline overlay
}

# Order cells appear in every figure legend - kept fixed so the reader's
# eye is trained the same way across the chapter.
CELL_ORDER: Final[tuple[str, ...]] = ("ipi", "poiA", "poiJ", "qInj")

# Human-readable display labels for axis tick labels.
CELL_DISPLAY: Final[dict[str, str]] = {
    "ipi":   "IPI\n(prompt inj.)",
    "poiA":  "PoiA\n(answer repl.)",
    "poiJ":  "PoiJ\n(jamming)",
    "qInj":  "QInj\n(query inj.)",
    "clean": "Clean\nbaseline",
}

# ASR metric colours used in figures that *aren't* per-cell - e.g. the F1
# grouped-bar of ASR-r / ASR-a / ASR-t, where the three bars per cell-group
# need their own distinguishable colours.
ASR_COLOURS: Final[dict[str, str]] = {
    "asr_retrieval": OKABE_ITO["blue"],
    "asr_answer":    OKABE_ITO["yellow"],
    "asr_target":    OKABE_ITO["reddish_purple"],
}

# RAGAS metric colours for figures that compare the three RAGAS scorers.
RAGAS_COLOURS: Final[dict[str, str]] = {
    "ragas_faithfulness":      OKABE_ITO["blue"],
    "ragas_answer_relevance":  OKABE_ITO["orange"],
    "ragas_context_relevance": OKABE_ITO["bluish_green"],
}


# ---------------------------------------------------------------------------
# Shared default matplotlib style
# ---------------------------------------------------------------------------


def apply_default_style() -> None:
    """Configure matplotlib for dissertation-grade figure output.

    Called by every plot function at entry. Idempotent. Sets:

    - serif font (Times-like) to match the LaTeX dissertation body text.
    - 10pt baseline font with the same hierarchy used in the chapter
      (axis labels 11, title 12, legend 10).
    - figure DPI=300 so PNG previews render crisply in the notebook and
      in any embedded preview the dissertation tooling generates.
    - a subtle gridline style (light grey, dashed, behind the data) that
      makes value-reading easier without competing with the data.
    - tight bounding boxes when saving so figures don't develop unwanted
      whitespace margins after multiple ``savefig`` calls.

    Why this lives in a function and not as module-level side effects:
    importing the palette module should not silently mutate the user's
    matplotlib rcParams (a problem the previous Day-10 module had). The
    explicit call site makes the dependency visible to anyone reading
    the plot code.
    """
    mpl.rcParams.update({
        # Typography
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "font.size":         10.0,
        "axes.titlesize":    12.0,
        "axes.labelsize":    11.0,
        "xtick.labelsize":   9.5,
        "ytick.labelsize":   9.5,
        "legend.fontsize":   9.5,
        "figure.titlesize":  13.0,

        # Output quality
        "figure.dpi":        110.0,
        "savefig.dpi":       300.0,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,

        # Axes / grid
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         "#CCCCCC",
        "grid.linestyle":     "--",
        "grid.linewidth":     0.6,
        "grid.alpha":         0.7,

        # Lines / markers
        "lines.linewidth":    1.8,
        "lines.markersize":   5.5,

        # Legend
        "legend.frameon":     False,
        "legend.borderaxespad": 0.3,

        # Default cycle: Okabe-Ito (skip black; reserve it for emphasis).
        "axes.prop_cycle": mpl.cycler(
            color=[
                OKABE_ITO["orange"],
                OKABE_ITO["sky_blue"],
                OKABE_ITO["bluish_green"],
                OKABE_ITO["vermillion"],
                OKABE_ITO["blue"],
                OKABE_ITO["reddish_purple"],
                OKABE_ITO["yellow"],
            ]
        ),
    })
