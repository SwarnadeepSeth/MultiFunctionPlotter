#!/usr/bin/env python3
"""
mfp_help.py — MFP Manual Page
==============================
Import and call ``show()`` to print the full manual, or call individual
section printers for targeted help.

Usage from CLI:
    python mfp_help.py               # full manual
    python mfp_help.py styles        # just the styles section
    python mfp_help.py errorbars
    python mfp_help.py colormap
    python mfp_help.py 2d
    python mfp_help.py logscale
    python mfp_help.py subplots
    python mfp_help.py save
"""

# ── ANSI colours (gracefully disabled on Windows / dumb terminals) ────────────
import os, sys

_NO_COLOR = not sys.stdout.isatty() or os.name == "nt"

def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

H1  = lambda t: _c("1;36",  t)   # bold cyan  — top-level heading
H2  = lambda t: _c("1;33",  t)   # bold yellow — section heading
KW  = lambda t: _c("1;32",  t)   # bold green  — keyword / token
EX  = lambda t: _c("90",    t)   # dark grey   — example line
NOTE= lambda t: _c("35",    t)   # magenta     — note / tip


# ══════════════════════════════════════════════════════════════════════════════
# Section printers
# ══════════════════════════════════════════════════════════════════════════════

def _header() -> None:
    print(H1("""
╔══════════════════════════════════════════════════════════════════════════╗
║          MFP — MultiFunctionPlotter  •  Manual / Reference               ║
║              Developed by Swarnadeep Seth  •  v1.0                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""))


def _synopsis() -> None:
    print(H2("SYNOPSIS"))
    print(f"""
  {KW('python mfp.py')} <command> [, <command> ...]  [--flags]

  Commands are separated by commas.  Each command describes one dataset
  or function to overlay on the same axes (or one subplot panel).

  Special subcommands:
    {KW('plot.json')}          Replay the last session from the saved JSON config.
    {KW('forecast')}           Run the Prophet time-series forecast helper.
    {KW('DM')}                 Launch the interactive Data Manipulator.
    {KW('--help')}             Print this manual.
    {KW('--list-styles')}      List every recognised plot style keyword.
""")


def _basic_tokens() -> None:
    print(H2("BASIC COMMAND TOKENS"))
    rows = [
        ("FILE",             "<name>.csv / .txt / .dat",
         "Data file to read."),
        ("using / u",        "X:Y",
         "Column indices (1-based for text files, 0-based for CSV)."),
        ("with / w",         "<style>",
         "Plot style — see STYLES section."),
        ("title",            '"My Title"',
         "Plot / panel title."),
        ("xlabel",           '"X label"',
         "Horizontal axis label."),
        ("ylabel",           '"Y label"',
         "Vertical axis label."),
        ("legend / lg",      "<label>",
         "Legend entry for this series (no spaces)."),
        ("linewidth / lw",   "<int>",
         "Line width in points (default 2)."),
        ("linecolor / lc",   "<color>",
         "Any matplotlib color string, e.g. tab:red, #3a7ab3."),
        ("xrange",           "min:max",
         "Force x-axis limits."),
        ("yrange",           "min:max",
         "Force y-axis limits."),
        ("bin",              "<int>",
         "Histogram bin count (default 'auto')."),
        ("func:",            '"f(x,a=1) = a*np.sin(x)"',
         "Plot a math expression.  Requires xrange."),
    ]
    col_w = [16, 20, 42]
    sep = "  "
    header = sep.join([
        KW("Token".ljust(col_w[0])),
        KW("Value".ljust(col_w[1])),
        KW("Description"),
    ])
    print("  " + header)
    print("  " + "─" * (sum(col_w) + len(sep) * 2))
    for tok, val, desc in rows:
        print("  " + sep.join([
            tok.ljust(col_w[0]),
            val.ljust(col_w[1]),
            desc,
        ]))
    print()


def _styles() -> None:
    print(H2("STYLES  (with / w)"))
    styles = [
        # ── Line/scatter ──────────────────────────────────────────────────────
        ("lines   / l",       "Solid line                      (matplotlib)"),
        ("dashed",            "Dashed line                     (matplotlib)"),
        ("dotted",            "Dotted line                     (matplotlib)"),
        ("points  / p",       "Circle markers only             (matplotlib)"),
        ("linespoints / lp",  "Line + circle markers           (matplotlib)"),
        ("stars",             "Star markers                    (matplotlib)"),
        ("d",                 "Diamond markers                 (matplotlib)"),
        # ── Error bars ────────────────────────────────────────────────────────
        ("errorbars / eb",    "Discrete error bars  —  needs  yerr <col>"),
        ("errorshade / es",   "Shaded ±σ band       —  needs  yerr <col>"),
        # ── Colormap scatter ──────────────────────────────────────────────────
        ("scatter",           "Scatter coloured by 3rd column  —  needs  cmap <col> [colormap <name>]"),
        # ── 2-D / matrix ─────────────────────────────────────────────────────
        ("heatmap",           "2-D heatmap (imshow)  —  needs  2D file"),
        ("contour",           "Contour lines         —  needs  2D file"),
        ("contourf",          "Filled contours       —  needs  2D file"),
        # ── Seaborn distribution ──────────────────────────────────────────────
        ("hist",              "Histogram               (seaborn histplot)"),
        ("kde",               "KDE density curve       (seaborn kdeplot)"),
        ("box",               "Box-and-whisker plot    (seaborn boxplot)"),
        ("violin",            "Violin plot             (seaborn violinplot)"),
    ]
    for name, desc in styles:
        print(f"    {KW(name.ljust(22))}  {desc}")
    print()


def _errorbars() -> None:
    print(H2("ERROR BARS"))
    print(f"""
  Two styles are available:

  {KW('errorbars / eb')}
      Plots discrete error bars using matplotlib errorbar().
      The error column is a separate column in your data file.

      Extra tokens:
        {KW('yerr <col>')}          1-based column index holding the ±σ values.
        {KW('capsize <int>')}       Cap width in points (default 4).

      Example:
        {EX('python mfp.py data.csv using 1:2 with errorbars yerr 3 lc tab:red legend "my data"')}

  {KW('errorshade / es')}
      Fills a translucent band of ±σ around the mean line.
      Same tokens as errorbars; capsize is ignored.

      Example:
        {EX('python mfp.py data.csv using 1:2 with errorshade yerr 3 lc steelblue legend "mean±σ"')}

  {NOTE('Tip: combine errorshade with a lines series in the same command')}
  {NOTE('to show the solid mean line on top of the shaded band:')}
        {EX('python mfp.py "data.csv using 1:2 with errorshade yerr 3 lc steelblue, data.csv using 1:2 with lines lc steelblue"')}
""")


def _logscale() -> None:
    print(H2("LOG SCALE"))
    print(f"""
  Append one or both flags to any plot command string:

    {KW('--xlog')}    Set the x-axis to logarithmic scale.
    {KW('--ylog')}    Set the y-axis to logarithmic scale.

  These are global flags — they apply to the entire figure / all panels.

  Example:
    {EX('python mfp.py spectrum.csv using 1:2 with lines --ylog')}
    {EX('python mfp.py spectrum.csv using 1:2 with lines --xlog --ylog')}

  {NOTE('Tip: log scale is applied after all series are drawn,')}
  {NOTE('so it works with subplots, errorbar, and function plots too.')}
""")


def _colormap() -> None:
    print(H2("COLORMAP SCATTER"))
    print(f"""
  Style: {KW('scatter')}

  Plots x vs y with each point coloured by the value in a third column.
  A colorbar is added automatically.

  Extra tokens:
    {KW('cmap <col>')}           1-based column index for the colour values.
    {KW('colormap <name>')}      Matplotlib colormap name (default: viridis).
    {KW('cbar_label "<text>"')}  Label for the colorbar (optional).

  Example:
    {EX('python mfp.py results.csv using 1:2 with scatter cmap 3 colormap plasma cbar_label "Temperature (K)"')}

  Useful colormaps:
    viridis, plasma, inferno, magma, cividis   (perceptually uniform)
    coolwarm, RdBu, seismic                    (diverging)
    Blues, Reds, YlOrRd                        (sequential)

  {NOTE('Run  python -c "import matplotlib; print(matplotlib.colormaps)"')}
  {NOTE('for the full list.')}
""")


def _2d() -> None:
    print(H2("2-D PLOT STYLES  (heatmap / contour / contourf)"))
    print(f"""
  These styles read the entire file as a 2-D numeric matrix (no header).
  Rows → y-axis,  columns → x-axis.

  Styles:
    {KW('heatmap')}     imshow with a colorbar — best for raster data.
    {KW('contour')}     Contour lines only.
    {KW('contourf')}    Filled contours.

  Extra tokens:
    {KW('colormap <name>')}      Colormap for the fill / image (default: viridis).
    {KW('levels <int>')}         Number of contour levels (default 10, contour/f only).
    {KW('cbar_label "<text>"')}  Colorbar label.

  The {KW('using')} / {KW('xrange')} / {KW('yrange')} tokens are ignored for 2-D plots
  because the entire matrix is rendered.

  Example:
    {EX('python mfp.py matrix.dat with heatmap colormap inferno cbar_label "Intensity"')}
    {EX('python mfp.py matrix.dat with contourf levels 20 colormap RdBu')}
""")


def _subplots() -> None:
    print(H2("SUBPLOTS"))
    print(f"""
  Use the {KW('--subplot')} flag followed by a layout string.
  The layout uses letters — each letter is a panel key.

  Row separator: use {KW('-')} between rows (MFP converts it to a newline).

  One command per panel, separated by commas (left-to-right, top-to-bottom).

  Example — 2×2 grid:
    {EX('python mfp.py --subplot AB-CD "data1.csv using 1:2 with lines, data2.csv using 1:2 with hist, data3.csv using 1:2 with kde, data4.csv using 1:2 with scatter cmap 3"')}

  Example — asymmetric (A spans the top row, B and C share the bottom):
    {EX('python mfp.py --subplot AA-BC ...')}
""")


def _save_section() -> None:
    print(H2("SAVING FIGURES"))
    print(f"""
  Append {KW('--save <path>')} to write the figure to disk.
  The format is inferred from the file extension.

  Supported formats:
    .png   Raster — good for screen / web.
    .pdf   Vector — best for LaTeX / publications.
    .svg   Vector — good for Inkscape / web.
    .eps   Vector — legacy journals.

  Example:
    {EX('python mfp.py data.csv using 1:2 with lines --save figure.pdf')}

  {NOTE('Tip: append --dpi 300 (future flag) for high-resolution raster output.')}
""")


def _axis_formatting() -> None:
    print(H2("ADVANCED AXIS FORMATTING"))
    print(f"""
  Control axis appearance with precision for publication-quality plots.

  Tokens:
    {KW('sci_notation <axis>')}      Enable scientific notation.
                           <axis> = x, y, or both (default: off).
    {KW('xticks "x0,x1,x2,..."')}    Set custom x-axis tick positions.
                           Use comma-separated numbers.
    {KW('yticks "y0,y1,y2,..."')}    Set custom y-axis tick positions.
    {KW('xtick_rotation <angle>')}   Rotate x-axis labels by angle in degrees.
                           Use negative angles for opposite rotation.
    {KW('ytick_rotation <angle>')}   Rotate y-axis labels by angle in degrees.
    {KW('date_format "<fmt>"')}      Parse and format x-axis as dates.
                           Format codes: %Y (year), %m (month), %d (day),
                           %H (hour), %M (min), %S (sec).
                           Example: "%Y-%m-%d" for 2024-01-15.

  Examples:
    {EX('python mfp.py data.csv using 1:2 with lines sci_notation both')}
    {EX('python mfp.py spectrum.csv using 1:2 with lines sci_notation y')}
    {EX('python mfp.py angles.csv using 1:2 with points xticks "0,90,180,270" xtick_rotation 45')}
    {EX('python mfp.py timeseries.csv using 1:2 with lines date_format "%Y-%m-%d"')}
    {EX('python mfp.py matrix.csv using 1:2 with scatter cmap 3 ytick_rotation 90')}

  {NOTE('Tip: date_format auto-rotates x labels 45° for readability.')}
  {NOTE('Combine xticks with xtick_rotation for precise control.')}
""")


def _examples() -> None:
    print(H2("QUICK EXAMPLES"))
    print(f"""
  Basic line plot:
    {EX('python mfp.py data.csv using 1:2 with lines lc tab:blue legend "Series A"')}

  Two overlaid series:
    {EX('python mfp.py "data.csv using 1:2 with lines, data.csv using 1:3 with dashed lc tab:red"')}

  Function plot:
    {EX('python mfp.py func: "f(x,a=1) = a*np.sin(x)" xrange 0:10 lc tab:green')}

  Error bars (discrete):
    {EX('python mfp.py data.csv using 1:2 with errorbars yerr 3')}

  Shaded error band:
    {EX('python mfp.py data.csv using 1:2 with errorshade yerr 3 lc steelblue')}

  Colormap scatter:
    {EX('python mfp.py results.csv using 1:2 with scatter cmap 3 colormap plasma')}

  Heatmap from matrix file:
    {EX('python mfp.py matrix.dat with heatmap colormap viridis cbar_label "Value"')}

  Filled contour:
    {EX('python mfp.py matrix.dat with contourf levels 15 colormap RdBu')}

  Log-scale spectrum:
    {EX('python mfp.py spectrum.csv using 1:2 with lines --xlog --ylog')}

  Histogram:
    {EX('python mfp.py samples.csv using 0:1 with hist bin 30 lc tab:orange')}

  2-panel subplot:
    {EX('python mfp.py --subplot AB "data.csv using 1:2 with lines, data.csv using 1:2 with hist"')}

  Save as PDF:
    {EX('python mfp.py data.csv using 1:2 with lines --save output.pdf')}

  Replay last session:
    {EX('python mfp.py plot.json')}
""")


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS: dict[str, callable] = {
    "synopsis":   _synopsis,
    "tokens":     _basic_tokens,
    "styles":     _styles,
    "errorbars":  _errorbars,
    "logscale":   _logscale,
    "colormap":   _colormap,
    "2d":         _2d,
    "subplots":   _subplots,
    "save":       _save_section,
    "formatting": _axis_formatting,
    "examples":   _examples,
}


def show(section: str = "all") -> None:
    """Print the MFP manual.

    Args:
        section: One of the keys in ``_SECTIONS``, or ``"all"`` for the full
                 manual.
    """
    _header()
    if section == "all":
        for fn in _SECTIONS.values():
            fn()
    elif section in _SECTIONS:
        _SECTIONS[section]()
    else:
        print(f"Unknown section '{section}'.  Available: {', '.join(_SECTIONS)}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    section = sys.argv[1] if len(sys.argv) > 1 else "all"
    show(section)
