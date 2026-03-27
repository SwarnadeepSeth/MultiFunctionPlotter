#!/usr/bin/env python3
"""
MultiFunctionPlotter (MFP) — A versatile tool for data visualization.

Developed by Swarnadeep Seth.
Version: 1.1.0
Date: June 19, 2024

Description:
    MultiFunctionPlotter (MFP) simplifies the creation of a wide range of plots
    from CSV and text files, as well as custom mathematical functions. With support
    for multiple plot styles and easy-to-use command-line configuration, MFP aims
    to be a versatile and powerful tool for data visualization.

New in v1.1:
    • Error bars  — discrete bars (errorbars/eb) or shaded band (errorshade/es)
    • Log scale   — --xlog / --ylog global flags
    • Colormap scatter — scatter with  cmap <col>  and  colormap <n>
    • 2-D styles  — heatmap, contour, contourf read an entire matrix file
    • Help system — --help / --list-styles  (backed by mfp_help.py)

Usage:
    python mfp.py --help
    python mfp.py --list-styles
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Project paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.style.use("custom_style")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_FIGSIZE   = (12, 6)
JSON_OUTPUT_PATH  = Path("plot.json")

#: Maps gnuplot-style style names → matplotlib linestyle/marker strings.
STYLE_MAP: dict[str, str] = {
    "lines":       "-",
    "l":           "-",
    "dashed":      "--",
    "dotted":      ":",
    "points":      "o",
    "p":           "o",
    "linespoints": "-o",
    "lp":          "-o",
    "stars":       "*",
    "d":           "d",
}

#: Seaborn distribution / summary styles (single column, no y needed).
SEABORN_STYLES = {"hist", "kde", "box", "violin"}

#: Styles that draw error bars / bands around y_data.
ERRORBAR_STYLES = {"errorbars", "eb", "errorshade", "es"}

#: Styles that map a third column to colour.
COLORMAP_STYLES = {"scatter"}

#: Styles that read the whole file as a 2-D matrix.
TWO_D_STYLES = {"heatmap", "contour", "contourf"}


# ══════════════════════════════════════════════════════════════════════════════
# PlotConfig — single source of truth for all plot parameters
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlotConfig:
    """Holds every configurable parameter for a single plot command.

    Adding a new parameter: add it here once, then update CommandParser
    and JsonParser.
    """

    # ── Data source ───────────────────────────────────────────────────────────
    file: Optional[str]   = None
    x_col: Optional[int]  = None
    y_col: Optional[int]  = None

    # ── Math function (alternative to file) ──────────────────────────────────
    function: Optional[str]  = None
    func_parameters: dict    = field(default_factory=dict)

    # ── Appearance ────────────────────────────────────────────────────────────
    style: str                = "lines"
    linewidth: int            = 2
    linecolor: Optional[str]  = None

    # ── Labels ────────────────────────────────────────────────────────────────
    title: Optional[str]  = None
    xlabel: str           = "X-axis"
    ylabel: str           = "Y-axis"
    legend: Optional[str] = None

    # ── Font sizes ────────────────────────────────────────────────────────────
    title_font_size: int = 20
    axis_font_size: int  = 18
    tick_font_size: int  = 14

    # ── Axes ranges & histogram bins ─────────────────────────────────────────
    xrange: Optional[list[int]] = None
    yrange: Optional[list[int]] = None
    bin: Union[int, str]        = "auto"

    # ── Error bars ────────────────────────────────────────────────────────────
    #: 1-based column index holding the ±σ (or ±error) values.
    yerr_col: Optional[int]  = None
    #: Cap width in points for discrete error bars (errorbars style).
    capsize: int             = 4

    # ── Colormap scatter ─────────────────────────────────────────────────────
    #: 1-based column index whose values drive the point colour.
    cmap_col: Optional[int]   = None
    #: Matplotlib colormap name for scatter / 2-D plots.
    colormap: str              = "viridis"
    #: Label for the colorbar (scatter and 2-D styles).
    cbar_label: Optional[str]  = None

    # ── 2-D / matrix styles ───────────────────────────────────────────────────
    #: Number of contour levels (contour / contourf).
    levels: int = 10

    # ── Advanced axis formatting ──────────────────────────────────────────────
    #: Date/time parsing: "auto", "%Y-%m-%d", "%d/%m/%Y", etc.
    date_format: Optional[str]  = None
    #: Scientific notation for axes: "x", "y", or "both".
    sci_notation: Optional[str] = None
    #: Custom x-axis tick positions: "0,90,180,270" or similar.
    xticks: Optional[str]       = None
    #: Custom y-axis tick positions.
    yticks: Optional[str]       = None
    #: Rotate x-axis labels: angle in degrees (e.g., 45).
    xtick_rotation: int         = 0
    #: Rotate y-axis labels: angle in degrees.
    ytick_rotation: int         = 0

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def mpl_style(self) -> str:
        """Return the matplotlib linestyle/marker string for this config."""
        return STYLE_MAP.get(self.style, "-")


# ══════════════════════════════════════════════════════════════════════════════
# Parsers — convert raw input into a PlotConfig
# ══════════════════════════════════════════════════════════════════════════════

class CommandParser:
    """Parse a gnuplot-style command string into a PlotConfig."""

    # Pre-compiled regex patterns
    _RE_CSV        = re.compile(r"(\S+\.csv)")
    _RE_TEXT       = re.compile(r"(\S+\.(?:txt|dat))")
    _RE_USING      = re.compile(r"(?:using|u) (\d+):(\d+)")
    _RE_STYLE      = re.compile(r"(?:with|w) (\w+)")
    _RE_TITLE      = re.compile(r'title "(.+?)"')
    _RE_XLABEL     = re.compile(r'xlabel "(.+?)"')
    _RE_YLABEL     = re.compile(r'ylabel "(.+?)"')
    _RE_LW         = re.compile(r"(?:linewidth|lw) (\d+)")
    _RE_LC         = re.compile(r"(?:linecolor|lc) (\S+)")
    _RE_LEGEND     = re.compile(r"(?:legend|lg) (\S+)")
    _RE_FUNC       = re.compile(r'func: "(.+?)"')
    _RE_XRANGE     = re.compile(r"xrange (\d+):(\d+)")
    _RE_YRANGE     = re.compile(r"yrange (\d+):(\d+)")
    _RE_BIN        = re.compile(r"bin (\d+)")
    _RE_PARAMS     = re.compile(r"(\w+)=([\d.]+)")
    # ── New tokens v1.1 ───────────────────────────────────────────────────────
    _RE_YERR       = re.compile(r"yerr (\d+)")
    _RE_CAPSIZE    = re.compile(r"capsize (\d+)")
    _RE_CMAP_COL   = re.compile(r"cmap (\d+)")
    _RE_COLORMAP   = re.compile(r"colormap (\S+)")
    _RE_CBAR_LABEL = re.compile(r'cbar_label "(.+?)"')
    _RE_LEVELS     = re.compile(r"levels (\d+)")

    # ── Advanced axis formatting v1.2 ──────────────────────────────────────────
    _RE_DATE_FORMAT = re.compile(r'date_format "(.+?)"')
    _RE_SCI_NOTATION = re.compile(r"sci_notation (\S+)")
    _RE_XTICKS = re.compile(r'xticks "(.+?)"')
    _RE_YTICKS = re.compile(r'yticks "(.+?)"')
    _RE_XTICK_ROT = re.compile(r"xtick_rotation (-?\d+)")
    _RE_YTICK_ROT = re.compile(r"ytick_rotation (-?\d+)")

    @classmethod
    def parse(cls, command: str) -> PlotConfig:
        """Parse *command* and return a populated PlotConfig.

        Raises:
            ValueError: If mandatory fields are missing.
        """
        log.info("Parsing command: %s", command)
        cfg = PlotConfig()

        # ── File ──────────────────────────────────────────────────────────────
        if m := cls._RE_CSV.search(command):
            cfg.file = m.group(1)
        elif m := cls._RE_TEXT.search(command):
            cfg.file = m.group(1)

        # ── Columns ───────────────────────────────────────────────────────────
        if m := cls._RE_USING.search(command):
            cfg.x_col = int(m.group(1))
            cfg.y_col = int(m.group(2))

        # ── Style / appearance ────────────────────────────────────────────────
        if m := cls._RE_STYLE.search(command):
            cfg.style = m.group(1)
        if m := cls._RE_LW.search(command):
            cfg.linewidth = int(m.group(1))
        if m := cls._RE_LC.search(command):
            cfg.linecolor = m.group(1)

        # ── Labels ────────────────────────────────────────────────────────────
        if m := cls._RE_TITLE.search(command):
            cfg.title = m.group(1)
        if m := cls._RE_XLABEL.search(command):
            cfg.xlabel = m.group(1)
        if m := cls._RE_YLABEL.search(command):
            cfg.ylabel = m.group(1)
        if m := cls._RE_LEGEND.search(command):
            cfg.legend = m.group(1)

        # ── Math function ─────────────────────────────────────────────────────
        if m := cls._RE_FUNC.search(command):
            cfg.function = m.group(1)
            cfg.func_parameters = dict(cls._RE_PARAMS.findall(cfg.function))

        # ── Ranges / bins ─────────────────────────────────────────────────────
        if m := cls._RE_XRANGE.search(command):
            cfg.xrange = [int(m.group(1)), int(m.group(2))]
        if m := cls._RE_YRANGE.search(command):
            cfg.yrange = [int(m.group(1)), int(m.group(2))]
        if m := cls._RE_BIN.search(command):
            cfg.bin = int(m.group(1))

        # ── Error bars ────────────────────────────────────────────────────────
        if m := cls._RE_YERR.search(command):
            cfg.yerr_col = int(m.group(1))
        if m := cls._RE_CAPSIZE.search(command):
            cfg.capsize = int(m.group(1))

        # ── Colormap ──────────────────────────────────────────────────────────
        if m := cls._RE_CMAP_COL.search(command):
            cfg.cmap_col = int(m.group(1))
        if m := cls._RE_COLORMAP.search(command):
            cfg.colormap = m.group(1)
        if m := cls._RE_CBAR_LABEL.search(command):
            cfg.cbar_label = m.group(1)

        # ── 2-D ───────────────────────────────────────────────────────────────
        if m := cls._RE_LEVELS.search(command):
            cfg.levels = int(m.group(1))

        # ── Advanced axis formatting ──────────────────────────────────────────
        if m := cls._RE_DATE_FORMAT.search(command):
            cfg.date_format = m.group(1)
        if m := cls._RE_SCI_NOTATION.search(command):
            cfg.sci_notation = m.group(1)
        if m := cls._RE_XTICKS.search(command):
            cfg.xticks = m.group(1)
        if m := cls._RE_YTICKS.search(command):
            cfg.yticks = m.group(1)
        if m := cls._RE_XTICK_ROT.search(command):
            cfg.xtick_rotation = int(m.group(1))
        if m := cls._RE_YTICK_ROT.search(command):
            cfg.ytick_rotation = int(m.group(1))

        cls._validate(cfg)
        return cfg

    @staticmethod
    def _validate(cfg: PlotConfig) -> None:
        style = cfg.style

        # 2-D styles only need a file, no column spec required.
        if style in TWO_D_STYLES:
            if not cfg.file:
                raise ValueError(f"Style '{style}' requires a file.")
            return

        # Error-bar styles need yerr_col.
        if style in ERRORBAR_STYLES and cfg.yerr_col is None:
            raise ValueError(
                f"Style '{style}' requires  yerr <col>  in the command."
            )

        # Colormap scatter needs cmap_col.
        if style in COLORMAP_STYLES and cfg.cmap_col is None:
            raise ValueError(
                "Style 'scatter' requires  cmap <col>  in the command."
            )

        # Standard styles need file + columns (unless func:).
        if not cfg.function:
            if not cfg.file or cfg.x_col is None or cfg.y_col is None:
                raise ValueError(
                    "File and columns (x, y) must be specified when not using func:."
                )
        if not cfg.file and not cfg.function:
            raise ValueError("Either a file or a func: expression must be provided.")


class JsonParser:
    """Build a PlotConfig from a JSON-derived dictionary."""

    @classmethod
    def parse(cls, data: dict) -> PlotConfig:
        """Parse *data* (a JSON object) and return a PlotConfig."""
        log.info("Parsing JSON command: %s", data)
        cfg = PlotConfig(
            file            = data.get("file"),
            x_col           = data.get("x_col"),
            y_col           = data.get("y_col"),
            style           = data.get("style", "lines"),
            linewidth       = data.get("linewidth", 2),
            linecolor       = data.get("linecolor", "tab:blue"),
            title           = data.get("title"),
            xlabel          = data.get("xlabel", "X-axis"),
            ylabel          = data.get("ylabel", "Y-axis"),
            title_font_size = data.get("title_font_size", 20),
            axis_font_size  = data.get("axis_font_size", 18),
            tick_font_size  = data.get("tick_font_size", 14),
            legend          = data.get("legend"),
            xrange          = data.get("xrange"),
            yrange          = data.get("yrange"),
            bin             = data.get("bin", 10),
            # v1.1 fields — safe to omit in old JSON files (defaults apply).
            yerr_col        = data.get("yerr_col"),
            capsize         = data.get("capsize", 4),
            cmap_col        = data.get("cmap_col"),
            colormap        = data.get("colormap", "viridis"),
            cbar_label      = data.get("cbar_label"),
            levels          = data.get("levels", 10),
            # v1.2 fields — axis formatting
            date_format     = data.get("date_format"),
            sci_notation    = data.get("sci_notation"),
            xticks          = data.get("xticks"),
            yticks          = data.get("yticks"),
            xtick_rotation  = data.get("xtick_rotation", 0),
            ytick_rotation  = data.get("ytick_rotation", 0),
        )

        if cfg.style not in TWO_D_STYLES:
            if not cfg.file or cfg.x_col is None or cfg.y_col is None:
                raise ValueError(
                    "JSON command must contain 'file', 'x_col', and 'y_col'."
                )
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Plotter — renders a PlotConfig onto a matplotlib figure / axes
# ══════════════════════════════════════════════════════════════════════════════

class Plotter:
    """Loads data and draws plots described by a PlotConfig."""

    def __init__(self, cfg: PlotConfig) -> None:
        self.cfg = cfg
        self.data: Optional[pd.DataFrame] = None

        if cfg.file:
            self._load_data()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        if self.cfg.style in TWO_D_STYLES:
            self._load_matrix()
        elif self.cfg.file.endswith(".csv"):
            self._load_csv()
        else:
            self._load_text()

    def _load_csv(self) -> None:
        self.data = pd.read_csv(self.cfg.file)

    def _load_text(self) -> None:
        try:
            self.data = pd.read_csv(self.cfg.file, delimiter=r"\s+", header=None)
        except Exception:
            self.data = pd.read_csv(self.cfg.file, delimiter="\t", header=None)

    def _load_matrix(self) -> None:
        """Load a headerless numeric matrix for 2-D plot styles."""
        try:
            self.data = pd.read_csv(self.cfg.file, delimiter=r"\s+", header=None)
        except Exception:
            self.data = pd.read_csv(self.cfg.file, header=None)

    # ── Column selection helpers ───────────────────────────────────────────────

    def _get_x_data(self, one_based: bool = False) -> pd.Series:
        if self.cfg.x_col == 0:
            return self.data.index
        offset = 1 if one_based else 0
        return self.data.iloc[:, self.cfg.x_col - offset]

    def _get_y_data(self, one_based: bool = False) -> pd.Series:
        offset = 1 if one_based else 0
        return self.data.iloc[:, self.cfg.y_col - offset]

    def _get_err_data(self, one_based: bool = False) -> Optional[pd.Series]:
        if self.cfg.yerr_col is None:
            return None
        offset = 1 if one_based else 0
        return self.data.iloc[:, self.cfg.yerr_col - offset]

    def _get_cmap_data(self, one_based: bool = False) -> Optional[pd.Series]:
        if self.cfg.cmap_col is None:
            return None
        offset = 1 if one_based else 0
        return self.data.iloc[:, self.cfg.cmap_col - offset]

    # ── Axis decoration ───────────────────────────────────────────────────────

    def _apply_axis_decorations(self, ax=None) -> None:
        """Apply title, labels, ranges, tick sizes, legend, and advanced formatting.

        ax=None  →  current pyplot axes (single-panel mode).
        ax=<Axes> → explicit axes (subplot mode).
        """
        if ax is None:
            set_title   = plt.title
            set_xlabel  = plt.xlabel
            set_ylabel  = plt.ylabel
            tick_params = plt.tick_params
            set_xlim    = plt.xlim
            set_ylim    = plt.ylim
            legend_fn   = plt.legend
            ax_obj      = plt.gca()
        else:
            set_title   = ax.set_title
            set_xlabel  = ax.set_xlabel
            set_ylabel  = ax.set_ylabel
            tick_params = ax.tick_params
            set_xlim    = ax.set_xlim
            set_ylim    = ax.set_ylim
            legend_fn   = ax.legend
            ax_obj      = ax

        set_title(self.cfg.title or "", fontsize=self.cfg.title_font_size)
        set_xlabel(self.cfg.xlabel, fontsize=self.cfg.axis_font_size)
        set_ylabel(self.cfg.ylabel, fontsize=self.cfg.axis_font_size)
        tick_params(labelsize=self.cfg.tick_font_size)

        if self.cfg.xrange:
            set_xlim(self.cfg.xrange)
        if self.cfg.yrange:
            set_ylim(self.cfg.yrange)
        if self.cfg.legend:
            legend_fn(frameon=False, fontsize=self.cfg.axis_font_size)

        # ── Advanced axis formatting ──────────────────────────────────────────
        self._apply_advanced_formatting(ax_obj)

    def _apply_advanced_formatting(self, ax) -> None:
        """Apply scientific notation, custom ticks, rotations, date formatting."""
        import matplotlib.ticker as ticker
        from matplotlib.dates import DateFormatter, AutoDateLocator

        # ── Scientific notation ───────────────────────────────────────────────
        if self.cfg.sci_notation in {"x", "both"}:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            log.info("X-axis: scientific notation enabled.")

        if self.cfg.sci_notation in {"y", "both"}:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            log.info("Y-axis: scientific notation enabled.")

        # ── Custom ticks ──────────────────────────────────────────────────────
        if self.cfg.xticks:
            try:
                x_positions = [float(x.strip()) for x in self.cfg.xticks.split(",")]
                ax.set_xticks(x_positions)
                log.info("X-axis custom ticks: %s", x_positions)
            except ValueError:
                log.error("Invalid xticks format. Use comma-separated numbers.")

        if self.cfg.yticks:
            try:
                y_positions = [float(y.strip()) for y in self.cfg.yticks.split(",")]
                ax.set_yticks(y_positions)
                log.info("Y-axis custom ticks: %s", y_positions)
            except ValueError:
                log.error("Invalid yticks format. Use comma-separated numbers.")

        # ── Tick rotation ─────────────────────────────────────────────────────
        if self.cfg.xtick_rotation != 0:
            ax.tick_params(axis="x", rotation=self.cfg.xtick_rotation)
            log.info("X-axis ticks rotated by %d°.", self.cfg.xtick_rotation)

        if self.cfg.ytick_rotation != 0:
            ax.tick_params(axis="y", rotation=self.cfg.ytick_rotation)
            log.info("Y-axis ticks rotated by %d°.", self.cfg.ytick_rotation)

        # ── Date formatting ───────────────────────────────────────────────────
        if self.cfg.date_format:
            try:
                ax.xaxis.set_major_formatter(DateFormatter(self.cfg.date_format))
                ax.xaxis.set_major_locator(AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
                log.info("X-axis date format: %s", self.cfg.date_format)
            except Exception as e:
                log.error("Failed to apply date format: %s", e)

    # ── Core plot renderers ───────────────────────────────────────────────────

    def plot(self) -> None:
        """Render a standard single-panel plot."""
        log.info("Plotting columns [%s, %s]", self.cfg.x_col, self.cfg.y_col)

        one_based = not (self.cfg.file or "").endswith(".csv")
        x_data = self._get_x_data(one_based=one_based)
        y_data = self._get_y_data(one_based=one_based)
        err    = self._get_err_data(one_based=one_based)
        cmap_z = self._get_cmap_data(one_based=one_based)

        self._draw_series(x_data, y_data, err=err, cmap_z=cmap_z, ax=None)
        self._apply_axis_decorations(ax=None)

    def function_plot(self) -> None:
        """Evaluate and render a mathematical function over xrange."""
        if not self.cfg.xrange:
            raise ValueError("xrange must be specified for function plots.")

        x0, x1 = self.cfg.xrange
        x = np.linspace(x0, x1, (x1 - x0) * 10)

        func_str = re.split(r"\)\s*=", self.cfg.function, maxsplit=1)[-1]
        params   = {k: float(v) for k, v in self.cfg.func_parameters.items()}
        y        = eval(func_str, {"x": x, "np": np, **params})  # noqa: S307

        plt.plot(x, y, self.cfg.mpl_style,
                 linewidth=self.cfg.linewidth,
                 color=self.cfg.linecolor,
                 label=self.cfg.legend)
        self._apply_axis_decorations(ax=None)

    def subplot_mosaic(self, ax) -> None:
        """Render this plot into subplot axes *ax*."""
        x_data = self._get_x_data(one_based=False)
        y_data = self.data.iloc[:, self.cfg.y_col]
        err    = self._get_err_data(one_based=False)
        cmap_z = self._get_cmap_data(one_based=False)

        self._draw_series(x_data, y_data, err=err, cmap_z=cmap_z, ax=ax)
        self._apply_axis_decorations(ax=ax)

    # ── Private drawing dispatch ──────────────────────────────────────────────

    def _draw_series(self, x_data, y_data, *, err=None, cmap_z=None, ax) -> None:
        """Route to the correct drawing method based on style."""
        style = self.cfg.style

        if style in TWO_D_STYLES:
            self._draw_2d(ax)
        elif style in ERRORBAR_STYLES:
            self._draw_errorbars(x_data, y_data, err, ax)
        elif style in COLORMAP_STYLES:
            self._draw_colormap_scatter(x_data, y_data, cmap_z, ax)
        elif style in SEABORN_STYLES:
            self._draw_seaborn(style, x_data, ax)
        else:
            self._draw_line(x_data, y_data, ax)

    # ── Individual draw methods ───────────────────────────────────────────────

    def _draw_line(self, x_data, y_data, ax) -> None:
        """Standard matplotlib line / marker plot."""
        kwargs = dict(linewidth=self.cfg.linewidth,
                      color=self.cfg.linecolor,
                      label=self.cfg.legend)
        if ax is None:
            plt.plot(x_data, y_data, self.cfg.mpl_style, **kwargs)
        else:
            ax.plot(x_data, y_data, self.cfg.mpl_style, **kwargs)

    def _draw_errorbars(self, x_data, y_data, err, ax) -> None:
        """Discrete error bars (errorbars/eb) or shaded band (errorshade/es)."""
        if err is None:
            log.error("yerr column not loaded — skipping error bar plot.")
            return

        style = self.cfg.style
        color = self.cfg.linecolor
        label = self.cfg.legend
        lw    = self.cfg.linewidth

        if style in {"errorbars", "eb"}:
            log.info("Drawing discrete error bars.")
            kwargs = dict(fmt=self.cfg.mpl_style, linewidth=lw,
                          color=color, ecolor=color,
                          capsize=self.cfg.capsize, label=label)
            if ax is None:
                plt.errorbar(x_data, y_data, yerr=err, **kwargs)
            else:
                ax.errorbar(x_data, y_data, yerr=err, **kwargs)

        else:  # errorshade / es
            log.info("Drawing shaded error band.")
            if ax is None:
                plt.plot(x_data, y_data, self.cfg.mpl_style,
                         linewidth=lw, color=color, label=label)
                plt.fill_between(x_data,
                                 y_data - err, y_data + err,
                                 alpha=0.25, color=color)
            else:
                ax.plot(x_data, y_data, self.cfg.mpl_style,
                        linewidth=lw, color=color, label=label)
                ax.fill_between(x_data,
                                y_data - err, y_data + err,
                                alpha=0.25, color=color)

    def _draw_colormap_scatter(self, x_data, y_data, cmap_z, ax) -> None:
        """Scatter plot coloured by a third column, with automatic colorbar."""
        if cmap_z is None:
            log.error("cmap column not loaded — skipping colormap scatter.")
            return

        log.info("Drawing colormap scatter (cmap='%s').", self.cfg.colormap)
        kwargs = dict(c=cmap_z, cmap=self.cfg.colormap,
                      label=self.cfg.legend, linewidths=0)
        if ax is None:
            sc = plt.scatter(x_data, y_data, **kwargs)
            cb = plt.colorbar(sc)
        else:
            sc = ax.scatter(x_data, y_data, **kwargs)
            cb = plt.colorbar(sc, ax=ax)

        if self.cfg.cbar_label:
            cb.set_label(self.cfg.cbar_label, fontsize=self.cfg.axis_font_size)

    def _draw_2d(self, ax) -> None:
        """Render the loaded matrix as heatmap, contour, or contourf.

        Rows → y-axis, columns → x-axis.
        """
        matrix = self.data.values.astype(float)
        style  = self.cfg.style
        cmap   = self.cfg.colormap
        levels = self.cfg.levels

        log.info("Drawing 2-D '%s' (shape=%s).", style, matrix.shape)

        if style == "heatmap":
            obj = (plt.imshow if ax is None else ax.imshow)(
                matrix, aspect="auto", cmap=cmap, origin="lower"
            )
        elif style == "contour":
            obj = (plt.contour if ax is None else ax.contour)(
                matrix, levels=levels, cmap=cmap
            )
        else:  # contourf
            obj = (plt.contourf if ax is None else ax.contourf)(
                matrix, levels=levels, cmap=cmap
            )

        cb = plt.colorbar(obj, ax=ax)
        if self.cfg.cbar_label:
            cb.set_label(self.cfg.cbar_label, fontsize=self.cfg.axis_font_size)

    def _draw_seaborn(self, style: str, x_data, ax) -> None:
        """Dispatch to the correct seaborn distribution function."""
        log.info("Rendering seaborn '%s' plot.", style)
        common = dict(color=self.cfg.linecolor, label=self.cfg.legend)
        if ax is not None:
            common["ax"] = ax

        if style == "hist":
            sns.histplot(data=x_data, bins=self.cfg.bin, **common)
        elif style == "kde":
            sns.kdeplot(data=x_data, **common)
        elif style == "box":
            sns.boxplot(data=x_data, **common)
        elif style == "violin":
            sns.violinplot(data=x_data, **common)

    # ── Expression evaluator ──────────────────────────────────────────────────

    def evaluate_expression(self, expr: str):
        """Replace $n column references and evaluate the expression."""
        expr = re.sub(
            r"\$(\d+)",
            lambda m: f"self.data.iloc[:, {int(m.group(1)) - 1}]",
            expr,
        )
        return eval(expr)  # noqa: S307


# ══════════════════════════════════════════════════════════════════════════════
# Log-scale helper — applied globally after all series are drawn
# ══════════════════════════════════════════════════════════════════════════════

def apply_log_scale(command: str, axd: Optional[dict] = None) -> None:
    """Set log scale on x and/or y axes when --xlog / --ylog are present.

    Works for both single figures and subplot mosaics.

    Args:
        command: Full raw command string checked for --xlog / --ylog.
        axd:     Axes dict from subplot_mosaic, or None for single-panel.
    """
    xlog = "--xlog" in command
    ylog = "--ylog" in command

    if not xlog and not ylog:
        return

    axes = list(axd.values()) if axd else [plt.gca()]
    for ax in axes:
        if xlog:
            ax.set_xscale("log")
            log.info("x-axis set to log scale.")
        if ylog:
            ax.set_yscale("log")
            log.info("y-axis set to log scale.")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level processing helpers
# ══════════════════════════════════════════════════════════════════════════════

def process_plots(commands: list[str]) -> None:
    """Parse and render a list of command strings; persist configs to JSON."""
    configs: list[PlotConfig] = []

    for command in commands:
        try:
            cfg     = CommandParser.parse(command)
            plotter = Plotter(cfg)
            if cfg.function:
                plotter.function_plot()
            else:
                plotter.plot()
            configs.append(cfg)
        except Exception as exc:
            log.error("Failed to process command %r: %s", command, exc)

    _save_configs_to_json(configs)


def process_plots_json(data: list[dict]) -> None:
    """Render plots from a list of JSON-deserialized config dicts."""
    for entry in data:
        try:
            Plotter(JsonParser.parse(entry)).plot()
        except Exception as exc:
            log.error("Failed to process JSON command: %s", exc)


def process_subplots(commands: list[str], layout: str, axd: dict) -> None:
    """Parse and render commands into a subplot mosaic layout."""
    panel_keys = [ch for ch in layout if ch not in ("\n", " ")]

    for idx, command in enumerate(commands):
        if idx >= len(panel_keys):
            log.warning("More commands than subplot panels — skipping extra.")
            break
        try:
            cfg = CommandParser.parse(command)
            key = panel_keys[idx]
            log.info("Rendering into subplot panel '%s'.", key)
            Plotter(cfg).subplot_mosaic(ax=axd[key])
        except Exception as exc:
            log.error("Failed subplot command %r: %s", command, exc)


def save_plot(command: str) -> None:
    """Save the figure to disk if --save <path> appears in *command*."""
    if "--save" not in sys.argv:
        return
    m = re.search(r"--save (\S+)", command)
    if m:
        path = m.group(1)
        plt.savefig(path)
        log.info("Plot saved as '%s'.", path)


# ── JSON persistence ──────────────────────────────────────────────────────────

def _save_configs_to_json(configs: list[PlotConfig]) -> None:
    serializable = [cfg.__dict__.copy() for cfg in configs]
    with JSON_OUTPUT_PATH.open("w") as fh:
        json.dump(serializable, fh, indent=4)
    log.info("Plot config saved to '%s'.", JSON_OUTPUT_PATH)


# ── Help utilities ─────────────────────────────────────────────────────────────

def _show_help(section: str = "all") -> None:
    """Import mfp_help and print the requested section."""
    try:
        import mfp_help
        mfp_help.show(section)
    except ImportError:
        log.error("mfp_help.py not found — place it alongside mfp.py and retry.")


def _list_styles() -> None:
    print("\n  ── Line / marker styles ─────────────────────────────────────")
    for k, v in STYLE_MAP.items():
        print(f"    {k:<20} → matplotlib '{v}'")
    print("\n  ── Error-bar styles ─────────────────────────────────────────")
    for s in sorted(ERRORBAR_STYLES):
        print(f"    {s}")
    print("\n  ── Colormap scatter ─────────────────────────────────────────")
    for s in sorted(COLORMAP_STYLES):
        print(f"    {s}")
    print("\n  ── 2-D / matrix styles ──────────────────────────────────────")
    for s in sorted(TWO_D_STYLES):
        print(f"    {s}")
    print("\n  ── Seaborn distribution styles ──────────────────────────────")
    for s in sorted(SEABORN_STYLES):
        print(f"    {s}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _build_command_string() -> str:
    return " ".join(
        f'"{arg}"' if " " in arg else arg
        for arg in sys.argv[1:]
    )


def main() -> None:  # noqa: C901
    """Parse CLI arguments and dispatch to the appropriate rendering path."""

    if len(sys.argv) < 2:
        _show_help()
        sys.exit(0)

    subcommand = sys.argv[1]

    # ── Help & info ───────────────────────────────────────────────────────────
    if subcommand in {"--help", "-h", "help"}:
        _show_help(sys.argv[2] if len(sys.argv) > 2 else "all")
        return

    if subcommand == "--list-styles":
        _list_styles()
        return

    # ── Special subcommands ───────────────────────────────────────────────────
    if subcommand == "plot.json":
        log.info("Reading configuration from plot.json …")
        with JSON_OUTPUT_PATH.open() as fh:
            data = json.load(fh)
        plt.figure(figsize=DEFAULT_FIGSIZE)
        process_plots_json(data)
        plt.tight_layout()
        plt.show()
        return

    if subcommand == "forecast":
        os.system(f"python3 {SCRIPT_DIR / 'prophet_pred.py'}")  # noqa: S605
        return

    if subcommand == "DM":
        os.system(f"python3 {SCRIPT_DIR / 'mfp_dmanp.py'}")  # noqa: S605
        return

    # ── Standard plot / subplot path ──────────────────────────────────────────
    command = _build_command_string()

    # Strip global flags before splitting into per-dataset commands.
    core_command = re.sub(
        r"--(?:xlog|ylog|save \S+|subplot \S+)\s*", "", command
    ).strip()
    commands = [cmd.strip() for cmd in core_command.split(",") if cmd.strip()]

    axd = None

    if "--subplot" in sys.argv:
        m = re.search(r"--subplot (\S+)", command)
        if not m:
            log.error("--subplot flag found but no layout string provided.")
            sys.exit(1)
        layout = m.group(1)
        if "-" in layout:
            layout = "\n".join(row.strip() for row in layout.split("-"))
        log.info("Subplot layout: %r", layout)
        fig, axd = plt.subplot_mosaic(layout, figsize=DEFAULT_FIGSIZE)
        process_subplots(commands, layout, axd)

    else:
        plt.figure(figsize=DEFAULT_FIGSIZE)
        process_plots(commands)

    # ── Post-draw global tweaks ───────────────────────────────────────────────
    apply_log_scale(command, axd=axd)
    plt.tight_layout()
    save_plot(command)
    plt.show()


if __name__ == "__main__":
    main()
