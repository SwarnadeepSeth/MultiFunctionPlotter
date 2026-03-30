#!/usr/bin/env python3
"""
MultiFunctionPlotter (MFP) — A versatile tool for data visualization.

Developed by Dr. Swarnadeep Seth.
Version: 1.0.0
Date: June 19, 2024

Description:
    MultiFunctionPlotter (MFP) simplifies the creation of a wide range of plots
    from CSV and text files, as well as custom mathematical functions. With support
    for multiple plot styles and easy-to-use command-line configuration, MFP aims
    to be a versatile and powerful tool for data visualization. Whether you need
    line plots, histograms, KDE plots, or complex subplots, MFP provides the
    functionality and flexibility to meet your plotting needs. MFP keeps the
    simplicity and style of gnuplot-like commands while leveraging Python packages
    to produce professional-quality graphs.

Usage:
    For more information, visit the GitHub page or documentation wiki.
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

DEFAULT_FIGSIZE = (12, 6)
JSON_OUTPUT_PATH = Path("plot.json")

#: Maps gnuplot-style style names to matplotlib linestyle/marker strings.
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

#: Plot styles that are handled by seaborn (distribution / summary plots).
SEABORN_STYLES = {"hist", "kde", "box", "violin"}


# ══════════════════════════════════════════════════════════════════════════════
# PlotConfig — single source of truth for all plot parameters
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlotConfig:
    """Holds every configurable parameter for a single plot command.

    This dataclass is the single source of truth shared by both the
    command-line parser and the JSON parser.  Adding a new parameter
    means adding it here *once*, then updating the relevant parser.
    """

    # ── Data source ───────────────────────────────────────────────────────────
    file: Optional[str] = None
    x_col: Optional[int] = None
    y_col: Optional[int] = None

    # ── Math function (alternative to file) ──────────────────────────────────
    function: Optional[str] = None
    func_parameters: dict = field(default_factory=dict)

    # ── Appearance ────────────────────────────────────────────────────────────
    style: str = "lines"
    linewidth: int = 2
    linecolor: Optional[str] = None

    # ── Labels ────────────────────────────────────────────────────────────────
    title: Optional[str] = None
    xlabel: str = "X-axis"
    ylabel: str = "Y-axis"
    legend: Optional[str] = None

    # ── Font sizes ────────────────────────────────────────────────────────────
    title_font_size: int = 20
    axis_font_size: int = 18
    tick_font_size: int = 14

    # ── Axes ranges & histogram bins ─────────────────────────────────────────
    xrange: Optional[list[int]] = None
    yrange: Optional[list[int]] = None
    bin: Union[int, str] = "auto"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def mpl_style(self) -> str:
        """Return the matplotlib linestyle/marker string for this config."""
        return STYLE_MAP.get(self.style, "-")


# ══════════════════════════════════════════════════════════════════════════════
# Parsers — convert raw input into a PlotConfig
# ══════════════════════════════════════════════════════════════════════════════

class CommandParser:
    """Parse a gnuplot-style command string into a :class:`PlotConfig`.

    Example command::

        mydata.csv using 1:2 with lines title "My Plot" xlabel "Time" ylabel "Value"
    """

    # Pre-compiled regex patterns for efficiency.
    _RE_CSV      = re.compile(r"(\S+\.csv)")
    _RE_TEXT     = re.compile(r"(\S+\.(?:txt|dat))")
    _RE_USING    = re.compile(r"(?:using|u) (\d+):(\d+)")
    _RE_STYLE    = re.compile(r"(?:with|w) (\w+)")
    _RE_TITLE    = re.compile(r'title "(.+?)"')
    _RE_XLABEL   = re.compile(r'xlabel "(.+?)"')
    _RE_YLABEL   = re.compile(r'ylabel "(.+?)"')
    _RE_LW       = re.compile(r"(?:linewidth|lw) (\d+)")
    _RE_LC       = re.compile(r"(?:linecolor|lc) (\S+)")
    _RE_LEGEND   = re.compile(r"(?:legend|lg) (\S+)")
    _RE_FUNC     = re.compile(r'func: "(.+?)"')
    _RE_XRANGE   = re.compile(r"xrange (\d+):(\d+)")
    _RE_YRANGE   = re.compile(r"yrange (\d+):(\d+)")
    _RE_BIN      = re.compile(r"bin (\d+)")
    _RE_PARAMS   = re.compile(r"(\w+)=([\d.]+)")

    @classmethod
    def parse(cls, command: str) -> PlotConfig:
        """Parse *command* and return a populated :class:`PlotConfig`.

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

        # ── Validation ────────────────────────────────────────────────────────
        cls._validate(cfg)
        return cfg

    @staticmethod
    def _validate(cfg: PlotConfig) -> None:
        if not cfg.function:
            if not cfg.file or cfg.x_col is None or cfg.y_col is None:
                raise ValueError(
                    "File and columns (x, y) must be specified when not using func:."
                )
        if not cfg.file and not cfg.function:
            raise ValueError("Either a file or a func: expression must be provided.")


class JsonParser:
    """Build a :class:`PlotConfig` from a JSON-derived dictionary.

    The expected keys match the field names of :class:`PlotConfig`.
    """

    @classmethod
    def parse(cls, data: dict) -> PlotConfig:
        """Parse *data* (a JSON object) and return a :class:`PlotConfig`.

        Raises:
            ValueError: If mandatory fields are missing.
        """
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
        )

        if not cfg.file or cfg.x_col is None or cfg.y_col is None:
            raise ValueError(
                "JSON command must contain 'file', 'x_col', and 'y_col'."
            )
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Plotter — renders a PlotConfig onto a matplotlib figure / axes
# ══════════════════════════════════════════════════════════════════════════════

class Plotter:
    """Loads data and draws plots described by a :class:`PlotConfig`.

    Args:
        cfg: The fully populated plot configuration to render.
    """

    def __init__(self, cfg: PlotConfig) -> None:
        self.cfg = cfg
        self.data: Optional[pd.DataFrame] = None

        if cfg.file:
            self._load_data()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Dispatch to the correct loader based on file extension."""
        if self.cfg.file.endswith(".csv"):
            self._load_csv()
        else:
            self._load_text()

    def _load_csv(self) -> None:
        self.data = pd.read_csv(self.cfg.file)

    def _load_text(self) -> None:
        """Try whitespace-separated first, fall back to tab-separated."""
        try:
            self.data = pd.read_csv(self.cfg.file, delimiter=r"\s+", header=None)
        except Exception:
            self.data = pd.read_csv(self.cfg.file, delimiter="\t", header=None)

    # ── Column selection helpers ───────────────────────────────────────────────

    def _get_x_data(self, one_based: bool = False) -> pd.Series:
        """Return the x-column Series.

        Args:
            one_based: If *True*, treat column indices as 1-based (text files).
        """
        if self.cfg.x_col == 0:
            return self.data.index
        offset = 1 if one_based else 0
        return self.data.iloc[:, self.cfg.x_col - offset]

    def _get_y_data(self, one_based: bool = False) -> pd.Series:
        """Return the y-column Series."""
        offset = 1 if one_based else 0
        return self.data.iloc[:, self.cfg.y_col - offset]

    # ── Axis decoration (shared by all rendering paths) ───────────────────────

    def _apply_axis_decorations(self, ax=None) -> None:
        """Apply title, labels, ranges, tick sizes, and legend.

        When *ax* is ``None`` the current pyplot axes are used (single-plot
        mode); otherwise the explicit axes object is used (subplot mode).
        """
        if ax is None:
            # Single-figure helpers
            set_title  = plt.title
            set_xlabel = plt.xlabel
            set_ylabel = plt.ylabel
            tick_params = plt.tick_params
            set_xlim   = plt.xlim
            set_ylim   = plt.ylim
            legend_fn  = plt.legend
        else:
            set_title  = ax.set_title
            set_xlabel = ax.set_xlabel
            set_ylabel = ax.set_ylabel
            tick_params = ax.tick_params
            set_xlim   = ax.set_xlim
            set_ylim   = ax.set_ylim
            legend_fn  = ax.legend

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

    # ── Core plot renderers ───────────────────────────────────────────────────

    def plot(self) -> None:
        """Render a standard single-panel plot onto the current figure."""
        log.info("Plotting columns [%s, %s]", self.cfg.x_col, self.cfg.y_col)

        one_based = not (self.cfg.file or "").endswith(".csv")
        x_data = self._get_x_data(one_based=one_based)
        y_data = self._get_y_data(one_based=one_based)

        self._draw_series(x_data, y_data, ax=None)
        self._apply_axis_decorations(ax=None)

    def function_plot(self) -> None:
        """Evaluate and render a mathematical function over ``xrange``."""
        if not self.cfg.xrange:
            raise ValueError("xrange must be specified for function plots.")

        x0, x1 = self.cfg.xrange
        num_points = (x1 - x0) * 10
        x = np.linspace(x0, x1, num_points)

        # Strip the "f(x) =" / "f(x)=" prefix to isolate the expression.
        func_str = re.split(r"\)\s*=", self.cfg.function, maxsplit=1)[-1]
        params = {k: float(v) for k, v in self.cfg.func_parameters.items()}

        y = eval(func_str, {"x": x, "np": np, **params})  # noqa: S307

        plt.plot(
            x, y,
            self.cfg.mpl_style,
            linewidth=self.cfg.linewidth,
            color=self.cfg.linecolor,
            label=self.cfg.legend,
        )
        self._apply_axis_decorations(ax=None)

    def subplot_mosaic(self, ax) -> None:
        """Render this plot into a specific subplot axes *ax*.

        Args:
            ax: The :class:`matplotlib.axes.Axes` object to draw into.
        """
        x_data = self._get_x_data(one_based=False)

        self._draw_series(x_data, self.data.iloc[:, self.cfg.y_col], ax=ax)
        self._apply_axis_decorations(ax=ax)

    # ── Private drawing dispatch ──────────────────────────────────────────────

    def _draw_series(self, x_data, y_data, ax) -> None:
        """Draw the data series using the configured style.

        Routes to seaborn for distribution/summary plots, or matplotlib for
        standard line/scatter plots.

        Args:
            x_data: The independent-variable data (Series or index).
            y_data: The dependent-variable data (Series).  Ignored by
                seaborn distribution plots that only need *x_data*.
            ax:     Target axes.  ``None`` means the current pyplot axes.
        """
        style = self.cfg.style

        if style in SEABORN_STYLES:
            self._draw_seaborn(style, x_data, ax)
        else:
            kwargs = dict(
                linewidth=self.cfg.linewidth,
                color=self.cfg.linecolor,
                label=self.cfg.legend,
            )
            if ax is None:
                plt.plot(x_data, y_data, self.cfg.mpl_style, **kwargs)
            else:
                ax.plot(x_data, y_data, self.cfg.mpl_style, **kwargs)

    def _draw_seaborn(self, style: str, x_data, ax) -> None:
        """Dispatch to the correct seaborn function for distribution plots."""
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
        """Replace ``$n`` column references and evaluate the expression.

        Args:
            expr: An expression string where ``$n`` refers to the n-th
                  column of the loaded data (1-based).

        Returns:
            The evaluated result (typically a :class:`pandas.Series`).
        """
        expr = re.sub(
            r"\$(\d+)",
            lambda m: f"self.data.iloc[:, {int(m.group(1)) - 1}]",
            expr,
        )
        return eval(expr)  # noqa: S307


# ══════════════════════════════════════════════════════════════════════════════
# Top-level processing helpers
# ══════════════════════════════════════════════════════════════════════════════

def process_plots(commands: list[str]) -> None:
    """Parse and render a list of command strings onto the current figure.

    After all commands are rendered the parsed configs are persisted to
    ``plot.json`` so the figure can be reproduced later.

    Args:
        commands: Raw gnuplot-style command strings.
    """
    configs: list[PlotConfig] = []

    for command in commands:
        try:
            cfg = CommandParser.parse(command)
            plotter = Plotter(cfg)

            if cfg.function:
                log.info("Function plot detected.")
                plotter.function_plot()
            else:
                plotter.plot()

            configs.append(cfg)

        except Exception as exc:
            log.error("Failed to process command %r: %s", command, exc)

    _save_configs_to_json(configs)


def process_plots_json(data: list[dict]) -> None:
    """Render plots from a list of JSON-deserialized config dictionaries.

    Args:
        data: List of dicts, each matching the :class:`PlotConfig` field names.
    """
    for entry in data:
        try:
            cfg = JsonParser.parse(entry)
            Plotter(cfg).plot()
        except Exception as exc:
            log.error("Failed to process JSON command: %s", exc)


def process_subplots(commands: list[str], layout: str, axd: dict) -> None:
    """Parse and render commands into a subplot mosaic layout.

    Args:
        commands: Raw gnuplot-style command strings (one per subplot panel).
        layout:   Matplotlib subplot-mosaic layout string.
        axd:      The axes dictionary returned by :func:`plt.subplot_mosaic`.
    """
    panel_keys = [ch for ch in layout if ch not in ("\n", " ")]

    for idx, command in enumerate(commands):
        if idx >= len(panel_keys):
            log.warning("More commands than subplot panels — skipping extra.")
            break
        try:
            cfg = CommandParser.parse(command)
            plotter = Plotter(cfg)
            key = panel_keys[idx]
            log.info("Rendering into subplot panel '%s'.", key)
            plotter.subplot_mosaic(ax=axd[key])
        except Exception as exc:
            log.error("Failed subplot command %r: %s", command, exc)


def save_plot(command: str) -> None:
    """Save the current figure if ``--save <path>`` is present in *command*.

    Args:
        command: The full raw command string (may contain ``--save path``).
    """
    if "--save" not in sys.argv:
        return
    m = re.search(r"--save (\S+)", command)
    if m:
        path = m.group(1)
        plt.savefig(path)
        log.info("Plot saved as '%s'.", path)


# ── JSON persistence ──────────────────────────────────────────────────────────

def _save_configs_to_json(configs: list[PlotConfig]) -> None:
    """Persist a list of :class:`PlotConfig` objects to :data:`JSON_OUTPUT_PATH`."""
    serializable = []
    for cfg in configs:
        d = cfg.__dict__.copy()
        # dataclasses may contain non-JSON-serialisable values; guard here.
        serializable.append(d)

    with JSON_OUTPUT_PATH.open("w") as fh:
        json.dump(serializable, fh, indent=4)
    log.info("Plot config saved to '%s'.", JSON_OUTPUT_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _build_command_string() -> str:
    """Re-join sys.argv arguments, quoting any that contain spaces."""
    return " ".join(
        f'"{arg}"' if " " in arg else arg
        for arg in sys.argv[1:]
    )


def main() -> None:  # noqa: C901 — complexity acceptable for a CLI dispatcher
    """Parse CLI arguments and dispatch to the appropriate rendering path."""

    if len(sys.argv) < 2:
        log.error("No arguments provided.  Pass a plot command or 'plot.json'.")
        sys.exit(1)

    subcommand = sys.argv[1]

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
        prophet_script = SCRIPT_DIR / "prophet_pred.py"
        os.system(f"python3 {prophet_script}")  # noqa: S605
        return

    if subcommand == "DM":
        dm_script = SCRIPT_DIR / "mfp_data_manipulator.py"
        os.system(f"python3 {dm_script}")  # noqa: S605
        return

    # ── Standard plot / subplot path ──────────────────────────────────────────
    command = _build_command_string()
    commands = [cmd.strip() for cmd in command.split(",")]

    if "--subplot" in sys.argv:
        layout_raw = re.search(r"--subplot (\S+)", command)
        if not layout_raw:
            log.error("--subplot flag found but no layout string provided.")
            sys.exit(1)

        layout = layout_raw.group(1)
        # Allow "AB-CD" as shorthand for a two-row layout "AB\nCD".
        if "-" in layout:
            layout = "\n".join(row.strip() for row in layout.split("-"))

        log.info("Subplot layout: %r", layout)
        fig, axd = plt.subplot_mosaic(layout, figsize=DEFAULT_FIGSIZE)
        process_subplots(commands, layout, axd)

    else:
        plt.figure(figsize=DEFAULT_FIGSIZE)
        process_plots(commands)

    plt.tight_layout()
    save_plot(command)
    plt.show()


if __name__ == "__main__":
    main()
