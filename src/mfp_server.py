#!/usr/bin/env python3
"""
mfp_server.py — MCP server for MultiFunctionPlotter (mfp)

Wraps mfp's full CLI as MCP tools so any AI agent (Claude Code, Cursor,
Windsurf, LangChain, etc.) can call it in plain English with no syntax
knowledge required.

Install dep:  pip install fastmcp
Run directly: python mfp_server.py
Add to Claude Code: claude mcp add mfp -- python /path/to/mfp_server.py

Author: based on mfp by Dr. Swarnadeep Seth
"""

import shutil
import subprocess
import sys
import io
from pathlib import Path

from fastmcp import FastMCP

# ── Locate the mfp executable ────────────────────────────────────────────────
# MCP clients launch servers with a stripped PATH (no conda/venv on it), so
# shutil.which("mfp") often fails even when mfp is properly installed.
# Resolution order:
#   1. Same bin/ directory as the Python running this server  ← most reliable
#   2. shutil.which("mfp")                                    ← normal PATH
#   3. src/mfp.py next to this file                           ← dev/repo layout

def _find_mfp() -> list[str]:
    # 1) Sibling of sys.executable (works inside any venv / conda env)
    mfp_sibling = Path(sys.executable).parent / "mfp"
    if mfp_sibling.exists():
        return [str(mfp_sibling)]

    # 2) PATH lookup (works when shell environment is inherited)
    mfp_path = shutil.which("mfp")
    if mfp_path:
        return [mfp_path]

    # 3) Dev/repo fallback: mfp.py lives in src/ next to this file
    repo_root = Path(__file__).resolve().parent
    mfp_py = repo_root / "src" / "mfp.py"
    if mfp_py.exists():
        return [sys.executable, str(mfp_py)]

    raise FileNotFoundError(
        "Cannot find mfp. Tried:\n"
        f"  {Path(sys.executable).parent / 'mfp'} (sibling of current Python)\n"
        f"  mfp on PATH\n"
        f"  {repo_root / 'src' / 'mfp.py'} (dev layout)\n"
        "Install with:  pip install multifunctionplotter"
    )

_MFP_CMD = _find_mfp()
_REPO_ROOT = Path(__file__).resolve().parent

def _run(args: list[str], stdin_text: str | None = None) -> str:
    """Run mfp with the given args and return combined stdout+stderr."""
    result = subprocess.run(
        _MFP_CMD + args,
        capture_output=True,
        text=True,
        input=stdin_text,
    )
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    # mfp writes INFO logs to stderr — only surface them on failure
    if result.returncode != 0:
        return f"Error (exit {result.returncode}):\n{err}\n{out}".strip()
    return out or "Done."


# ═════════════════════════════════════════════════════════════════════════════
mcp = FastMCP("MultiFunctionPlotter")
# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — plot  (single series from a data file)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool
def plot(
    file: str,
    x_col: int,
    y_col: int,
    style: str = "lines",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: str = "",
    linecolor: str = "",
    linewidth: int = 2,
    xrange: str = "",
    yrange: str = "",
    save: str = "plot.png",
    xlog: bool = False,
    ylog: bool = False,
    sci_notation: str = "",
    xticks: str = "",
    yticks: str = "",
    xtick_rotation: int = 0,
    ytick_rotation: int = 0,
    date_format: str = "",
    yerr_col: int = 0,
    capsize: int = 4,
    cmap_col: int = 0,
    colormap: str = "viridis",
    cbar_label: str = "",
    levels: int = 10,
    bin: int = 0,
) -> str:
    """
    Plot a single data series from a CSV, TXT, or DAT file.

    Column indexing:
      - CSV files: 0-based  (first col = 0)
      - TXT / DAT files: 1-based (first col = 1)

    Styles available:
      Line/marker : lines (l), dashed, dotted, points (p), linespoints (lp),
                    stars, d
      Error bars  : errorbars (eb) — needs yerr_col
                    errorshade (es) — needs yerr_col
      Colormap    : scatter — needs cmap_col
      2-D matrix  : heatmap, contour, contourf  (no x_col/y_col needed)
      Distribution: hist, kde, box, violin

    Args:
        file:           Path to data file (.csv, .txt, .dat)
        x_col:          Column index for x-axis
        y_col:          Column index for y-axis
        style:          Plot style (see above). Default: lines
        title:          Plot title (will be quoted automatically)
        xlabel:         X-axis label
        ylabel:         Y-axis label
        legend:         Legend entry for this series (single word, no spaces)
        linecolor:      Any matplotlib color: 'tab:blue', 'red', '#3a7ab3', 'steelblue'
        linewidth:      Line width in points. Default: 2
        xrange:         X-axis limits as 'min:max', e.g. '0:100'
        yrange:         Y-axis limits as 'min:max'
        save:           Output file path. Supports .png .pdf .svg .eps. Default: plot.png
        xlog:           Use log scale on x-axis
        ylog:           Use log scale on y-axis
        sci_notation:   Scientific notation axis: 'x', 'y', or 'both'
        xticks:         Custom x-tick positions as '0,90,180,270'
        yticks:         Custom y-tick positions as '0,1e-5,2e-5'
        xtick_rotation: Rotate x-axis labels by this many degrees
        ytick_rotation: Rotate y-axis labels by this many degrees
        date_format:    Parse x-axis as dates, e.g. '%Y-%m-%d' or '%d/%m/%Y'
        yerr_col:       Column index with ±σ error values (for errorbars/errorshade)
        capsize:        Error bar cap width in points. Default: 4
        cmap_col:       Column index for scatter colormap values
        colormap:       Matplotlib colormap name. Default: viridis
        cbar_label:     Colorbar label text
        levels:         Number of contour levels for contour/contourf. Default: 10
        bin:            Number of histogram bins (0 = auto)

    Returns:
        Success message with the output path, or error details.

    Examples:
        plot("data.csv", 0, 4, style="lines", title="Close Price", save="price.png")
        plot("results.dat", 1, 2, style="errorbars", yerr_col=3, linecolor="tab:red")
        plot("samples.csv", 0, 1, style="hist", bin=30, save="dist.pdf")
        plot("matrix.dat", 0, 0, style="heatmap", colormap="inferno", save="heat.png")
        plot("data.csv", 1, 2, style="scatter", cmap_col=3, colormap="plasma")
    """
    # Build the gnuplot-style command string that mfp parses
    cmd_parts = [file, "using", f"{x_col}:{y_col}", "with", style]

    if title:           cmd_parts += [f'title "{title}"']
    if xlabel:          cmd_parts += [f'xlabel "{xlabel}"']
    if ylabel:          cmd_parts += [f'ylabel "{ylabel}"']
    if legend:          cmd_parts += ["legend", legend]
    if linecolor:       cmd_parts += ["lc", linecolor]
    if linewidth != 2:  cmd_parts += ["lw", str(linewidth)]
    if xrange:          cmd_parts += ["xrange", xrange]
    if yrange:          cmd_parts += ["yrange", yrange]
    if sci_notation:    cmd_parts += ["sci_notation", sci_notation]
    if xticks:          cmd_parts += [f'xticks "{xticks}"']
    if yticks:          cmd_parts += [f'yticks "{yticks}"']
    if xtick_rotation:  cmd_parts += ["xtick_rotation", str(xtick_rotation)]
    if ytick_rotation:  cmd_parts += ["ytick_rotation", str(ytick_rotation)]
    if date_format:     cmd_parts += [f'date_format "{date_format}"']
    if yerr_col:        cmd_parts += ["yerr", str(yerr_col), "capsize", str(capsize)]
    if cmap_col:        cmd_parts += ["cmap", str(cmap_col), "colormap", colormap]
    if cbar_label:      cmd_parts += [f'cbar_label "{cbar_label}"']
    if levels != 10:    cmd_parts += ["levels", str(levels)]
    if bin:             cmd_parts += ["bin", str(bin)]

    args = cmd_parts + ["--save", save]
    if xlog: args += ["--xlog"]
    if ylog: args += ["--ylog"]

    result = _run(args)
    if result.startswith("Error"):
        return result
    return f"Plot saved to: {save}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — plot_function  (math expression, no data file)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool
def plot_function(
    expression: str,
    xrange: str,
    save: str = "plot.png",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: str = "",
    linecolor: str = "",
    linewidth: int = 2,
    yrange: str = "",
    ylog: bool = False,
) -> str:
    """
    Plot a mathematical function directly — no data file needed.

    Uses numpy (np.) functions. Parameters can be embedded in the expression.

    Args:
        expression: Function definition string. Format: 'f(x) = <expr>'
                    or 'f(x, param=value) = <expr>'.
                    Use np. prefix for numpy functions.
                    Examples:
                      'f(x) = np.sin(x)'
                      'f(x) = x**2 + np.cos(x)'
                      'f(x, a=2) = a * np.exp(-x)'
                      'f(x, a=1, b=2) = a * np.exp(-b * x)'
                      'f(x) = np.sin(x) / x'
        xrange:     Required. X-axis range as 'min:max'. E.g. '0:10', '-5:5'
        save:       Output file path (.png, .pdf, .svg, .eps). Default: plot.png
        title:      Plot title
        xlabel:     X-axis label
        ylabel:     Y-axis label
        legend:     Legend entry (single word)
        linecolor:  Matplotlib color string
        linewidth:  Line width. Default: 2
        yrange:     Y-axis limits as 'min:max'
        ylog:       Use log scale on y-axis

    Returns:
        Success message with output path, or error details.

    Examples:
        plot_function("f(x) = np.sin(x)", xrange="-10:10", save="sin.png")
        plot_function("f(x) = x**2", xrange="0:5", linecolor="tab:red", save="parabola.pdf")
        plot_function("f(x,a=1,b=2) = a*np.exp(-b*x)", xrange="0:5", title="Decay")
    """
    # mfp parses: func: "f(x) = ..." xrange min:max [tokens] --save path
    cmd_parts = [f'func: "{expression}"', "xrange", xrange]

    if title:           cmd_parts += [f'title "{title}"']
    if xlabel:          cmd_parts += [f'xlabel "{xlabel}"']
    if ylabel:          cmd_parts += [f'ylabel "{ylabel}"']
    if legend:          cmd_parts += ["legend", legend]
    if linecolor:       cmd_parts += ["lc", linecolor]
    if linewidth != 2:  cmd_parts += ["lw", str(linewidth)]
    if yrange:          cmd_parts += ["yrange", yrange]

    args = cmd_parts + ["--save", save]
    if ylog: args += ["--ylog"]

    result =_run(args)
    if result.startswith("Error"):
        return result
    return f"Plot saved to: {save}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — multi_plot  (multiple series / subplots in one call)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool
def multi_plot(
    commands: str,
    save: str = "plot.png",
    subplot_layout: str = "",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    xlog: bool = False,
    ylog: bool = False,
) -> str:
    """
        Plot multiple series on one figure, or arrange plots in a subplot grid.

        This exposes mfp's full multi-command syntax — the most powerful tool
        for complex figures. Comma-separate individual plot commands.

        IMPORTANT: Each sub-command must start with a filename or 'func:'.
        mfp uses the first token of each comma-separated part to detect splits.

        IMPORTANT: For subplots, axis labels must be specified per-command, not globally.
        IMPORTANT: hist style requires both x and y columns — use 0 as y placeholder if needed.

        Args:
            commands:       One or more mfp commands, comma-separated.
                            Each command follows the same syntax as the plot tool.

                            Multiple series on one axes:
                            'data.csv using 0:2 with lines lc green legend Open,
                            data.csv using 0:4 with lines lc blue legend Close'

                            Error band + mean line overlay (classic combo):
                            'data.dat using 1:2 with errorshade yerr 3 lc steelblue,
                            data.dat using 1:2 with lines lc steelblue'

                            Multiple functions:
                            'func: "f(x) = np.sin(x)" xrange -10:10 lc blue,
                            func: "f(x) = np.cos(x)" xrange -10:10 lc red'

                            Subplots with per-command labels (CORRECT):
                            'data.csv using 1:2 with lines xlabel "Time" ylabel "Value",
                            data.csv using 1:0 with hist bin 20 xlabel "X" ylabel "Count"'

            save:           Output file path. Default: plot.png

            subplot_layout: Optional layout string for subplot grids.
                            Letters = panels, '-' separates rows.
                            Each panel gets one comma-separated command (left→right, top→bottom).
                            Examples:
                            'AB'      → 1 row, 2 panels side by side
                            'AB-CD'   → 2×2 grid
                            'AA-BC'   → A spans full top row, B and C share bottom
                            Leave empty to overlay all series on one axes.

            title:          Figure title (applied to last series — for subplots, embed
                            xlabel/ylabel inside each command instead).
            xlabel:         X-axis label. WARNING: for subplots embed inside each command
                            e.g. 'data.csv using 1:2 with lines xlabel "Time"'
            ylabel:         Y-axis label. Same caveat as xlabel for subplots.
            xlog:           Log scale on x-axis (applies to all panels)
            ylog:           Log scale on y-axis (applies to all panels)

        Returns:
            Success message with output path, or error details.

        NOTES:
            - hist style requires both x and y columns — use 0 as y placeholder:
            CORRECT:  'data.csv using 5:0 with hist bin 25'
            WRONG:    'data.csv using 5 with hist bin 25'  ← will error

        Examples:
            # Two overlaid series with axis labels
            multi_plot(
                "data.csv using 0:2 with lines lc green legend Open, data.csv using 0:4 with lines lc blue legend Close",
                save="comparison.png",
                title="Open vs Close Price",
                xlabel="Date",
                ylabel="Price",
            )

            # 2-panel subplot grid — labels per command
            multi_plot(
                'data.csv using 1:2 with lines xlabel "Time" ylabel "Value", '
                'data.csv using 1:0 with hist bin 30 xlabel "X" ylabel "Count"',
                subplot_layout="AB",
                save="grid.png"
            )

            # Asymmetric layout: full-width top, two panels bottom
            multi_plot(
                'data.csv using 1:2 with lines xlabel "Time" ylabel "Value", '
                'data.csv using 1:0 with hist xlabel "X" ylabel "Count", '
                'data.csv using 0:2 with kde xlabel "X" ylabel "Density"',
                subplot_layout="AA-BC",
                save="layout.png"
            )
        """
    args = []
    if subplot_layout:
        args += ["--subplot", subplot_layout]

    # Build the full command string, appending global tokens after the series commands
    cmd = commands
    if title:  cmd += f' title "{title}"'
    if xlabel: cmd += f' xlabel "{xlabel}"'
    if ylabel: cmd += f' ylabel "{ylabel}"'

    args += [cmd, "--save", save]
    if xlog: args += ["--xlog"]
    if ylog: args += ["--ylog"]

    result = _run(args)
    if result.startswith("Error"):
        return result
    return f"Plot saved to: {save}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — inspect_data  (non-mutating data exploration)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool
def inspect_data(
    file: str,
    action: str = "properties",
    n_rows: int = 10,
    column: str = "",
) -> str:
    """
    Inspect a data file without modifying it.

    Uses mfp's Data Manipulator (DM) in non-interactive mode by importing
    MFPDataManipulator directly (not via subprocess) so output is captured.

    Args:
        file:    Path to CSV, Excel (.xlsx/.xls), or JSON file
        action:  What to do. Options:
                   'properties' — column names, dtypes, NaN counts, summary stats
                   'head'       — first n_rows rows
                   'tail'       — last n_rows rows
                   'show'       — full DataFrame
                   'counts'     — value frequency for a column (requires column=)
        n_rows:  Number of rows for head/tail. Default: 10
        column:  Column name, required for action='counts'

    Returns:
        Formatted text output of the inspection result.

    Examples:
        inspect_data("data.csv")
        inspect_data("data.csv", action="head", n_rows=5)
        inspect_data("data.csv", action="counts", column="Category")
    """
    import sys
    import io

    # Import the Data Manipulator directly to capture its print output
    from multifunctionplotter.mfp_dmanp import MFPDataManipulator

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    try:
        dm = MFPDataManipulator(file)
        if action == "properties":
            dm.properties()
        elif action == "head":
            dm.head(n_rows)
        elif action == "tail":
            dm.tail(n_rows)
        elif action == "show":
            dm.show()
        elif action == "counts":
            if not column:
                sys.stdout = old_stdout
                return "Error: 'counts' action requires a column name."
            dm.counts(column)
        else:
            sys.stdout = old_stdout
            return f"Unknown action '{action}'. Use: properties, head, tail, show, counts"
    except Exception as exc:
        sys.stdout = old_stdout
        return f"Error: {exc}"
    finally:
        sys.stdout = old_stdout

    return buf.getvalue().strip() or "Done."


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — clean_data  (mutating transformations, saves result)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool
def clean_data(
    file: str,
    save_as: str,
    filter_query: str = "",
    sort_col: str = "",
    sort_order: str = "asc",
    rename_pairs: str = "",
    cast_col: str = "",
    cast_dtype: str = "",
    add_col_name: str = "",
    add_col_expr: str = "",
    drop_columns: str = "",
    dedup: bool = False,
    dedup_cols: str = "",
    fillna_col: str = "",
    fillna_value: str = "",
    dropna_col: str = "",
    slice_start: int = -1,
    slice_end: int = -1,
) -> str:
    """
    Clean and transform a data file, saving the result to a new file.

    All operations run in sequence on the data. Only specify the ones you need.

    Args:
        file:           Input file path (CSV, Excel, JSON)
        save_as:        Output file path — format from extension (.csv, .xlsx, .json)

        filter_query:   Keep rows matching a pandas query expression.
                        Examples: 'price > 100', 'city == "Roanoke"',
                                  'score >= 90 and grade == "A"'
                        Column names with spaces: backtick them: '`first name` == "Alice"'

        sort_col:       Column name to sort by
        sort_order:     'asc' or 'desc'. Default: asc

        rename_pairs:   Rename columns as 'old:new' or multiple 'old1:new1,old2:new2'

        cast_col:       Column name to change dtype
        cast_dtype:     Target dtype: 'int', 'float', 'str', or 'datetime'

        add_col_name:   Name for a new computed column
        add_col_expr:   Expression for the new column (pandas eval syntax)
                        Examples: 'price * qty', 'revenue - cost', 'score / score.max()'

        drop_columns:   Comma-separated column names to drop: 'col1,col2'

        dedup:          Remove duplicate rows if True
        dedup_cols:     Comma-separated columns to consider for dedup (empty = all)

        fillna_col:     Column to fill NaN/empty values in
        fillna_value:   Value to fill with (auto-converts to int/float if numeric)

        dropna_col:     Drop rows where this column is NaN (empty string = any column)

        slice_start:    Keep rows from this index (0-based, -1 = disabled)
        slice_end:      Keep rows up to this index exclusive (-1 = disabled)

    Returns:
        Summary of operations performed and the saved file path.

    Examples:
        # Filter, sort, deduplicate, save
        clean_data(
            "raw.csv", "cleaned.csv",
            filter_query="price > 100",
            sort_col="volume", sort_order="desc",
            dedup=True
        )

        # Add computed column, rename, save as Excel
        clean_data(
            "sales.csv", "sales_with_profit.xlsx",
            add_col_name="profit", add_col_expr="revenue - cost",
            rename_pairs="cust_id:customer_id,txn_dt:date"
        )

        # Fill missing values and drop bad rows
        clean_data(
            "data.csv", "data_clean.csv",
            fillna_col="price", fillna_value="0",
            dropna_col="date"
        )
    """
    
    from multifunctionplotter.mfp_dmanp import MFPDataManipulator

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    steps_done = []
    try:
        dm = MFPDataManipulator(file)

        if slice_start >= 0 and slice_end > slice_start:
            dm.slice(slice_start, slice_end)
            steps_done.append(f"Sliced rows {slice_start}:{slice_end}")

        if filter_query:
            dm.filter(filter_query)
            steps_done.append(f"Filtered: {filter_query}")

        if sort_col:
            dm.sort(sort_col, sort_order)
            steps_done.append(f"Sorted by {sort_col} ({sort_order})")

        if rename_pairs:
            dm.rename(rename_pairs)
            steps_done.append(f"Renamed: {rename_pairs}")

        if cast_col and cast_dtype:
            dm.cast(cast_col, cast_dtype)
            steps_done.append(f"Cast {cast_col} → {cast_dtype}")

        if add_col_name and add_col_expr:
            dm.addcol(add_col_name, add_col_expr)
            steps_done.append(f"Added column '{add_col_name}' = {add_col_expr}")

        if drop_columns:
            dm.delete(drop_columns)
            steps_done.append(f"Dropped columns: {drop_columns}")

        if fillna_col and fillna_value:
            dm.fillna(fillna_col, fillna_value)
            steps_done.append(f"Filled NaN in '{fillna_col}' with {fillna_value}")

        if dropna_col:
            dm.dropna(dropna_col if dropna_col else None)
            scope = f"'{dropna_col}'" if dropna_col else "any column"
            steps_done.append(f"Dropped NaN rows in {scope}")

        if dedup:
            dm.dedup(dedup_cols if dedup_cols else None)
            steps_done.append(f"Deduplicated" + (f" on {dedup_cols}" if dedup_cols else ""))

        saved_path = dm.save(save_as)
        steps_done.append(f"Saved → {saved_path}")

    except Exception as exc:
        sys.stdout = old_stdout
        return f"Error during cleaning: {exc}\nSteps completed: {steps_done}"
    finally:
        sys.stdout = old_stdout

    log_output = buf.getvalue().strip()
    summary = "\n".join(steps_done)
    return f"Steps performed:\n{summary}\n\nDetail log:\n{log_output}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6 — replay_config  (replay plot.json)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool
def replay_config(config_file: str = "plot.json") -> str:
    """
    Replay a saved plot configuration from a JSON file.

    mfp automatically saves the last plot's configuration to plot.json.
    You can also craft these files manually for reproducible figures.

    Args:
        config_file: Path to the JSON config file. Default: plot.json

    Returns:
        Success message or error details.

    Example JSON format:
        {
            "file": "data.csv",
            "x_col": 0,
            "y_col": 4,
            "style": "lines",
            "title": "Stock Price",
            "linecolor": "tab:blue",
            "linewidth": 2
        }
    """
    return _run([config_file])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    mcp.run()

if __name__ == "__main__":
    main()