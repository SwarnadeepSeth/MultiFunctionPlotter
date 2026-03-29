# MultiFunctionPlotter (MFP)

A versatile Python-based tool for creating publication-quality plots from CSV, TXT, or DAT files. MFP combines the simplicity of gnuplot-style commands with the power of Python's matplotlib and seaborn libraries.

## Version: 1.2.0

---

## Table of Contents

- [MultiFunctionPlotter (MFP)](#multifunctionplotter-mfp)
  - [Version: 1.2.0](#version-120)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Option 1: Install Dependencies](#option-1-install-dependencies)
    - [Option 2: Install as Command-Line Tool (Recommended)](#option-2-install-as-command-line-tool-recommended)
    - [MCP Server Installation](#mcp-server-installation)
    - [Usage with AI Assistants](#usage-with-ai-assistants)
  - [Quick Start](#quick-start)
    - [Command Line](#command-line)
    - [Python API](#python-api)
  - [Basic Usage](#basic-usage)
    - [Command Syntax](#command-syntax)
    - [Common Tokens](#common-tokens)
    - [Full Example](#full-example)
  - [Plot Styles](#plot-styles)
    - [Line \& Marker Styles](#line--marker-styles)
    - [Error Bars](#error-bars)
    - [Scatter with Colormap](#scatter-with-colormap)
    - [Distribution Plots](#distribution-plots)
    - [2D Plots](#2d-plots)
  - [Advanced Axis Formatting (v1.2)](#advanced-axis-formatting-v12)
    - [Scientific Notation](#scientific-notation)
    - [Custom Tick Positions](#custom-tick-positions)
    - [Tick Rotation](#tick-rotation)
    - [Date Formatting](#date-formatting)
    - [Combined Example](#combined-example)
  - [Mathematical Functions](#mathematical-functions)
  - [Subplots](#subplots)
  - [Log Scale](#log-scale)
  - [Saving Figures](#saving-figures)
    - [Custom Figure Size](#custom-figure-size)
  - [JSON Configuration](#json-configuration)
  - [Time Series Forecasting](#time-series-forecasting)
  - [Data Manipulator](#data-manipulator)
    - [Launch](#launch)
    - [Supported File Formats](#supported-file-formats)
    - [Available Actions](#available-actions)
      - [Inspection Commands](#inspection-commands)
      - [Transformation Commands](#transformation-commands)
      - [Data Cleaning](#data-cleaning)
      - [I/O Commands](#io-commands)
      - [History Commands](#history-commands)
    - [Examples](#examples)
    - [Tips](#tips)
  - [Examples](#examples-1)
    - [Stock Price Analysis](#stock-price-analysis)
    - [Error Analysis](#error-analysis)
    - [Scientific Data](#scientific-data)
    - [Publication Quality](#publication-quality)
  - [Help System](#help-system)
    - [Command Line Help](#command-line-help)
    - [List All Styles](#list-all-styles)
  - [Contributing](#contributing)
  - [License](#license)
  - [Author](#author)

---

## Features

- **Multiple File Formats**: Load data from CSV, TXT, or DAT files
- **Rich Plot Styles**: Line plots, points, dashed lines, error bars, scatter, histograms, KDE, box plots, violin plots
- **2D Visualizations**: Heatmaps, contour plots, filled contours
- **Colormap Scatter**: Color points by a third variable
- **Custom Functions**: Plot mathematical expressions directly
- **Advanced Axis Formatting**: Scientific notation, custom ticks, rotations, date formatting
- **Subplot Layouts**: Organize multiple plots in grid layouts
- **Log Scale**: Logarithmic axes for spectrum analysis
- **JSON Config**: Save and replay plot configurations
- **CLI & API**: Use from command line or Python code

---

## Installation

### Option 1: Install Dependencies

Install the required Python packages:

```
pip install -r requirements.txt
```

**Required packages:**
- matplotlib
- numpy
- pandas
- seaborn

### Option 2: Install as Command-Line Tool (Recommended)

Install directly from PyPI:

```
pip3 install multifunctionplotter
```

After installation, you can run `mfp` from anywhere:

```
mfp data.csv using 1:2 with lines
mfp --help
mfp forecast
```

### MCP Server Installation

For AI assistant integration (Claude Code, opencode, etc.):

```
pip3 install multifunctionplotter
mfp-mcp
```

### Usage with AI Assistants

**Claude Code:**
```
claude mcp add mfp -- mfp-mcp
```

**opencode:** Add to `~/.config/opencode/opencode.json`:

1. Find the mfp-mcp path:
   ```bash
   which mfp-mcp
   ```

2. Open the config file:
   ```bash
   nano ~/.config/opencode/opencode.json
   ```

3. Add the mfp configuration:
   ```json
   {
     "$schema": "https://opencode.ai/config.json",
     "mcp": {
       "mfp": {
         "type": "local",
         "command": ["<path-to-mfp-mcp>"]
       }
     }
   }
   ```

Replace `<path-to-mfp-mcp>` with the path from step 1.

---

## Important Notes

### CLI vs Python Tool

The MFP CLI command and Python `mfp_multi_plot` tool work slightly differently:
- CLI parses labels but may not save them to `plot.json` for replay
- Python tool (`mfp_multi_plot`) handles subplot configurations more reliably
- For complex plots, prefer the Python tool or the `mfp_plot_function` tool

---

## Quick Start

### Command Line

```
# Basic line plot
mfp data.csv using 1:2 with lines

# With title and labels
mfp data.csv using 1:2 with lines title "My Plot" xlabel "X" ylabel "Y"

# Save to file
mfp data.csv using 1:2 with lines --save plot.png
```

### Python API

```python
import sys
sys.path.insert(0, 'src')
from mfp import PlotConfig, Plotter

# Create plot configuration
cfg = PlotConfig(
    file="data.csv",
    x_col=1, y_col=2,
    style="lines",
    title="My Plot",
    xlabel="X Axis",
    ylabel="Y Axis"
)

# Generate plot
Plotter(cfg).plot()
```

---

## Basic Usage

### Command Syntax

```
mfp <file> using <x_col>:<y_col> with <style> [options]
```

### Common Tokens

| Token | Description | Example |
|-------|-------------|---------|
| `using` / `u` | Column indices (1-based for text, 0-based for CSV) | `using 1:2` |
| `with` / `w` | Plot style | `with lines` |
| `title` | Plot title | `title "My Title"` |
| `xlabel` | X-axis label | `xlabel "Time"` |
| `ylabel` | Y-axis label | `xlabel "Price"` |
| `legend` / `lg` | Legend entry | `legend "Series 1"` |
| `linewidth` / `lw` | Line width (default: 2) | `linewidth 3` |
| `linecolor` / `lc` | Line/marker color | `linecolor tab:red` |
| `xrange` | X-axis limits | `xrange 0:100` |
| `yrange` | Y-axis limits | `yrange 0:1000` |

### Full Example

```
mfp data.csv using 1:2 with lines title "Stock Prices" xlabel "Date" ylabel "Close Price" linecolor tab:blue linewidth 2 --save plot.png
```

---

## Plot Styles

### Line & Marker Styles

| Style | Alias | Description |
|-------|-------|-------------|
| `lines` | `l` | Solid line |
| `dashed` | | Dashed line |
| `dotted` | | Dotted line |
| `points` | `p` | Circle markers only |
| `linespoints` | `lp` | Line + markers |
| `stars` | `*` | Star markers |
| `d` | | Diamond markers |

**Example:**
```
mfp data.csv using 1:2 with points linecolor tab:red
mfp data.csv using 1:2 with dashed linecolor tab:green
```

### Error Bars

Two styles available:

**Discrete Error Bars** (`errorbars` / `eb`):
```
mfp data.dat using 1:2 with errorbars yerr 3
```

**Shaded Error Band** (`errorshade` / `es`):
```
mfp data.dat using 1:2 with errorshade yerr 3
```

**Extra tokens:**
- `yerr <col>` - Column with ±σ values
- `capsize <int>` - Cap width (default: 4)

**Combine with lines:**
```
mfp data.csv using 1:2 with errorshade yerr 3 lc steelblue, data.csv using 1:2 with lines lc steelblue
```

### Scatter with Colormap

Plot x vs y colored by a third variable:

```
mfp data.csv using 1:2 with scatter cmap 3 colormap plasma
```

**Tokens:**
- `cmap <col>` - Column for color values (required)
- `colormap <name>` - Matplotlib colormap (default: viridis)
- `cbar_label` - Colorbar label

**Useful Colormaps:**
- Perceptually uniform: `viridis`, `plasma`, `inferno`, `magma`, `cividis`
- Diverging: `coolwarm`, `RdBu`, `seismic`
- Sequential: `Blues`, `Reds`, `YlOrRd`

### Distribution Plots

| Style | Description |
|-------|-------------|
| `hist` | Histogram (use `bin <n>` for number of bins) |
| `kde` | Kernel Density Estimation |
| `box` | Box-and-whisker plot |
| `violin` | Violin plot |

**Examples:**
```
mfp data.csv using 0:1 with hist bin 30
mfp data.csv using 0:1 with kde
mfp data.csv using 0:1 with box
mfp data.csv using 0:1 with violin
```

### 2D Plots

For matrix/grid data (no column specification needed):

| Style | Description |
|-------|-------------|
| `heatmap` | 2D heatmap (imshow) |
| `contour` | Contour lines |
| `contourf` | Filled contours |

**Examples:**
```
mfp matrix.dat with heatmap colormap viridis
mfp matrix.dat with contourf levels 20 colormap RdBu
mfp matrix.dat with contour levels 15
```

**Tokens:**
- `colormap <name>` - Colormap (default: viridis)
- `levels <n>` - Number of contour levels (default: 10)
- `cbar_label` - Colorbar label

---

## Advanced Axis Formatting (v1.2)

Control axis appearance with precision for publication-quality plots.

### Scientific Notation

```
mfp data.csv using 1:2 with lines sci_notation x
mfp data.csv using 1:2 with lines sci_notation y
mfp data.csv using 1:2 with lines sci_notation both
```

### Custom Tick Positions

```
mfp data.csv using 1:2 with lines xticks "0,90,180,270"
mfp data.csv using 1:2 with lines yticks "0,1e-5,2e-5"
```

### Tick Rotation

```
mfp data.csv using 1:2 with lines xtick_rotation 45
mfp data.csv using 1:2 with lines ytick_rotation 90
```

### Date Formatting

```
mfp data.csv using 1:2 with lines date_format "%Y-%m-%d"
```

Format codes: `%Y` (year), `%m` (month), `%d` (day), `%H` (hour), `%M` (min), `%S` (sec)

### Combined Example

```
mfp data.csv using 1:2 with lines sci_notation both xticks "0,300,600" xtick_rotation 30
```

---

## Mathematical Functions

Plot mathematical expressions directly:

```
mfp func: "f(x) = np.sin(x)" xrange 0:10
```

**With parameters:**
```
mfp func: "f(x,a=2) = a*np.cos(x)" xrange 0:10
```

**Tokens:**
- `func:` - Start function definition
- `xrange` - Required x-axis range
- Use `np.` prefix for numpy functions

**Examples:**
```
mfp func: "f(x) = np.sin(x)" xrange 0:10 lc red
mfp func: "f(x) = x**2" xrange 0:5 lc blue lw 2
mfp func: "f(x,a=1,b=2) = a*np.exp(-b*x)" xrange 0:5
```

---

## Subplots

Create multi-panel figures using `--subplot`:

```
mfp --subplot AB data.csv using 1:2 with lines, data.csv using 1:2 with hist
```

**Layout format:**
- Letters represent panels (A, B, C, ...)
- Use `-` to separate rows
- Each command separated by comma

**2x2 Grid:**
```
mfp --subplot AB-CD "plot1, plot2, plot3, plot4"
```

**Asymmetric Layout:**
```
mfp --subplot AA-BC "top_full, bottom_left, bottom_right"
```

**Important: Subplot Labels**

When using subplots, axis labels must be specified **per-command**, not globally:

```
# CORRECT - labels per subplot
mfp "cmd1 xlabel 'X' ylabel 'Y', cmd2 xlabel 'A' ylabel 'B'"

# Labels may not apply correctly to individual subplots when set globally
```

**Histogram Requirement:**

The `hist` style requires both x and y columns (even if y is just a placeholder):

```
# CORRECT
mfp data.csv using 5:0 with hist bin 25

# WILL ERROR - missing y column
mfp data.csv using 5 with hist bin 25
```

---

## Log Scale

Apply logarithmic scale to axes:

```
mfp spectrum.csv using 1:2 with lines --ylog
mfp spectrum.csv using 1:2 with lines --xlog --ylog
```

**Tokens:**
- `--xlog` - Logarithmic x-axis
- `--ylog` - Logarithmic y-axis

---

## Saving Figures

Save plots to file:

```
mfp data.csv using 1:2 with lines --save plot.png
mfp data.csv using 1:2 with lines --save plot.pdf
mfp data.csv using 1:2 with lines --save plot.svg
```

**Supported formats:**
- `.png` - Raster (good for web)
- `.pdf` - Vector (best for publications)
- `.svg` - Vector (good for editing)
- `.eps` - Vector (legacy journals)

### Custom Figure Size

Set custom figure dimensions:

```
mfp data.csv using 1:2 with lines --figsize 10:5
```

This sets figure to 10×5 inches. Default is 12×6.

---

## JSON Configuration

MFP saves your last plot configuration to `plot.json`:

```
mfp data.csv using 1:2 with lines --save plot.png
# plot.json is automatically saved
mfp plot.json  # Replay last plot
```

**Manual JSON creation:**
```json
{
    "file": "data.csv",
    "x_col": 1,
    "y_col": 2,
    "style": "lines",
    "title": "My Plot",
    "linecolor": "tab:blue",
    "linewidth": 2
}
```

Then run:
```
mfp plot.json
```

---

## Time Series Forecasting

Forecast future values using Facebook Prophet:

```
mfp forecast
```

This launches an interactive forecasting tool.

**Data format for forecasting:**
```
Date,Value
2019-01-01,100
2019-01-02,105
...
```

---

## Data Manipulator

MFP includes a powerful interactive data manipulation tool for exploring, cleaning, and transforming tabular data without writing code.

### Launch

```
mfp DM
```

Or directly:
```
python src/mfp_dmanp.py data.csv
```

### Supported File Formats

- CSV files (.csv)
- Excel files (.xlsx, .xls)
- JSON files (.json)

### Available Actions

#### Inspection Commands

| Command | Description |
|---------|-------------|
| `show` | Print the current DataFrame |
| `head [N]` | Print first N rows (default: 10) |
| `tail [N]` | Print last N rows (default: 10) |
| `properties` / `props` | Column dtypes, NaN counts, summary statistics |
| `counts <col>` | Frequency count of unique values |

#### Transformation Commands

| Command | Description |
|---------|-------------|
| `filter <expr>` | Keep rows matching a pandas query expression |
| `slice <start:end>` | Keep rows in range [start, end) |
| `sort <col> asc/desc` | Sort by a column |
| `rename <old:new,...>` | Rename columns |
| `cast <col> <type>` | Change column dtype (int/float/str/datetime) |
| `addcol <name> <expr>` | Add new column from expression |
| `modify <col> <old> <new>` | Replace values in a column |
| `delete <col>` | Drop columns or rows |

#### Data Cleaning

| Command | Description |
|---------|-------------|
| `dedup [col1,col2]` | Remove duplicate rows |
| `fillna <col> <value>` | Fill NaN cells with value |
| `dropna <col>` | Drop rows with NaN |

#### I/O Commands

| Command | Description |
|---------|-------------|
| `load <file>` | Load a new file |
| `generate <expr>` | Generate data from expression |
| `gen` | Alias for generate |
| `append <file>` | Append rows from another file |
| `merge <file> <on_col>` | Merge with another file |
| `save <file>` | Save to file |

#### History Commands

| Command | Description |
|---------|-------------|
| `undo` | Revert last operation |
| `redo` | Re-apply undone operation |

### Examples

```
Action> load data.csv
Action> head
Action> properties
Action> filter price > 100
Action> sort volume desc
Action> rename old_name:new_name
Action> addcol profit revenue - cost
Action> dedup
Action> save cleaned_data.csv
```

### Tips

- Use `help` to see all commands
- Use `undo` / `redo` to navigate changes
- Expressions support pandas query syntax
- Use `addcol` with `df.eval()` expressions

---

## Examples

### Stock Price Analysis
```
# Close price over time
mfp data.csv using 0:4 with lines title "Close Price" xlabel "Date" ylabel "Price"

# Multiple prices
mfp "data.csv using 0:2 with lines lc green, data.csv using 0:4 with lines lc blue"
```

### Error Analysis
```
mfp data.dat using 1:2 with errorbars yerr 3 lc red
mfp data.dat using 1:2 with errorshade yerr 3 lc orange
```

### Scientific Data
```
# Log scale for spectrum
mfp spectrum.csv using 1:2 with lines --ylog

# Scientific notation
mfp data.csv using 1:2 with lines sci_notation both
```

### Publication Quality
```
mfp data.csv using 1:2 with lines title "Results" xlabel "Time (s)" ylabel "Voltage (mV)" linewidth 3 sci_notation y xtick_rotation 45 --save figure.pdf
```

---

## Help System

### Command Line Help
```
mfp --help
mfp --help tokens
mfp --help styles
mfp --help errorbars
mfp --help colormap
mfp --help 2d
mfp --help logscale
mfp --help subplots
mfp --help save
mfp --help formatting
mfp --help examples
```

### List All Styles
```
mfp --list-styles
```

---

## Contributing

Contributions are welcome! Please feel free to:
- Open an issue for bug reports or feature requests
- Submit a pull request for improvements
- Share your use cases and examples

---

## License

MIT License

---

## Author

**Dr. Swarnadeep Seth**  
MultiFunctionPlotter (MFP) - A versatile data visualization tool
