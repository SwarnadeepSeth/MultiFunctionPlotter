#!/usr/bin/env python3
"""
Unit tests for MultiFunctionPlotter (MFP)
Tests different plot types using the provided CSV and DAT files.
"""

import os
import sys
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
TESTS_DIR = SCRIPT_DIR
sys.path.insert(0, os.path.join(PARENT_DIR, 'src'))

DATA_CSV = os.path.join(TESTS_DIR, "data.csv")
DATA_DAT = os.path.join(TESTS_DIR, "data.dat")

import mfp
from mfp import PlotConfig, CommandParser, JsonParser, Plotter


class TestCommandParser:
    """Test CommandParser for various plot types."""

    def test_basic_lines(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines")
        assert cfg.file == DATA_CSV
        assert cfg.x_col == 1
        assert cfg.y_col == 2
        assert cfg.style == "lines"

    def test_dashed_style(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with dashed")
        assert cfg.style == "dashed"

    def test_points_style(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with points")
        assert cfg.style == "points"

    def test_with_title(self):
        cfg = CommandParser.parse(f'{DATA_CSV} using 1:2 with lines title "Test Plot"')
        assert cfg.title == "Test Plot"

    def test_with_labels(self):
        cfg = CommandParser.parse(
            f'{DATA_CSV} using 1:2 with lines xlabel "Time" ylabel "Price"'
        )
        assert cfg.xlabel == "Time"
        assert cfg.ylabel == "Price"

    def test_with_legend(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines legend mydata")
        assert cfg.legend == "mydata"

    def test_with_linewidth(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines linewidth 3")
        assert cfg.linewidth == 3

    def test_with_linecolor(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines linecolor tab:red")
        assert cfg.linecolor == "tab:red"

    def test_with_ranges(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines xrange 0:100 yrange 0:1000")
        assert cfg.xrange == [0, 100]
        assert cfg.yrange == [0, 1000]

    def test_errorbars_style(self):
        cfg = CommandParser.parse(f"{DATA_DAT} using 1:2 with errorbars yerr 3")
        assert cfg.style == "errorbars"
        assert cfg.yerr_col == 3

    def test_errorshade_style(self):
        cfg = CommandParser.parse(f"{DATA_DAT} using 1:2 with errorshade yerr 3")
        assert cfg.style == "errorshade"

    def test_capsize(self):
        cfg = CommandParser.parse(f"{DATA_DAT} using 1:2 with errorbars yerr 3 capsize 5")
        assert cfg.capsize == 5

    def test_scatter_colormap(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with scatter cmap 3 colormap plasma")
        assert cfg.style == "scatter"
        assert cfg.cmap_col == 3
        assert cfg.colormap == "plasma"

    def test_hist_style(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 0:1 with hist bin 30")
        assert cfg.style == "hist"
        assert cfg.bin == 30

    def test_kde_style(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 0:1 with kde")
        assert cfg.style == "kde"

    def test_box_style(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 0:1 with box")
        assert cfg.style == "box"

    def test_violin_style(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 0:1 with violin")
        assert cfg.style == "violin"

    def test_2d_heatmap(self):
        cfg = CommandParser.parse(f"{DATA_DAT} with heatmap")
        assert cfg.style == "heatmap"

    def test_2d_contour(self):
        cfg = CommandParser.parse(f"{DATA_DAT} with contour")
        assert cfg.style == "contour"

    def test_2d_contourf(self):
        cfg = CommandParser.parse(f"{DATA_DAT} with contourf levels 15")
        assert cfg.style == "contourf"
        assert cfg.levels == 15

    def test_function_plot(self):
        cfg = CommandParser.parse('func: "f(x) = np.sin(x)" xrange 0:10')
        assert cfg.function == "f(x) = np.sin(x)"
        assert cfg.xrange == [0, 10]


class TestAdvancedAxisFormatting:
    """Test the new advanced axis formatting features (v1.2)."""

    def test_sci_notation_x(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines sci_notation x")
        assert cfg.sci_notation == "x"

    def test_sci_notation_y(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines sci_notation y")
        assert cfg.sci_notation == "y"

    def test_sci_notation_both(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines sci_notation both")
        assert cfg.sci_notation == "both"

    def test_xticks(self):
        cfg = CommandParser.parse(f'{DATA_CSV} using 1:2 with lines xticks "0,90,180,270"')
        assert cfg.xticks == "0,90,180,270"

    def test_yticks(self):
        cfg = CommandParser.parse(f'{DATA_CSV} using 1:2 with lines yticks "0,1e-5,2e-5"')
        assert cfg.yticks == "0,1e-5,2e-5"

    def test_xtick_rotation(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines xtick_rotation 45")
        assert cfg.xtick_rotation == 45

    def test_ytick_rotation(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines ytick_rotation 90")
        assert cfg.ytick_rotation == 90

    def test_negative_xtick_rotation(self):
        cfg = CommandParser.parse(f"{DATA_CSV} using 1:2 with lines xtick_rotation -30")
        assert cfg.xtick_rotation == -30

    def test_date_format(self):
        cfg = CommandParser.parse(f'{DATA_CSV} using 1:2 with lines date_format "%Y-%m-%d"')
        assert cfg.date_format == "%Y-%m-%d"

    def test_combined_formatting(self):
        cfg = CommandParser.parse(
            f'{DATA_CSV} using 1:2 with lines sci_notation y yticks "0,1e-5,2e-5" xtick_rotation 30'
        )
        assert cfg.sci_notation == "y"
        assert cfg.yticks == "0,1e-5,2e-5"
        assert cfg.xtick_rotation == 30


class TestJsonParser:
    """Test JsonParser for loading plot configs from JSON."""

    def test_basic_json(self):
        data = {
            "file": DATA_CSV,
            "x_col": 1,
            "y_col": 2,
            "style": "lines",
        }
        cfg = JsonParser.parse(data)
        assert cfg.file == DATA_CSV
        assert cfg.x_col == 1
        assert cfg.y_col == 2
        assert cfg.style == "lines"

    def test_json_with_style(self):
        data = {
            "file": DATA_CSV,
            "x_col": 1,
            "y_col": 2,
            "style": "points",
            "linecolor": "tab:red",
        }
        cfg = JsonParser.parse(data)
        assert cfg.style == "points"
        assert cfg.linecolor == "tab:red"

    def test_json_errorbars(self):
        data = {
            "file": DATA_DAT,
            "x_col": 1,
            "y_col": 2,
            "style": "errorbars",
            "yerr_col": 3,
        }
        cfg = JsonParser.parse(data)
        assert cfg.yerr_col == 3

    def test_json_2d(self):
        data = {
            "file": DATA_DAT,
            "style": "heatmap",
            "colormap": "viridis",
        }
        cfg = JsonParser.parse(data)
        assert cfg.style == "heatmap"
        assert cfg.colormap == "viridis"

    def test_json_advanced_formatting(self):
        data = {
            "file": DATA_CSV,
            "x_col": 1,
            "y_col": 2,
            "style": "lines",
            "sci_notation": "both",
            "xticks": "0,50,100",
            "yticks": "0,100,200",
            "xtick_rotation": 45,
            "ytick_rotation": 90,
            "date_format": "%Y-%m-%d",
        }
        cfg = JsonParser.parse(data)
        assert cfg.sci_notation == "both"
        assert cfg.xticks == "0,50,100"
        assert cfg.yticks == "0,100,200"
        assert cfg.xtick_rotation == 45
        assert cfg.ytick_rotation == 90
        assert cfg.date_format == "%Y-%m-%d"


class TestPlotter:
    """Test Plotter class with various plot styles."""

    def setup_method(self):
        plt.close('all')

    def teardown_method(self):
        plt.close('all')

    def test_plot_lines(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="lines",
        )
        plotter = Plotter(cfg)
        plotter.plot()
        assert len(plt.gca().lines) > 0

    def test_plot_points(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="points",
        )
        plotter = Plotter(cfg)
        plotter.plot()
        ax = plt.gca()
        assert len(ax.lines) > 0 or len(ax.collections) > 0

    def test_plot_dashed(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="dashed",
        )
        plotter = Plotter(cfg)
        plotter.plot()
        assert len(plt.gca().lines) > 0

    def test_plot_errorbars(self):
        cfg = PlotConfig(
            file=DATA_DAT,
            x_col=1,
            y_col=2,
            style="errorbars",
            yerr_col=3,
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_plot_errorshade(self):
        cfg = PlotConfig(
            file=DATA_DAT,
            x_col=1,
            y_col=2,
            style="errorshade",
            yerr_col=3,
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_plot_scatter(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="scatter",
            cmap_col=3,
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_plot_hist(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="hist",
            bin=20,
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_plot_kde(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="kde",
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_plot_box(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="box",
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_plot_violin(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="violin",
        )
        plotter = Plotter(cfg)
        plotter.plot()


class TestAdvancedFormatting:
    """Test advanced axis formatting in Plotter."""

    def setup_method(self):
        plt.close('all')

    def teardown_method(self):
        plt.close('all')

    def test_sci_notation_x(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="lines",
            sci_notation="x",
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_sci_notation_y(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="lines",
            sci_notation="y",
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_sci_notation_both(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="lines",
            sci_notation="both",
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_xticks(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="lines",
            xticks="0,50,100",
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_yticks(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="lines",
            yticks="0,100,200",
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_xtick_rotation(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="lines",
            xtick_rotation=45,
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_ytick_rotation(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="lines",
            ytick_rotation=90,
        )
        plotter = Plotter(cfg)
        plotter.plot()

    def test_combined_formatting(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="lines",
            sci_notation="y",
            xticks="0,50,100",
            yticks="0,100,200",
            xtick_rotation=30,
            ytick_rotation=45,
        )
        plotter = Plotter(cfg)
        plotter.plot()


class TestFunctionPlot:
    """Test function plotting."""

    def setup_method(self):
        plt.close('all')

    def teardown_method(self):
        plt.close('all')

    def test_function_sin(self):
        cfg = PlotConfig(
            function="f(x) = np.sin(x)",
            xrange=[0, 10],
            style="lines",
        )
        plotter = Plotter(cfg)
        plotter.function_plot()
        assert len(plt.gca().lines) > 0

    def test_function_with_params(self):
        cfg = PlotConfig(
            function="f(x) = 2*np.cos(x)",
            xrange=[0, 10],
            style="lines",
        )
        plotter = Plotter(cfg)
        plotter.function_plot()
        assert len(plt.gca().lines) > 0


class TestValidation:
    """Test validation logic."""

    def test_errorbars_requires_yerr(self):
        cfg = PlotConfig(
            file=DATA_DAT,
            x_col=1,
            y_col=2,
            style="errorbars",
        )
        try:
            CommandParser._validate(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "yerr" in str(e)

    def test_scatter_requires_cmap(self):
        cfg = PlotConfig(
            file=DATA_CSV,
            x_col=1,
            y_col=2,
            style="scatter",
        )
        try:
            CommandParser._validate(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cmap" in str(e)

    def test_2d_no_columns_needed(self):
        cfg = PlotConfig(
            file=DATA_DAT,
            style="heatmap",
        )
        CommandParser._validate(cfg)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
