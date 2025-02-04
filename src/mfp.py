#!/usr/bin/env python3

"""
MultiFunctionPlotter (MFP) - A versatile tool for data visualization.

Developed by Dr. Swarnadeep Seth.
Version: 1.0.0
Date: June 19, 2024

Description:
MultiFunctionPlotter (MFP) is designed to simplify the creation of a wide range of plots from CSV and text files,
as well as custom mathematical functions. With support for multiple plot styles and easy-to-use command-line 
configuration, MFP aims to be a versatile and powerful tool for data visualization. Whether you need line plots, 
histograms, KDE plots, or complex subplots, MFP provides the functionality and flexibility to meet your plotting needs.
MFP keeps the simplicity and style of gnuplot-like commands while leveraging Python packages to produce professional-quality graphs.

Usage:
For more information, visit our GitHub page or documentation wiki.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, re, os, json
from pathlib import Path

try:
    # Load custom style
    plt.style.use('custom_style')
except:
    # Update default settings
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.labelpad'] = 10
    plt.rcParams['axes.titlepad'] = 15
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

class PlotCommand:
    def __init__(self, command):
        self.command = command
        self.file = None
        self.x_col = None
        self.y_col = None
        self.style = 'lines'
        self.linewidth = 2  
        self.linecolor = None
        self.title = None  
        self.xlabel = 'X-axis'
        self.ylabel = 'Y-axis'
        self.title_font_size = 20
        self.axis_font_size = 18
        self.tick_font_size = 14
        self.legend = None
        self.xrange = None
        self.yrange = None
        self.bin = 'auto'

        self.function = None
        self.func_parameters = {}  # To store function parameters

        self.parse_command()

    def parse_command(self):
        print ("Input Command:", self.command)

        # File match (.csv)
        csv_file_match = re.search(r"(\S+\.csv)", self.command)

        # Text file match (.txt or .dat)
        text_file_match = re.search(r"(\S+\.(?:txt|dat))", self.command)
        
        using_match = re.search(r"(?:using|u) (\d+):(\d+)", self.command)
        style_match = re.search(r"(?:with|w) (\w+)", self.command)
        title_match = re.search(r"title \"(.+?)\"", self.command)
        xlabel_match = re.search(r"xlabel \"(.+?)\"", self.command)
        ylabel_match = re.search(r"ylabel \"(.+?)\"", self.command)
        linewidth_match = re.search(r"(?:linewidth|lw) (\d+)", self.command)
        linecolor_match = re.search(r"(?:linecolor|lc) (\S+)", self.command)
        legend_match = re.search(r"(?:legend|lg) (\S+)", self.command)     
        function_match = re.search(r"func: \"(.+?)\"", command)
        xrange_match = re.search(r"xrange (\d+):(\d+)", self.command)
        yrange_match = re.search(r"yrange (\d+):(\d+)", self.command)
        bin_match = re.search(r"bin (\d+)", self.command)
        
        if csv_file_match:
            self.file = csv_file_match.group(1)
        if text_file_match:
            self.file = text_file_match.group(1)
        if using_match:
            self.x_col = int(using_match.group(1))
            self.y_col = int(using_match.group(2))
        if style_match:
            self.style = style_match.group(1)
        if title_match:
            self.title = title_match.group(1)
        if xlabel_match:
            self.xlabel = xlabel_match.group(1)
        if ylabel_match:
            self.ylabel = ylabel_match.group(1)
        if linewidth_match:
            self.linewidth = int(linewidth_match.group(1))
        if linecolor_match:
            self.linecolor = linecolor_match.group(1)
        if legend_match:
            self.legend = legend_match.group(1)
        if function_match:
            self.function = function_match.group(1)
            self.func_parameters = dict(re.findall(r"(\w+)=([\d.]+)", self.function))
        if xrange_match:
            self.xrange = [int(xrange_match.group(1)), int(xrange_match.group(2))]
        if yrange_match:
            self.yrange = [int(yrange_match.group(1)), int(yrange_match.group(2))]
        if bin_match:
            self.bin = int(bin_match.group(1))

        if not function_match:
            if not self.file or self.x_col is None or self.y_col is None:
                raise ValueError("File and columns for x and y must be specified in the command.")
       
        if not self.file and not self.function:
            raise ValueError("File or function must be specified in the command.")

class Plot_from_Json:
    def __init__(self, command):
        self.command = command
        self.file = None
        self.x_col = None
        self.y_col = None
        self.style = 'lines'
        self.linewidth = 2  
        self.linecolor = 'tab:blue' 
        self.title = None  
        self.xlabel = 'X-axis'
        self.ylabel = 'Y-axis'
        self.title_font_size = 20
        self.axis_font_size = 18
        self.tick_font_size = 14
        self.legend = None
        self.xrange = None
        self.yrange = None
        self.bin=10

        self.parse_command()

    def parse_command(self):
        print ("Input Command:", self.command)

        self.file = self.command['file']
        self.x_col = self.command['x_col']
        self.y_col = self.command['y_col']
        self.style = self.command['style']
        self.linewidth = self.command['linewidth']
        self.linecolor = self.command['linecolor']
        self.title = self.command['title']
        self.xlabel = self.command['xlabel']
        self.ylabel = self.command['ylabel']
        self.title_font_size = self.command['title_font_size']
        self.axis_font_size = self.command['axis_font_size']
        self.tick_font_size = self.command['tick_font_size']
        self.legend = self.command['legend']
        self.xrange = self.command['xrange']
        self.yrange = self.command['yrange']
        self.bin = self.command['bin']

        if not self.file or self.x_col is None or self.y_col is None:
            raise ValueError("File and columns for x and y must be specified in the command.")

class Plotter:
    def __init__(self, plot_command):
        self.plot_command = plot_command
        self.data = None

        if self.plot_command.file:
            self.load_data()
    
    def load_data(self):
        if self.plot_command.file.endswith('.csv'):
            self.load_csvdata()
        else:
            self.load_textdata()

    def load_csvdata(self):
        self.data = pd.read_csv(self.plot_command.file)
    
    def load_textdata(self):
        try:
            # Try to read the file as a space-separated file
            self.data = pd.read_csv(self.plot_command.file, delimiter=r"\s+", header=None)
        except:
            # If it fails, read the file as a normal text file
            self.data = pd.read_csv(self.plot_command.file, delimiter="\t", header=None)

    def map_style(self):
        style_map = {
            'lines': '-',
            'l': '-',
            'dashed': '--',
            'dotted': ':',
            'points': 'o',
            'p': 'o',
            'linespoints': '-o',
            'lp': '-o',
            'stars': '*',
            'd': 'd',
        }
        return style_map.get(self.plot_command.style, '-')
            
    def plot(self):
        if self.plot_command.file:
            print(f"Selected columns: [{self.plot_command.x_col}, {self.plot_command.y_col}]")
            if self.plot_command.file.endswith('.csv'):
                if self.plot_command.x_col == 0:
                    x_data = self.data.index  
                else:
                    x_data = self.data.iloc[:, self.plot_command.x_col] 
                y_data = self.data.iloc[:, self.plot_command.y_col]
            else:
                # This part processes .dat files and uses iloc to correctly fetch columns
                if self.plot_command.x_col == 0:
                    x_data = self.data.index
                else:
                    x_data = self.data.iloc[:, self.plot_command.x_col - 1]  # Adjust for 1-based to 0-based index
                y_data = self.data.iloc[:, self.plot_command.y_col - 1]  # Adjust for y_col similarly

        if self.plot_command.style == 'hist':
            print ("Plotting Histogram")
            sns.histplot(data=x_data, bins=self.plot_command.bin, color=self.plot_command.linecolor, label=self.plot_command.legend)
        
        elif self.plot_command.style == 'kde':
            sns.kdeplot(data=x_data, color=self.plot_command.linecolor, label=self.plot_command.legend)

        elif self.plot_command.style == 'box':
            sns.boxplot(data=x_data, color=self.plot_command.linecolor, label=self.plot_command.legend)

        elif self.plot_command.style == 'violin':
            sns.violinplot(data=x_data, color=self.plot_command.linecolor, label=self.plot_command.legend)

        else:
            plt.plot(
                x_data,
                y_data,
                self.map_style(),
                linewidth=self.plot_command.linewidth,
                color=self.plot_command.linecolor,
                label=self.plot_command.legend
            )
        if self.plot_command.title:
            plt.title(self.plot_command.title)

        if self.plot_command.xrange:
            plt.xlim(self.plot_command.xrange)

        if self.plot_command.yrange:
            plt.ylim(self.plot_command.yrange)

        plt.tick_params(labelsize=self.plot_command.tick_font_size)
        plt.title(self.plot_command.title, fontsize=self.plot_command.title_font_size)

        plt.xlabel(self.plot_command.xlabel, fontsize=self.plot_command.axis_font_size)
        plt.ylabel(self.plot_command.ylabel, fontsize=self.plot_command.axis_font_size)

        if self.plot_command.legend:
            plt.legend(frameon=False, fontsize=self.plot_command.axis_font_size)

    def evaluate_expression(self, expr):
        # Replace $n with self.data.iloc[:, n-1] to reference columns
        expr = re.sub(r"\$(\d+)", lambda m: f"self.data.iloc[:, {int(m.group(1))-1}]", expr)
        return eval(expr)


    def function_plot(self):

        num_points = (self.plot_command.xrange[1] - self.plot_command.xrange[0])*10
        x = np.linspace(self.plot_command.xrange[0], self.plot_command.xrange[1], num_points)

        try:
            function_str = self.plot_command.function.split(') =', 1)[1]
        except:
            function_str = self.plot_command.function.split(')=', 1)[1]

        params = self.plot_command.func_parameters
        params = {k: float(v) for k, v in params.items()}

        y = eval(function_str, {"x": x, "np": np, **params})

        plt.plot(x, y, 
                 self.map_style(), 
                 linewidth=self.plot_command.linewidth, 
                 color=self.plot_command.linecolor, 
                 label=self.plot_command.legend)
        
        if self.plot_command.title:
            plt.title(self.plot_command.title)
        
        if self.plot_command.xrange:
            plt.xlim(self.plot_command.xrange)

        if self.plot_command.yrange:
            plt.ylim(self.plot_command.yrange)

        plt.tick_params(labelsize=self.plot_command.tick_font_size)
        plt.title(self.plot_command.title, fontsize=self.plot_command.title_font_size)

        plt.xlabel(self.plot_command.xlabel, fontsize=self.plot_command.axis_font_size)
        plt.ylabel(self.plot_command.ylabel, fontsize=self.plot_command.axis_font_size)

        if self.plot_command.legend:
            plt.legend(frameon=False, fontsize=self.plot_command.axis_font_size)


    def subplot_mosaic(self, index):

        if self.plot_command.file:
            if self.plot_command.x_col == 0:
                x_data = self.data.index
            else:
                x_data = self.data.iloc[:, self.plot_command.x_col]

        if self.plot_command.style == 'hist':
            print ("Plotting Histogram")
            sns.histplot(data=x_data, bins=self.plot_command.bin, color=self.plot_command.linecolor, ax=axd[f'{index}'], label=self.plot_command.legend)
        
        elif self.plot_command.style == 'kde':
            sns.kdeplot(data=x_data, color=self.plot_command.linecolor, ax=axd[f'{index}'], label=self.plot_command.legend)

        elif self.plot_command.style == 'box':
            sns.boxplot(data=x_data, color=self.plot_command.linecolor, ax=axd[f'{index}'], label=self.plot_command.legend)

        elif self.plot_command.style == 'violin':
            sns.violinplot(data=x_data, color=self.plot_command.linecolor, ax=axd[f'{index}'], label=self.plot_command.legend)

        else:
            axd[index].plot(
                x_data,
                self.data.iloc[:, self.plot_command.y_col],
                self.map_style(),
                linewidth=self.plot_command.linewidth,
                color=self.plot_command.linecolor,
                label=self.plot_command.legend
            )

        if self.plot_command.title:
            axd[index].set_title(self.plot_command.title)

        if self.plot_command.xrange:
            axd[index].set_xlim(self.plot_command.xrange)

        if self.plot_command.yrange:
            axd[index].set_ylim(self.plot_command.yrange)
        
        axd[index].tick_params(labelsize=self.plot_command.tick_font_size)
        axd[index].set_title(self.plot_command.title, fontsize=self.plot_command.title_font_size)

        axd[index].set_xlabel(self.plot_command.xlabel, fontsize=self.plot_command.axis_font_size)
        axd[index].set_ylabel(self.plot_command.ylabel, fontsize=self.plot_command.axis_font_size)

        if self.plot_command.legend:
            axd[index].legend(frameon=False, fontsize=self.plot_command.axis_font_size)
    
# =================================================================================================
def process_plots(commands):
    commands_list = []
    for command in commands:
        if command.startswith('func:'):
            print ("Function Plotting")
            try:
                plot_command = PlotCommand(command)
                plotter = Plotter(plot_command)
                plotter.function_plot()
            except Exception as e:
                print(f"Error: {e}")
        else:
            try:
                plot_command = PlotCommand(command)
                plotter = Plotter(plot_command)
                plotter.plot()
            except Exception as e:
                print(f"Error: {e}")

        commands_list.append(plot_command)

    # Save the plot command to a json file
    with open('plot.json', 'w') as f:
        json.dump([cmd.__dict__ for cmd in commands_list], f, indent=4)

def process_plots_json(commands):
    for command in commands:
        try:
            plot_command = Plot_from_Json(command)
            plotter = Plotter(plot_command)
            plotter.plot()
        except Exception as e:
            print(f"Error: {e}")

def process_subplots(commands, layout):
    count = 0
    layout_list = list(layout)
    layout_list = [char for char in layout_list if char != '\n']
    for command in commands:
        try:
            plot_command = PlotCommand(command)
            plotter = Plotter(plot_command)
            index = layout_list[count]
            print ("> Plotted at Index:", index)
            plotter.subplot_mosaic(index)
            count += 1
        except Exception as e:
            print(f"Error: {e}")

def save_plot(command):
    if '--save' in sys.argv:
        path = re.search(r"--save (\S+)", command).group(1)
        plt.savefig(path)
        print (f"Plot saved as {path}.")

# ================================================================================================= 
if __name__ == "__main__":

    figsize = (12, 6)

    # Check if the user has provided any command-line arguments
    if len(sys.argv) < 2:
        print ("Usage: mfp.py [plot command] [--save path] [--subplot layout] [forecast] [DM]")
        exit()

    # Check if user wants to plot from a json file: so see if extension is .json
    if sys.argv[1].endswith('.json'):
        print ("Reading from plot.json file...")
        with open('plot.json') as f:
            data = json.load(f)

        plt.figure(figsize=figsize)
        process_plots_json(data)
        plt.tight_layout()
        plt.show()
        exit()

    elif len(sys.argv) > 1 and sys.argv[1] == 'forecast':
        # Construct the absolute path to prophet_pred.py
        prophet_script = script_dir / 'prophet_pred.py'
        os.system(f'python3 {prophet_script}')
        exit()

    elif len(sys.argv) > 1 and sys.argv[1] == 'DM':
        # Construct the absolute path to mfp_data_manipulator
        data_manipulator_script = script_dir / 'mfp_data_manipulator.py'
        os.system(f'python3 {data_manipulator_script}')
        exit()

    elif len(sys.argv) > 1:
        command = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in sys.argv[1:])

    commands = [cmd.strip() for cmd in command.split(',')]

    if '--subplot' in sys.argv:
        layout = re.search(r"--subplot (\S+)", command).group(1)

        if '-' in layout:
            layout = layout.split('-')
            layout = [row.strip() for row in layout]
            layout = '\n'.join(layout)

        print ("> Subplot Layout:", layout)

        fig, axd = plt.subplot_mosaic(layout, figsize=figsize)
        process_subplots(commands, layout)
        plt.tight_layout()
        save_plot(command)
       
    else:
        plt.figure(figsize=figsize)
        process_plots(commands)
        plt.tight_layout()
        save_plot(command)

    plt.show()


exit()