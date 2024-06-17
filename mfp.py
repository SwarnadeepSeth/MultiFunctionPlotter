#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, math
import re
import json

class PlotCommand:
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
        
        if csv_file_match:
            self.file = csv_file_match.group(1)
        if text_file_match:
            self.file = text_file_match.group(1)
        # if using_match:
        #     self.x_col = int(using_match.group(1)) - 1
        #     self.y_col = int(using_match.group(2)) - 1

        if using_match:
            self.x_col = int(using_match.group(1))
            self.y_col = int(using_match.group(2)) - 1
            if self.x_col > 0:
                self.x_col -= 1
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
            'points': 'o',
            'p': 'o',
            'linespoints': '-o',
            'lp': '-o',
        }
        return style_map.get(self.plot_command.style, '-')
            
    def plot(self):
        if self.plot_command.file:
            if self.plot_command.x_col == 0:
                x_data = self.data.index
            else:
                x_data = self.data.iloc[:, self.plot_command.x_col]

        if self.plot_command.style == 'hist':
            print ("Plotting Histogram")
            sns.histplot(self.data.iloc[:, self.plot_command.x_col], color=self.plot_command.linecolor, label=self.plot_command.legend)
        
        elif self.plot_command.style == 'kde':
            sns.kdeplot(self.data.iloc[:, self.plot_command.x_col], color=self.plot_command.linecolor, label=self.plot_command.legend)

        elif self.plot_command.style == 'box':
            sns.boxplot(data=self.data.iloc[:, self.plot_command.x_col], color=self.plot_command.linecolor, label=self.plot_command.legend)

        elif self.plot_command.style == 'violin':
            sns.violinplot(data=self.data.iloc[:, self.plot_command.x_col], color=self.plot_command.linecolor, label=self.plot_command.legend)

        else:
            plt.plot(
                x_data,
                self.data.iloc[:, self.plot_command.y_col],
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

    def function_plot(self):

        x = np.linspace(self.plot_command.xrange[0], self.plot_command.xrange[1], 50)

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
            sns.histplot(self.data.iloc[:, self.plot_command.x_col], color=self.plot_command.linecolor, ax=axd[f'{index}'], label=self.plot_command.legend)
        
        elif self.plot_command.style == 'kde':
            sns.kdeplot(self.data.iloc[:, self.plot_command.x_col], color=self.plot_command.linecolor, ax=axd[f'{index}'], label=self.plot_command.legend)

        elif self.plot_command.style == 'box':
            sns.boxplot(data=self.data.iloc[:, self.plot_command.x_col], color=self.plot_command.linecolor, ax=axd[f'{index}'], label=self.plot_command.legend)

        elif self.plot_command.style == 'violin':
            sns.violinplot(data=self.data.iloc[:, self.plot_command.x_col], color=self.plot_command.linecolor, ax=axd[f'{index}'], label=self.plot_command.legend)

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

def process_func_plots(commands):
    for command in commands:
        try:
            plot_command = PlotCommand(command)
            plotter = Plotter(plot_command)
            plotter.function_plot()
        except Exception as e:
            print(f"Error: {e}")

# ================================================================================================= 
if __name__ == "__main__":

    figsize = (12, 6)

    if sys.argv[1] == 'plot.json':
        print ("Reading from plot.json file...")
        with open('plot.json') as f:
            data = json.load(f)

        plt.figure(figsize=figsize)
        process_plots_json(data)
        plt.tight_layout()
        plt.show()
        exit()

    if len(sys.argv) > 1:
        command = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in sys.argv[1:])

    else:
        command = input("Enter mfp command: ")

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

    if '--save' in sys.argv:
        # Path to save the plot
        path = re.search(r"--save (\S+)", command).group(1)
        plt.savefig(path)
        print (f"Plot saved as {path}.")
       
    elif 'func:' in sys.argv:
        print ("Function Plotting")
        process_func_plots(commands)
        
    else:
        plt.figure(figsize=figsize)
        process_plots(commands)
        plt.tight_layout()


    plt.show()


exit()