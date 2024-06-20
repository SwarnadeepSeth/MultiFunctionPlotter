# MultiFunctionPlotter (MFP)

MultiFunctionPlotter (MFP) is a versatile Python-based tool for creating a variety of plots from CSV and text files. It supports line plots, histograms, KDE plots, box plots, violin plots, and custom mathematical function plots. MFP keeps the simplicity and style of gnuplot-like commands while leveraging Python packages to produce professional-quality graphs.

## Version: 1.0.0

## Features

- **Support for CSV and Text Files**: Load data from CSV, TXT, or DAT files.
- **Multiple Plot Styles**: Create line plots, histograms, KDE plots, box plots, and violin plots.
- **Custom Function Plots**: Plot mathematical functions directly.
- **Subplot Layouts**: Organize multiple plots in a grid layout.
- **Configuration via Command Line**: Use simple commands to specify plot configurations.
- **JSON Configuration**: Save and load plot configurations from JSON files.
- **Flexible Syntax**: Fluid syntax layout.

## Requirements
Install the required packages using: pip install -r requirements.txt

## Usages:

### Command Line Interface

- **Plot from CSV or Text File**: <br>
-- mfp data.csv u 0:4 w l lc red lw 2 title 'Sample Plot' xlabel 'Days ' ylabel 'Close ' xrange 100:1000 --save example1.png <br>
-- Please add an extra space after the axis labels and title if you are using a single word to render the title/axis labels properly. <br><br>
- **Plot a Mathematical Function**: <br>
-- mfp func: 'f(x;a=2;b=3.1) = a*x**2*np.exp(-b*x)' w l xrange 1:3 lc red legend 'function_plot' --save example2.png <br>
-- Use numpy to evaluate the function. <br>
