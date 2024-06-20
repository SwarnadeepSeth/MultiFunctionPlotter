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

## Requirements
Install the required packages using: pip install -r requirements.txt

## Usages:

### Command Line Interface
** Plot from CSV or Text File: mfp data.csv u 1:2 w lp title 'Sample Plot' xlabel 'x-Axis' xrange 0:10
** Plot a Mathematical Function
