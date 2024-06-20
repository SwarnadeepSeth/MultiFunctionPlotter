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

### Command Line Options and Arguments:

- **Plot from CSV or Text File**: <br>
-- mfp data.csv u 0:4 w l lc red lw 2 title 'Sample Plot' xlabel 'Days ' ylabel 'Close ' xrange 100:1000 --save example1.png <br>
Arguments: <br>
-- plot styles --> line plot: line or l, points: points or p, line and points: lp <br>
-- columns to plot --> using col1:col2 or u col1:col2 <br>
-- line color --> lc color_name <br>
-- line width --> lw line_width <br>
-- set x axis range --> xrange lower_x_limit:higher_x_limit <br>
-- set y axis range --> yrange lower_y_limit:higher_y_limit <br>
-- set title --> title 'Sample Plot' <br>
-- set x axis label --> xlabel 'X_Axis ' <br>
-- set y axis label --> ylabel 'Y_Axis ' <br>
-- saving the plot --> --save filename <br>
-- Please add an extra space after the axis labels and title if you are using a single word to ensure the title/axis labels render properly. <br> 

<br>

- **Plot a Mathematical Function**: <br>
-- mfp func: 'f(x;a=2;b=3.1) = a*x**2*np.exp(-b*x)' w l xrange 1:3 lc red legend 'function_plot' --save example2.png <br>
Arguments: <br>
-- calling the function plotter --> use func: to plot a function. <br>
-- specifying the constants in the function --> use ; to separate the variable (x) and the constants. <br>
-- specifying the x axis range --> always needs the xrange values. <br>
-- describing the mathematical functions --> add np. before expression to evaluate the function properly. <br>
-- use quotation to enclose the functional expression.
-- lengeds --> lengend 'function_name'

<br>

- **Plot Histograms/KDE**: <br>
- mfp data.csv u 4:0 w hist bin 50 --save example3.png <br>
Arguments: <br>
-- number of bins: bin num_bins <br>
-- histogram is calculated on col1 only.

<br>

- **Plot KDE**: <br>
-- mfp data.csv u 4:0 w kde --save example4.png <br>
-- kde is calculated on col1 only.

<br>

- **Plot Violin**: <br>
-- mfp data.csv u 4:0 w violin --save example5.png <br>
-- violin plot is generated for col1 only.

<br>

- **Multiple Plots**: <br>
-- mfp data.csv u 1:3 w l lc magenta lw 2 legend 'plot1', data.csv u 1:4 w l lc green legend 'plot2' xlabel 'Day ' ylabel 'Y_data ' --save example6.png <br>
Arguments: <br>
-- use comma to separate each plots.

<br>

- **Subplots**: <br>
-- mfp data.csv u 1:3 w l lc magenta lw 2 legend 'plot1', data.csv u 1:4 w l lc green, data.csv u 4:0 w violin, data.csv u 4:0 w kde lc maroon lw 3, data.csv u 2:3 w stars xlabel 'col2 ' ylabel 'col3 ' --subplot AAB-CDD -save example7.png <br>
Arguments: <br>
-- subplot specification --> --subplot mosaic_style

<br>

- **Plot from json file**: <br>
-- mfp plot.json <br>
Arguments: <br>
-- MFP always saves the arguments as 'plot.json' file for future use.
-- use plot.json to re-plot the same figure in future. 
-- subplot save is not currently supported.

<br>

- **Forecast your data using FB prophet**: <br>
-- mfp forecast <br>
Arguments: <br>
-- Data Syntax: data.csv u 1:2 <br>
-- Split Percentage: 0.8 <br>
-- Show FFT: True <br>
-- Show Decompose: True <br>
-- Daily Seasonality: True <br>
-- Frequency: D <br>