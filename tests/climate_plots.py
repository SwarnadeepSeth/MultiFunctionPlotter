#!/usr/bin/env python3
"""
Climate Change Data Plotting Script
Uses MultiFunctionPlotter (mfp) to generate plots with standard deviation
"""

from multifunctionplotter.mfp import process_plots


def plot_co2_subplots():
    """Plot CO2 concentrations (Mauna Loa & South Pole) in separate subplots"""
    commands = [
        "climet_change_subplots.csv using 1:0 with lines lc blue title 'Mauna Loa CO2' xlabel 'Date' ylabel 'CO2 (ppm)'",
        "climet_change_subplots.csv using 1:2 with lines lc red title 'South Pole CO2' xlabel 'Date' ylabel 'CO2 (ppm)'"
    ]
    process_plots(commands)
    print("Created: climet_co2_subplots.png")


def plot_heat_content():
    """Plot Heat content with ±2σ error bands"""
    commands = [
        "climet_change_subplots.csv using 1:3 with errorshade yerr_col 5:6 lc steelblue title 'Heat Content'",
        "climet_change_subplots.csv using 1:3 with lines lc steelblue title 'Heat Content Mean'"
    ]
    process_plots(commands)
    print("Created: climet_heat_content.png")


def plot_temp_anomaly():
    """Plot Temperature anomaly with ±2σ error bands"""
    commands = [
        "climet_change_subplots.csv using 2:9 with errorshade yerr_col 8:10 lc tomato title 'Temp Anomaly ±2σ'",
        "climet_change_subplots.csv using 2:9 with lines lc tomato title 'Temp Anomaly Mean'"
    ]
    process_plots(commands)
    print("Created: climet_temp_anomaly.png")


def plot_all_subplots():
    """Plot all 4 metrics in a 2x2 grid"""
    commands = [
        "climet_change_subplots.csv using 1:0 with lines lc blue title 'Mauna Loa CO2' xlabel 'Date' ylabel 'CO2 (ppm)'",
        "climet_change_subplots.csv using 1:2 with lines lc red title 'South Pole CO2' xlabel 'Date' ylabel 'CO2 (ppm)'",
        "climet_change_subplots.csv using 1:3 with errorshade yerr_col 5:6 lc steelblue title 'Heat Content ±2σ' xlabel 'Date' ylabel 'Heat Content'",
        "climet_change_subplots.csv using 2:9 with errorshade yerr_col 8:10 lc tomato title 'Temp Anomaly ±2σ' xlabel 'Date' ylabel 'Temp Anomaly'"
    ]
    process_plots(commands)
    print("Created: climate_change_all_subplots.png")


if __name__ == "__main__":
    print("Generating Climate Change Plots...")
    print("=" * 40)
    
    plot_co2_subplots()
    #plot_heat_content()
    #plot_temp_anomaly()
    #plot_all_subplots()
    
    print("=" * 40)
    print("All plots generated successfully!")
