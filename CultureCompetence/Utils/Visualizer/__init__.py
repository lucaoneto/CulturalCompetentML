"""
Visualizer Package - Plotting and Visualization

This package provides visualization utilities for result analysis and interpretation.

Classes:
    - VisualizerClass: Main visualization utilities

Visualization Types:
    - Confusion Matrices: Per-culture classification performance
    - Performance Charts: Accuracy, precision, recall per culture
    - GradCAM Heatmaps: Model attention visualization
    - Comparison Plots: Standard vs. mitigated models
    - Distribution Plots: Data and prediction distributions

Features:
    - Per-culture visualization
    - Model comparison plots
    - Feature importance visualization
    - Interactive and static plots
    - Export to various formats (PNG, SVG, PDF)

Usage:
    vizObj = VisualizerClass()
    vizObj.plot_confusion_matrix(cm, labels)
"""
