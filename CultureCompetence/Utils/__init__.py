"""
Utils Package - Utility Modules

This package provides utility functionality for data management, preprocessing, results analysis,
and visualization across the CultureAwarenessTest framework.

Modules:
    - Data: Data loading and preprocessing utilities
        * DataClass: Main data loader for images per culture/label
        * PreprocessingClass: Data augmentation (classical and diffusion-based)
        * Path definitions for lamp and carpet datasets
    
    - FileManager: Result logging and file management
        * FileManagerClass: CSV-based result logging and file operations
    
    - Preprocessing: Data augmentation techniques
        * Classical augmentation (rotation, noise, brightness, zoom)
        * Diffusion-based augmentation
        * Adversarial perturbation generation
    
    - Results: Evaluation metrics and statistics
        * ResultsClass: Computation of accuracy, precision, recall, F1-score per culture
        * Confusion matrix analysis
        * Per-culture performance metrics
    
    - Visualizer: Plotting and visualization utilities
        * Confusion matrix plots
        * GradCAM visualizations
        * Performance charts
    
    - Debug: Debugging utilities
"""
