"""
FileManager Package - Result Logging and File Operations

This package handles file management and result logging using CSV format.

Classes:
    - FileManagerClass: File operations and CSV logging

Features:
    - Create and manage experiment output directories
    - Log results to CSV files
    - Organize results by configuration (model type, culture, threshold)
    - Track experiment metadata
    - File naming conventions for reproducibility

Usage:
    fileObj = FileManagerClass(base_path)
    fileObj.save_results(results_dict)
"""
