"""
Data Package - Data Loading and Management

This package handles all data loading, preprocessing, and management functionality.

Modules:
    - Data.py: DataClass for loading images per culture and label
    - deep_paths.py: Path configurations for deep learning datasets
    - shallow_paths.py: Path configurations for shallow learning datasets
    - PreprocessingClass: Data augmentation (rotation, noise, brightness, zoom)

Key Classes:
    - DataClass: Main data loader with culture-stratified splitting
    - PreprocessingClass: Augmentation techniques

Features:
    - Load images grouped by culture and class
    - Train/validation/test splitting with culture stratification
    - Classical augmentation (rotation, noise, brightness adjustments)
    - Data normalization and preprocessing
    - Support for both deep learning (RGB) and shallow learning (grayscale+flattened)

Culture Encoding:
    - One-hot encoding for deep learning models
    - Numerical encoding for shallow learning
"""
