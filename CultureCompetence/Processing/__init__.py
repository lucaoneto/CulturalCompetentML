"""
Processing Package - Pipeline Orchestration

This package handles the main data processing and training pipeline for the CultureAwarenessTest framework.

Modules:
    - processing: ProcessingClass for end-to-end pipeline orchestration

Key Features:
    - Data preparation with culture stratification
    - Classical augmentation (rotation, noise, brightness)
    - Diffusion-based synthetic data generation
    - Culture-balanced batch sampling (parify_batches)
    - Multi-culture model training and evaluation
    - Integration with all model types

Pipeline Stages:
    1. Data loading and splitting
    2. Augmentation (optional)
    3. Model training
    4. Testing and evaluation
    5. Result logging and visualization
"""
