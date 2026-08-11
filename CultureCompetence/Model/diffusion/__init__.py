"""
Diffusion Models Package - Synthetic Data Generation

This package implements denoising diffusion probabilistic models (DDPM) for generating
synthetic images to augment minority culture datasets.

Classes:
    - DiffusionStandardModel: Implements DDPM for image generation

Features:
    - Denoising process for image generation
    - Targeted generation for minority cultures
    - Sequential noise removal (reverse diffusion)
    - Integration with training pipeline

Usage:
    1. Train diffusion model on existing images
    2. Generate synthetic images for minority cultures
    3. Use generated images for data augmentation
    4. Optionally balance batches using parify_batches

Options:
    - only_minority_diffusion: Generate only for non-majority cultures
    - parify_batches_diffusion: Sample balanced batches across cultures

This addresses data imbalance issues common in cultural datasets.
"""
