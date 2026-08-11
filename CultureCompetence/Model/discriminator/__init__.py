"""
Discriminator Package - Culture Bias Detection

This package implements discriminator models for detecting whether trained classifiers
leak cultural information in their predictions.

Classes:
    - Discriminator: SVM-based culture classifier

Purpose:
    Tests if a trained model's decision boundary is correlated with culture by training
    a secondary classifier to predict culture from model outputs.

Functionality:
    - Binary classification: One culture vs. all others
    - SVM-based discrimination
    - Per-culture accuracy scoring
    - Bias measurement metrics

Usage:
    1. Get predictions from trained model
    2. Train discriminator on predictions + culture labels
    3. High discriminator accuracy = high bias leakage
    4. Low discriminator accuracy = low bias (good mitigation)

This provides empirical evidence of whether bias mitigation is effective.
"""
