"""
Mitigated Models Package - Bias Mitigation Strategies

This package implements models with custom bias mitigation techniques for reducing
cultural disparities in image classification.

Classes:
    - MitigatedModels: Main mitigated model class with custom regularization
    - MitigatedModels (advanced): Advanced mitigation strategies

Mitigation Strategies:
    - Custom Regularization: Minimizes variance of per-culture weight matrices
    - Culture-Inclusive Loss: Measures and reduces per-culture loss variance
    - Architecture: ResNet50V2 with modified final layers for multi-culture awareness

Key Parameters:
    - lambda_index: Regularization strength (0-20+)
    - culture: Majority culture ID
    - n_cultures: Number of cultures in dataset

Features:
    - Per-culture loss tracking
    - Custom weight regularization
    - Culture-aware training procedures
    - Evaluation metrics for bias reduction

The regularization objective minimizes:
    L_reg = λ * Σ ||w_i - mean(w)||²
where w_i are per-culture weight matrices.
"""
