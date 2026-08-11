"""
Model Package - ML Model Implementations

This package contains all machine learning model implementations for the CultureAwarenessTest framework.

Modules:
    - GeneralModel: Base class providing common functionality for all models
    - standard: Baseline models (SVM, Random Forest, ResNet50V2)
    - mitigated: Bias-mitigated models using custom regularization
    - diffusion: Denoising diffusion probabilistic models for synthetic data generation
    - discriminator: Culture classification models for bias detection
    - adversarial: Adversarial attack generators (PGD perturbations)
    - GAN: Generative Adversarial Network implementations
    - SMOTE: Synthetic Minority Over-sampling Technique implementations
    - teacher_student: Teacher-student model architectures

All models inherit from GeneralModelClass and implement standardized interfaces for:
    - Inference (__call__, test)
    - Evaluation (get_model_stats)
    - Explanation (explain with GradCAM)
"""
