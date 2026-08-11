"""
Adversarial Models Package - Robustness Testing

This package implements adversarial attack generators for testing model robustness
against culture-specific adversarial perturbations.

Classes:
    - AdversarialStandard: Implements PGD (Projected Gradient Descent) attacks

Attack Methods:
    - PGD (Projected Gradient Descent): Multi-step gradient-based perturbations
    - Culture-specific attacks: Perturbations targeting particular cultures
    - Epsilon-bounded perturbations: Controlled perturbation magnitude

Parameters:
    - epsilon: Maximum perturbation budget (pixel value range)
    - num_steps: Number of PGD iterations
    - step_size: Step size for gradient updates

Usage:
    1. Select epsilon (e.g., 0.1 for 10% perturbation)
    2. Generate adversarial examples from test samples
    3. Evaluate model on adversarial examples
    4. Compare robustness across cultures

This evaluates if models are equally robust to attacks on all cultures,
revealing biases in adversarial robustness.
"""
