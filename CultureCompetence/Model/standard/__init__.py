"""
Standard Models Package - Baseline Classification Models

This package contains baseline machine learning and deep learning models without
bias mitigation strategies.

Classes:
    - StandardModels: Main class implementing SVM, Random Forest, and ResNet50V2
    - StandardModels4GradCam: Extended version with GradCAM explanation capabilities

Model Types:
    - SVM: Support Vector Machine with linear/RBF kernels
    - RFC: Random Forest Classifier
    - DL: Deep Learning with ResNet50V2 (pre-trained on ImageNet)

Features:
    - Shallow learning (SVM, RFC) for feature-based classification
    - Deep learning (ResNet50V2) for end-to-end image classification
    - Optional imbalanced learning handling
    - Adversarial robustness testing

These models serve as baselines for comparison with mitigated versions.
"""
