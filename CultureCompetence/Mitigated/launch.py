#!/usr/bin/env python
"""
Main Execution Script for Bias Mitigation Experiments.

This script serves as the primary launcher for evaluating and comparing 
Standard vs. Culture-Mitigated Deep Learning models across multiple datasets 
(Lamps and Carpets) under varying experimental configurations.

Configurations control:
- Standard baseline vs. Mitigated model architecture
- Target majority culture vs. minority cultures
- Data augmentation strategy (Classical augmentation, Diffusion synthetic generation)
- Adversarial robustness evaluation (PGD attacks)

Author: Enzo Ubaldo Petrocco
"""

import sys
import os
import gc
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

# Insert parent directory to access top-level modules (Processing, Model, Utils)
sys.path.insert(1, "../")
from Processing.processing import ProcessingClass

# ==============================================================================
# GPU CONFIGURATION & MEMORY MANAGEMENT
# ==============================================================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Select target GPU device ID

# Set virtual GPU memory limit (in MB) to avoid allocating entire VRAM
memory_limit = 6000
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
        )
        print(f"[GPU Setup] Allocated {memory_limit} MB limit on GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"[GPU Setup Warning] {e}")

# ==============================================================================
# HYPERPARAMETERS & EXPERIMENT SETTINGS
# ==============================================================================
percent = 0.05         # Fraction of minority culture data retained (5% minority ratio)
n = 1000               # Number of samples per class/culture block
g_aug = 0.1            # Augment intensity / Gaussian noise strength
ep = 0.2               # Adversarial perturbation epsilon for PGD attack
basePath = "./try3/"   # Output directory for saving model checkpoints and results
verbose_param = 1      # Verbosity level (0=silent, 1=detailed)

# ==============================================================================
# EXPERIMENT CONFIGURATION MAPPING TABLE
# ==============================================================================
# Tuple signature: (standard, lamp, culture, diffusion, only_min, parify, augment)
# - standard: 1 = Standard Control Model, 0 = Bias-Mitigated Model
# - lamp:     1 = Lamps Dataset, 0 = Carpets Dataset
# - culture:  Majority culture index (0, 1, or 2)
# - diffusion: 1 = Enable Diffusion synthetic data generation, 0 = Disabled
# - only_min:  1 = Generate diffusion data ONLY for minority cultures, 0 = All
# - parify:    1 = Enable culture-balanced batch sampling, 0 = Disabled
# - augment:   1 = Enable classical image data augmentation, 0 = Disabled
# Note: Rule requirement: If diffusion == 1, then augment MUST be 1.

todo_configs = [
    # --- STD Control Group (Standard ResNet models without mitigation) ---
    (1, 0, 0, 0, 0, 0, 0), # Standard Model -> Carpets Dataset, Culture 0 Majority
    (1, 0, 1, 0, 0, 0, 0), # Standard Model -> Carpets Dataset, Culture 1 Majority
    (1, 0, 2, 0, 0, 0, 0), # Standard Model -> Carpets Dataset, Culture 2 Majority
    (1, 1, 0, 0, 0, 0, 0), # Standard Model -> Lamps Dataset,   Culture 0 Majority
    (1, 1, 1, 0, 0, 0, 0), # Standard Model -> Lamps Dataset,   Culture 1 Majority
    (1, 1, 2, 0, 0, 0, 0), # Standard Model -> Lamps Dataset,   Culture 2 Majority
]

# Reverse configuration list to prioritize specific test orders if needed
todo_configs = todo_configs[::-1]

# ==============================================================================
# EXPERIMENT EXECUTION LOOP
# ==============================================================================
for i in range(2):  # Run 2 independent random seed iterations for statistical stability
    for cls_div in [0, 1]:  # Class division split options (binary class separation)
        # Seed random number generators for reproducibility
        seed_val = int(datetime.now().timestamp())
        random.seed(seed_val)
        tf.random.set_seed(seed_val)
        
        for std, lp, cult, diff, omin, par, aug in todo_configs:
            print(f"\n========================================================")
            print(f"Running Experiment [Iter {i} | ClsDiv {cls_div}]:")
            print(f"  Standard Model: {std} | Dataset (Lamp={lp}) | Majority Culture: {cult}")
            print(f"  Diffusion: {diff} | OnlyMinority: {omin} | Augment: {aug}")
            print(f"========================================================")
            
            # Instantiate pipeline processor for target dataset
            procObj = ProcessingClass(
                shallow=0,          # 0 = Deep Learning (ResNet architecture)
                lamp=lp,            # Dataset selector (1=Lamps, 0=Carpets)
                gpu=False,          # CPU/GPU processing delegation handled via TF config
                memory_limit=memory_limit,
                basePath=basePath,
            )

            # Step 1: Preprocess data and train model
            procObj.process(
                standard=std,
                type="DL",
                verbose_param=verbose_param,
                culture=cult,
                percent=percent,
                n=n,
                augment=0,
                gaug=0,
                adversary=1,       # Include adversarial PGD training evaluation
                eps=ep,
                class_division=cls_div,
                imbalanced=0, 
                diffusion=0,
            )

            # Step 2: Evaluate model performance and log results
            procObj.test(
                standard=std,
                culture=cult,
                augment=0,
                gaug=0,
                adversary=0,
            )

            # Step 3: Resource Cleanup to prevent GPU/RAM memory leaks
            procObj.partial_clear(basePath)
            del procObj
            gc.collect()
            tf.keras.backend.clear_session()

print("\n--------------------------------------------------------")
print("All targeted bias mitigation experiments completed successfully.")
print("--------------------------------------------------------")
