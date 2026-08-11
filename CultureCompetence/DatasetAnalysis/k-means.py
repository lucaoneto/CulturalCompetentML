#!/usr/bin/env python
"""
K-Means Clustering Analysis for Culture-Awareness Evaluation.

This script performs unsupervised K-Means clustering (K=6) on raw image feature vector representations 
(flattened image arrays) across culture subsets to evaluate natural cluster formation vs. target culture labels.

Author: Enzo Ubaldo Petrocco
"""

import sys
import os
import random
from math import floor
from datetime import datetime
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Insert parent directory for module imports
sys.path.insert(1, "../")
from Utils.FileManager.FileManager import FileManagerClass
from Processing.processing import ProcessingClass

# ==============================================================================
# 1. SETUP & DATASET PREPARATION
# ==============================================================================
lamps = [0, 1]
basePath = "./KMeans/"
lamp = 0  # 0 = Carpet dataset, 1 = Lamp dataset

print(f"[K-Means Analysis] Initializing data loader for dataset (Lamp={lamp})...")
procObj = ProcessingClass(
    shallow=0,
    lamp=lamp,
    basePath=basePath,
)
procObj.dataobj.prepare(
    standard=0,
    culture=0,
    percent=1.0,
    shallow=0,
    val_split=0.1,
    test_split=0.2,
    n=1000,
    adversarial=0,
    imbalanced=0,
)

# ==============================================================================
# 2. FEATURE FLATTENING & K-MEANS FIT
# ==============================================================================
# Flatten training images from (N, H, W, C) to 2D (N, H*W*C)
X_flat = np.asarray(procObj.dataobj.X).reshape(np.shape(procObj.dataobj.X)[0], -1)
n_clusters = 6

print(f"[K-Means Analysis] Fitting KMeans (clusters={n_clusters}) on training set shape {X_flat.shape}...")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_flat)

print(f"Computed Cluster Centers Shape: {kmeans.cluster_centers_.shape}")

# ==============================================================================
# 3. PREDICTION & EVALUATION ON CULTURE SUBSETS
# ==============================================================================
for i in range(3):
    print(f"\n--- Culture Subset {i} Evaluation ---")
    Xt_flat = np.asarray(procObj.dataobj.Xt[i]).reshape(np.shape(procObj.dataobj.Xt[i])[0], -1)
    yt = np.asarray(procObj.dataobj.yt[i])
    
    predicted_labels = kmeans.predict(Xt_flat)
    print(f"Predicted Cluster Labels (first 20): {predicted_labels[:20]}")
    print(f"True Class Labels        (first 20): {yt[:20, 3] if yt.ndim > 1 else yt[:20]}")

print("\n[K-Means Analysis] Clustering evaluation finished.")