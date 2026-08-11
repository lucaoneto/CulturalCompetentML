#!/usr/bin/env python
"""
Dataset Feature Extraction and Inter/Intra-Class Distance Examination.

This script extracts deep feature embeddings using a pre-trained ResNet50 model 
on the target dataset (Lamps or Carpets). It computes:
1. ResNet50 average-pooled feature embeddings for each image sample.
2. Intra-class Euclidean distances (measuring feature dispersion within culture-label clusters).
3. Inter-class Euclidean distances between cluster centroids.
4. Class Separation Ratio (Mean Inter / Mean Intra distance).
5. Global Silhouette Score across culture-label clusters.
6. Serializes analysis metrics to a JSON report (results_lamp_<lamp>.json).

Author: Enzo Ubaldo Petrocco
"""

import sys
import os
import json
from collections import defaultdict
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Insert parent directory for module imports
sys.path.insert(1, "../")
from Utils.FileManager.FileManager import FileManagerClass
from Processing.processing import ProcessingClass

# ==============================================================================
# 1. LOAD PRETRAINED RESNET50 EMBEDDING EXTRACTOR
# ==============================================================================
print("[Dataset Analysis] Initializing ResNet50 feature extractor (ImageNet weights, global average pooling)...")
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# ==============================================================================
# 2. PIPELINE SETUP & DATA LOADING
# ==============================================================================
basePath = "./KMeans/"
lamp = 1  # 1 = Lamps dataset, 0 = Carpets dataset

# Set standard image dimensions based on dataset type
if lamp == 1:
    IMAGE_SIZE = 120
else:
    IMAGE_SIZE = 200

BATCH_SIZE = 32

print(f"[Dataset Analysis] Loading dataset (Lamp={lamp}, Image Size={IMAGE_SIZE}x{IMAGE_SIZE})...")
procObj = ProcessingClass(
    shallow=0,
    lamp=lamp,
    basePath=basePath,
)
procObj.dataobj.prepare(
    standard=0,
    culture=0,
    percent=1.0,  # Use 100% of dataset samples for distance examination
    shallow=0,
    val_split=0.01,
    test_split=0.99,
    n=1000,
    adversarial=0,
    imbalanced=0,
)

X = np.array([])
y = np.array([])

# Group images into culture-label combinations (e.g. Culture 0 Label 0, Culture 0 Label 1...)
class_names = {}
for i, xt in enumerate(procObj.dataobj.Xt):
    print(f"  Culture set {i}: {len(xt)} images loaded.")
    X_c = np.asarray(xt)
    # Create combined culture-label target labels (2 * culture_index + class_label)
    y_c = np.asarray(np.add(2 * i, np.asarray(procObj.dataobj.yt[i])[:, 3]))
    y_c_uniques = np.unique(y_c)
    print(f"  Unique cluster labels in culture set {i}: {y_c_uniques}")
    for j, label in enumerate(y_c_uniques):
        class_names[2 * i + j] = f"Culture_{i}_label:{j}"

    if len(X) == 0:
        X = X_c
        y = y_c
    else:
        X = np.vstack([X, X_c])
        y = np.concatenate([y, y_c])

print(f"[Dataset Analysis] Total images loaded: {len(X)} across {len(class_names)} target clusters.")

# ==============================================================================
# 3. DEEP FEATURE EMBEDDING EXTRACTION
# ==============================================================================
features = []
print("[Dataset Analysis] Extracting ResNet50 feature embeddings for all samples...")
for X_img in tqdm(X, desc="Feature Extraction"):
    img = cv2.resize(X_img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = base_model.predict(img_array, verbose=0)
    features.append(embedding.flatten())

X_features = np.array(features)

# ==============================================================================
# 4. FEATURE STANDARDIZATION
# ==============================================================================
X_scaled = StandardScaler().fit_transform(X_features)

# ==============================================================================
# 5. INTRA-CLASS DISTANCE COMPUTATION
# ==============================================================================
print("\n--- Intra-Class Distances (Feature Variance Within Clusters) ---")
intra_distances = {}

for cls_idx, cls_name in enumerate(class_names):
    cls_feats = X_scaled[y == cls_idx]

    if len(cls_feats) < 2:
        intra_distances[cls_name] = np.nan
        print(f"  {cls_name}: N/A (insufficient images)")
        continue

    # Pairwise Euclidean distances within class
    dists = cdist(cls_feats, cls_feats, metric="euclidean")
    mean_dist = dists[np.triu_indices_from(dists, k=1)].mean()
    intra_distances[cls_name] = mean_dist
    print(f"  {cls_name}: {mean_dist:.4f}")

mean_intra = np.nanmean(list(intra_distances.values()))

# ==============================================================================
# 6. INTER-CLASS DISTANCE COMPUTATION (CENTROID DISTANCES)
# ==============================================================================
print("\n--- Inter-Class Distances (Distances Between Cluster Centroids) ---")
centroids = {}
for cls_idx, cls_name in enumerate(class_names):
    centroids[cls_name] = X_scaled[y == cls_idx].mean(axis=0)

inter_distances = []
inter_distances_dict = defaultdict(dict)
for i, cls_i in enumerate(class_names):
    for j, cls_j in enumerate(class_names):
        if j > i:
            dist = np.linalg.norm(centroids[cls_i] - centroids[cls_j])
            inter_distances.append(dist)
            inter_distances_dict[cls_i][cls_j] = dist
            print(f"  {cls_i} <-> {cls_j}: {dist:.4f}")

mean_inter = np.mean(inter_distances)

# ==============================================================================
# 7. SEPARATION RATIO & SILHOUETTE SCORE
# ==============================================================================
separation_ratio = mean_inter / mean_intra

print("\n========================================================")
print("              DATASET QUALITY SUMMARY REPORT            ")
print("========================================================")
print(f" Mean Intra-Class Distance  : {mean_intra:.4f}")
print(f" Mean Inter-Class Distance  : {mean_inter:.4f}")
print(f" Separation Ratio (Inter/Intra): {separation_ratio:.4f}")

if len(np.unique(y)) > 1:
    sil = silhouette_score(X_scaled, y)
    print(f" Global Silhouette Score    : {sil:.4f}")
else:
    sil = float('nan')
    print(" Global Silhouette Score    : N/A")

# ==============================================================================
# 8. SAVE ANALYSIS METRICS TO JSON
# ==============================================================================
results = {
    "lamp": lamp,
    "dataset": "lamp_dataset" if lamp == 1 else "carpet_dataset",
    "total_images": int(len(y)),
    "num_classes": int(len(np.unique(y))),
    "mean_intra_class_distance": float(mean_intra),
    "mean_inter_class_distance": float(mean_inter),
    "separation_ratio": float(separation_ratio),
    "silhouette_score": float(sil) if not np.isnan(sil) else None,
    "class_names": class_names,
    "intra_distances": {k: float(v) if not np.isnan(v) else None for k, v in intra_distances.items()},
    "inter_distances": {k: {kk: float(vv) for kk, vv in vv_dict.items()} for k, vv_dict in inter_distances_dict.items()}
}

output_filename = f"results_lamp_{lamp}.json"
with open(output_filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Dataset Analysis] Metrics successfully saved to: {output_filename}")