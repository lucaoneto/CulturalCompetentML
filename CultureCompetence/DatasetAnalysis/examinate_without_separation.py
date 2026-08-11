#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

import cv2

sys.path.insert(1, "../")
from Utils.FileManager.FileManager import FileManagerClass
from Processing.processing import ProcessingClass
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from collections import defaultdict
from tqdm import tqdm
import json

# =========================
# CONFIG
# =========================


# =========================
# LOAD PRETRAINED MODEL
# =========================
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# =========================
# LOAD IMAGE PATHS
# =========================
basePath = "./KMeans/"
lamp = 1
if lamp == 1:
    IMAGE_SIZE = 120
else:
    IMAGE_SIZE = 200

BATCH_SIZE = 32
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
            val_split=0.01,
            test_split=0.99,
            n=1000,
            adversarial=0,
            imbalanced=0,
        )

X = np.array([])
y = np.array([])

class_names = {}
for i, xt in enumerate(procObj.dataobj.Xt):
    print(f"Number of images in culture set: {len(xt)}")
    X_c = np.asarray(xt)
    print(f"Shape of procObj.dataobj.yt[i][:,3]: {np.shape(np.asarray(procObj.dataobj.yt[i])[:,3])}")
    print(f"Shape of np.multiply(procObj.dataobj.yt[i],i): {np.shape(np.multiply(procObj.dataobj.yt[i],i))}")
    print(f"Shape of np.add(np.multiply(procObj.dataobj.yt[i],i), procObj.dataobj.yt[i]): {np.shape(np.add(np.multiply(procObj.dataobj.yt[i],i), procObj.dataobj.yt[i]))}")
    y_c = np.asarray(procObj.dataobj.yt[i])[:,3]

    if len(X) == 0:
        X = X_c
        y = y_c
    else:
        X = np.vstack([X, X_c])
        y = np.concatenate([y, y_c])

y_c_uniques = np.unique(y_c)
print(f"Unique labels in culture set {i}: {y_c_uniques}")
for j, label in enumerate(y_c_uniques):
    class_names[2*i+j] = f"Label:{j}"

    

print("Total images loaded:", len(X))
print("Class names:", class_names)
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")


# =========================
# FEATURE EXTRACTION
# =========================
features = []

print("Extracting image embeddings...")
for X_img in tqdm(X):
    img = cv2.resize(X_img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = base_model.predict(img_array, verbose=0)
    features.append(embedding.flatten())

X = np.array(features)

# =========================
# STANDARDIZE FEATURES
# =========================
X = StandardScaler().fit_transform(X)

# =========================
# INTRA-CLASS DISTANCES
# =========================
print("\nIntra-class distances:")
intra_distances = {}

for cls_idx, cls_name in enumerate(class_names):
    cls_feats = X[y == cls_idx]

    if len(cls_feats) < 2:
        intra_distances[cls_name] = np.nan
        print(f"  {cls_name}: N/A (only one image)")
        continue

    dists = cdist(cls_feats, cls_feats, metric="euclidean")
    mean_dist = dists[np.triu_indices_from(dists, k=1)].mean()
    intra_distances[cls_name] = mean_dist

    print(f"  {cls_name}: {mean_dist:.4f}")

mean_intra = np.nanmean(list(intra_distances.values()))

# =========================
# INTER-CLASS DISTANCES (CENTROIDS)
# =========================
print("\nInter-class distances (centroids):")

centroids = {}
for cls_idx, cls_name in enumerate(class_names):
    centroids[cls_name] = X[y == cls_idx].mean(axis=0)

inter_distances = []

for i, cls_i in enumerate(class_names):
    for j, cls_j in enumerate(class_names):
        if j > i:
            dist = np.linalg.norm(centroids[cls_i] - centroids[cls_j])
            inter_distances.append(dist)
            print(f"  {cls_i} ↔ {cls_j}: {dist:.4f}")

mean_inter = np.mean(inter_distances)

# =========================
# SEPARATION RATIO
# =========================
separation_ratio = mean_inter / mean_intra

print("\n=========================")
print("DATASET QUALITY SUMMARY")
print("=========================")
print(f"Mean intra-class distance : {mean_intra:.4f}")
print(f"Mean inter-class distance : {mean_inter:.4f}")
print(f"Separation ratio          : {separation_ratio:.4f}")

# =========================
# SILHOUETTE SCORE
# =========================
if len(np.unique(y)) > 1:
    sil = silhouette_score(X, y)
    print(f"Silhouette score          : {sil:.4f}")
else:
    print("Silhouette score          : N/A (single class)")


# =========================
# SAVE RESULTS
# =========================
results = {
    "lamp": lamp,
    "dataset": "lamp_dataset" if lamp == 1 else "carpet_dataset",
    "total_images": len(y),
    "num_classes": len(np.unique(y)),
    "mean_intra_class_distance": float(mean_intra),
    "mean_inter_class_distance": float(mean_inter),
    "separation_ratio": float(separation_ratio),
    "silhouette_score": float(sil) if len(np.unique(y)) > 1 else None,
    "class_names": class_names,
    "intra_distances": {k: float(v) if not np.isnan(v) else None for k, v in intra_distances.items()}
}

output_filename = f"results_lamp_{lamp}_without_c_sep.json"
with open(output_filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_filename}")