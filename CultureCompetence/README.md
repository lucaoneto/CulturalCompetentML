# Culturally Competent ML

This repository is a machine learning research framework designed to investigate, measure, and mitigate cultural bias in deep learning models for image classification. The repository contains implementation scripts for training baseline control models (e.g., standard ResNet), bias-mitigated architectures using a multi-task learning inspired regularizer, diffusion-based, and adversarial data generation.

This repository contains also the two datasets (LAMPS and CARPETS) for testing the Cultural Competence of a model and the related code. 

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

---

## Key Features

- **Bias Detection**: Implementation of the Cultural Incompetence (CIC) metric.
- **Multi-task-learning-inspired Mitigation**: Regularized objective function designed to minimize the distance among the weight across cultural groups.
- **Synthetic Augmentation**: Denoising Diffusion Probabilistic Models (DDPM) to synthesize representative samples for underrepresented cultures.
- **Adversarial Robustness Evaluator**: Tests resilience under culture-specific Projected Gradient Descent (PGD) perturbations.

---

## Installation & Environment Setup

This repository contains two options for setting up your environment depending on your system's hardware capabilities:

### Option A: Local GPU Acceleration (TensorFlow 2.10 — Recommended for Windows)
This setup installs **TensorFlow 2.10.1** and **Keras 2.10.0**, which is the final version supporting native GPU acceleration on Windows (without requiring WSL2). 

- **Conda Environment Setup (Includes CUDA/cuDNN)**:
  ```bash
  conda env create -f environment_tf210_gpu.yml
  conda activate windows-tf210-gpu
  ```
- **Pip Virtual Environment Setup**:
  ```bash
  python -m venv .venv
  # Windows:
  .venv\Scripts\Activate.ps1
  # Linux:
  source .venv/bin/activate
  
  pip install -r requirements_tf210_gpu.txt
  ```

### Option B: CPU-Only / Alternative Setup (TensorFlow 2.14)
This configuration uses **TensorFlow 2.14.0** and **Keras 2.14.0**. This option is suitable for CPU-only environments or systems utilizing Linux / WSL2 for GPU acceleration.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux:
source .venv/bin/activate

pip install -r requirements_tf214_cpu.txt
```

---

## Project Structure

```
CultureAwarenessTest/
├── Model/                      # Model implementations
│   ├── standard/               # Standard baselines (ResNet50V2, SVM, RFC)
│   ├── mitigated/              # Mitigated architectures with weight variance regularization
│   ├── diffusion/              # Denoising Diffusion Probabilistic Models (DDPM)
│   ├── discriminator/          # Classifiers to detect latent culture signatures
│   ├── adversarial/            # PGD adversarial attack generators
│   └── GeneralModel.py         # Base model wrapper class
├── Processing/
│   └── processing.py           # Core pipeline orchestration (ProcessingClass)
├── Utils/                      # Helper libraries
│   ├── Data/                   # Data load routines, culture splits, paths
│   ├── FileManager/            # Logging directories & outputs management
│   ├── Preprocessing/          # Standalone dataset prep (create_ds)
│   ├── Results/                # CM metrics & results aggregation
│   └── Visualizer/             # Confusion matrix & performance plotting
├── Mitigated/                  # Primary launch scripts
│   ├── launch.py               # Main deep learning pipeline experiment loop
│   └── launch_shallow.py       # Main shallow baseline pipeline loop
├── DatasetAnalysis/            # Feature analysis and clustering scripts
│   ├── examinate.py            # Deep feature embedding & distance examiner
│   └── k-means.py              # KMeans unsupervised clustering ($K=6$)
├── environment_tf210_gpu.yml   # Conda GPU environment specification
├── requirements_tf210_gpu.txt  # Pip GPU requirements (includes tf-explain)
├── requirements_tf214_cpu.txt  # Pip CPU requirements
└── LAUNCH_GUIDE.md             # Detailed execution reference guide
```

---

## Basic Usage

Detailed instructions for running each subsystem are located in [`LAUNCH_GUIDE.md`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/LAUNCH_GUIDE.md). Below are the primary quickstart commands:

### 1. Run Core Benchmarking Experiments
To run the standard control vs. bias-mitigated ResNet training loop:
```bash
python Mitigated/launch.py
```

### 2. Run Shallow Machine Learning Baselines
To evaluate SVM and Random Forest baselines:
```bash
python Mitigated/launch_shallow.py
```

### 3. Extract Dataset Features & Compute Cluster Distance Reports
To extract ResNet embeddings and output intra/inter-class distances to JSON:
```bash
python DatasetAnalysis/examinate.py
```

---

## ProcessingClass Configurations

To run custom pipelines programmatically, initialize and run `ProcessingClass` inside Python:

```python
from Processing.processing import ProcessingClass

# Initialize processor
procObj = ProcessingClass(
    shallow=0,            # 0 = Deep Learning (ResNet), 1 = Shallow (SVM/RFC)
    lamp=1,               # 1 = Lamps dataset, 0 = Carpets dataset
    gpu=True,             # Enable GPU acceleration
    memory_limit=6000,    # GPU virtual memory limit in MB
    basePath="./results/" # Log destination directory
)

# Run full pipeline
procObj.process(
    standard=0,           # 0 = Mitigated, 1 = Control
    culture=0,            # Culture index 0 as majority culture
    percent=0.05,         # Keep 5% minority data ratio
    augment=1,            # Enable classical augmentation
    diffusion=1,          # Enable diffusion augmentation
    only_minority_diffusion=1,
    parify_batches_diffusion=1
)

# Evaluate results
procObj.test(standard=0, culture=0)
```

---

## Troubleshooting

1. **GPU Out-Of-Memory (OOM)**:
   Ensure you allocate virtual memory limit parameters before training, as set up in [`Mitigated/launch.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Mitigated/launch.py):
   ```python
   tf.config.experimental.set_virtual_device_configuration(
       gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)] # Decrease to fit GPU
   )
   ```
2. **Missing Dataset Directories**:
   Verify raw input files are mapped to the directory targets defined inside [`Utils/Data/deep_paths.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Utils/Data/deep_paths.py) and [`Utils/Data/shallow_paths.py`](file:///C:/Users/Utente/Desktop/CultureAwarenessTest/Utils/Data/shallow_paths.py).
