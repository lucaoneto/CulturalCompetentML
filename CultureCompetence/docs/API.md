# API Documentation

This document provides detailed API documentation for the key classes and modules in the CultureAwarenessTest project.

## Core Classes

### GeneralModelClass

Located in `Model/GeneralModel.py`

The base class for all model implementations, providing common inference and evaluation functionality.

#### `__init__(self, standard=0, n_cultures=3, adversarial=0, imbalanced=0)`
Initializes the model instance.

**Parameters:**
- `standard` (int): 0 for mitigated models, 1 for standard models
- `n_cultures` (int): Number of cultures in the dataset
- `adversarial` (int): Flag for adversarial training
- `imbalanced` (int): Flag for imbalanced learning

#### `__call__(self, X, out=-1)`
Performs inference on input samples.

**Parameters:**
- `X`: Input samples
- `out` (int): Output index to select (-1 for all outputs)

**Returns:** List of model predictions

#### `test(self, Xt, out=-1)`
Tests model quality on a set of samples.

**Parameters:**
- `Xt`: Test samples
- `out` (int): Desired output index

**Returns:** Quantized predictions

#### `get_model_stats(self, Xt, yT, out=-1, discriminator=0, j=-1)`
Computes confusion matrix for model evaluation.

**Parameters:**
- `Xt`: Test samples
- `yT`: True labels
- `out` (int): Output index
- `discriminator` (int): Flag for discriminator evaluation

**Returns:** Confusion matrix

#### `explain(self, validation_data, class_index, layer_name=None, use_guided_grads=True, colormap=cv2.COLORMAP_VIRIDIS, image_weight=0.7)`
Computes GradCAM explanations for model predictions.

**Parameters:**
- `validation_data`: Tuple of (images, labels)
- `class_index` (int): Target class index
- `layer_name` (str): Target layer name (auto-inferred if None)
- `use_guided_grads` (bool): Whether to use guided gradients
- `colormap`: OpenCV colormap for visualization
- `image_weight` (float): Weight for overlaying original image

**Returns:** Grid of GradCAM visualizations

### ProcessingClass

Located in `Processing/processing.py`

Main pipeline orchestrator for data processing, training, and evaluation.

#### `__init__(self, shallow, lamp, gpu=False, memory_limit=2700, basePath="./")`
Initializes the processing pipeline.

**Parameters:**
- `shallow` (bool): Use shallow learning (SVM, etc.) vs deep learning
- `lamp` (bool): Use lamp dataset (True) or carpet dataset (False)
- `gpu` (bool): Enable GPU usage
- `memory_limit` (int): GPU memory limit in MB
- `basePath` (str): Base path for outputs

#### `prepare_data(self, standard, culture, percent=0, val_split=0.2, test_split=0.2, n=1000, augment=0, gaug=0.01, adversarial=0, imbalanced=0, discriminator=0, diffusion=0, aug=0, weights=0, only_minority_diffusion=0, parify_batches_diffusion=0, plt_imgs=True)`
Prepares training data with various preprocessing options.

**Parameters:**
- `standard` (int): 0 for mitigated, 1 for standard models
- `culture` (int): Majority culture index
- `percent` (float): Percentage of minority culture data
- `val_split` (float): Validation set proportion
- `test_split` (float): Test set proportion
- `n` (int): Maximum images per culture/class
- `augment` (int): Enable classical augmentation
- `gaug` (float): Augmentation gain parameter
- `adversarial` (int): Enable adversarial training
- `imbalanced` (int): Enable imbalanced learning
- `discriminator` (int): Enable discriminator training
- `diffusion` (int): Enable diffusion-based augmentation
- `only_minority_diffusion` (int): Generate only for minority cultures
- `parify_batches_diffusion` (int): Enable culture-balanced batching

#### `process(self, standard, culture, percent=0.05, augment=0, diffusion=0, only_minority_diffusion=0, parify_batches_diffusion=0, adversarial=0, imbalanced=0, discriminator=0, weights=0, shallow=None, lamp=None, gpu=None, memory_limit=None, basePath=None)`
Main processing pipeline: data preparation, model training, and evaluation.

**Parameters:** (same as prepare_data plus training parameters)

#### `test(self, standard, culture, discriminator=0)`
Tests trained model on test data.

**Parameters:**
- `standard` (int): Model type
- `culture` (int): Culture index
- `discriminator` (int): Discriminator flag

### DataClass

Located in `Utils/Data/Data.py`

Handles data loading and preprocessing for different cultures and classes.

#### Key Methods:
- `prepare()`: Loads and splits data by culture
- `load_images()`: Loads images from specified paths
- `get_culture_data()`: Retrieves data for specific culture

### MitigatedModels

Located in `Model/mitigated/mitigated_models.py`

Implements culture-bias mitigation through custom regularization.

#### Key Features:
- Custom regularization loss to minimize culture-specific weight variances
- Culture-inclusive loss (CIC) for measuring bias
- Modified ResNet50V2 architecture

#### `fit(self, train_data, val_data, epochs=15, batch_size=32)`
Trains the mitigated model.

**Parameters:**
- `train_data`: Training data tuple (X, y)
- `val_data`: Validation data tuple (X, y)
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size

### DiffusionStandardModel

Located in `Model/diffusion/diffusion_standard.py`

Implements denoising diffusion probabilistic model for synthetic data generation.

#### Key Methods:
- `generate()`: Generates synthetic images
- `train()`: Trains the diffusion model

### Discriminator

Located in `Model/discriminator/discriminator.py`

SVM-based culture discrimination classifier for bias detection.

### AdversarialStandard

Located in `Model/adversarial/adversarial.py`

Implements PGD adversarial attacks for robustness testing.

## Utility Classes

### FileManagerClass

Located in `Utils/FileManager/FileManager.py`

Handles CSV-based result logging and file management.

### ResultsClass

Located in `Utils/Results/Results.py`

Computes evaluation metrics and statistics.

### VisualizerClass

Located in `Utils/Visualizer/visualizer.py`

Provides plotting utilities for confusion matrices and other visualizations.

## Configuration Parameters

### Model Types
- `standard = 0`: Mitigated model with regularization
- `standard = 1`: Standard model without mitigation

### Datasets
- `lamp = 1`: Lamps dataset
- `lamp = 0`: Carpets dataset

### Cultures
- Integer indices (0-2) representing different cultural origins

### Augmentation Options
- `augment = 1`: Enable classical augmentation
- `diffusion = 1`: Enable diffusion-based generation
- `only_minority_diffusion = 1`: Generate only for minority cultures
- `parify_batches_diffusion = 1`: Culture-balanced batch sampling

## Data Flow

1. **Data Loading**: `DataClass` loads images per culture/label
2. **Preprocessing**: Train/val/test splits with culture stratification
3. **Augmentation**: Optional classical and diffusion-based augmentation
4. **Training**: Model training with optional culture balancing
5. **Evaluation**: Testing with metrics computation and visualization

## Error Handling

Common exceptions:
- `Exception("Carpet Problem has not been tackled in shallow learning")`: Carpets not supported in shallow mode
- Memory errors: Reduce `memory_limit` or use CPU
- CUDA errors: Check TensorFlow-GPU compatibility

## Dependencies

- TensorFlow 2.10.1
- Keras 2.10.0
- scikit-learn 1.6.1
- OpenCV
- NumPy, Pandas, Matplotlib
- tf-explain (for GradCAM)