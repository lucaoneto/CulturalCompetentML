#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import cv2
from tf_explain.utils.display import grid_display, heatmap_display
from tf_explain.utils.saver import save_rgb
import gc
from keras.models import Model
import random
from datetime import datetime
random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())


class GeneralModelClass:
    """
    Base class providing common functionality for all model implementations.
    
    This class serves as middleware for collecting and standardizing common actions across
    different model types (standard, mitigated, adversarial, discriminator).
    
    Provides unified interfaces for:
        - Inference (__call__, test)
        - Evaluation (get_model_stats)
        - Explanation (explain with GradCAM)
        - Model persistence (save_model, get_model_from_weights)
    
    Attributes:
        model: The underlying Keras/TensorFlow model
        standard (int): 0 for mitigated models, 1 for standard models
        n_cultures (int): Number of cultures in dataset
        adversarial (int): Flag for adversarial mode
        imbalanced (int): Flag for imbalanced learning mode
    """
    def __init__(self, standard=0, n_cultures=3, adversarial=0, imbalanced=0) -> None:
        """
        Initialize the base model class.
        
        Args:
            standard (int): 0=mitigated (per-culture outputs), 1=standard (single output)
            n_cultures (int): Number of cultural groups in the dataset
            adversarial (int): Whether adversarial robustness is being tested
            imbalanced (int): Whether to use imbalanced learning strategies
        """
        self.model = Model()
        self.standard = standard
        self.n_cultures = n_cultures
        self.adversarial = adversarial
        self.imbalanced = imbalanced

    def __call__(self, X, out=-1):
        """
        Perform inference on input samples.
        
        For mitigated models: Returns predictions for a specific culture output.
        For standard models: Returns binary predictions directly.
        
        Args:
            X (array-like): Input samples for inference (images or flattened features)
            out (int): Output index for mitigated models (-1 for default/all outputs)
        
        Returns:
            array: Model predictions
            - Mitigated: Predictions for specified culture output
            - Standard: Binary predictions (0 or 1)
        """
        if self.model != None:
                # Get raw predictions from the model
                res = self.model.predict(np.asarray(X, dtype='int32'))
                if not self.standard:
                    # For mitigated models: extract specific culture output
                    # Model outputs shape: [batch_size, n_cultures, 1]
                    # Select output index and get probability values
                    res = np.asarray(res, dtype=np.float32)[out][:, 0]
                return res
        else:
            print("Try fitting the model before")
            return None

    def quantize(self, yF):
        """
        Convert continuous predictions to discrete class labels (0 or 1).
        
        For binary classification, applies a threshold of 0.5 to convert
        probability scores to class predictions.
        
        Note: The threshold is fixed at 0.5 because class imbalance is intra-class
        (within each cultural group) rather than inter-class, so optimal threshold
        remains at midpoint.
        
        Args:
            yF (array-like): Continuous predictions (floats) from model
        
        Returns:
            list: Quantized predictions as binary values (0 or 1)
        """
        values = []
        for y in yF:
            # Binary classification: threshold at 0.5
            if y > 0.5:
                values.append(1)
            else:
                values.append(0)
            gc.collect()
        return values

    def test(self, Xt, out=-1):
        """
        Evaluate model on test samples and return quantized predictions.
        
        Combines inference (__call__) and quantization (quantize) for complete
        evaluation workflow.
        
        Args:
            Xt (array-like): Test samples
            out (int): Output index for mitigated models (-1 for default)
        
        Returns:
            list: Quantized predictions (0 or 1) for test samples,
                  or None if model not fitted
        """
        if self.model:
            # Get raw predictions
            yF = self(Xt, out)
            # Quantize to class labels
            yFq = self.quantize(yF)
            gc.collect()
            return yFq
        else:
            gc.collect()
            print("Try fitting the model before")
            return None

    def get_model_stats(self, Xt, yT, out=-1, discriminator=0, j=-1):
        """
        Compute confusion matrix for model evaluation.
        
        Supports two evaluation modes:
        1. Standard classification: Compare predictions vs true labels
        2. Discriminator evaluation: Per-culture discrimination accuracy
        
        Args:
            Xt (array-like): Test samples
            yT (array-like): True labels/one-hot encoded labels
            out (int): Output index for mitigated models
            discriminator (int): 0=classification, 1=discriminator evaluation
            j (int): Unused parameter
        
        Returns:
            ndarray: Confusion matrix showing classification performance
                     shape: (n_classes, n_classes)
        """
        
        if discriminator == 0:
            # Standard classification evaluation
            yFq = self.test(Xt, out)
            if len(np.shape(yT)) > 1:
                if type(yT) == list:
                    yT = np.asarray(yT)
                if self.standard:
                    # Standard model: class label at index 1
                    if not self.adversarial:
                        yT = yT[:, 1]
                    else:
                        # Adversarial: class label at n_cultures index
                        yT = yT[:, self.n_cultures]
                else:
                    # Mitigated model: class label at n_cultures index
                    yT = yT[:, self.n_cultures]
                gc.collect()
            gc.collect()
            if yFq != None:
                cm = confusion_matrix(y_true=yT, y_pred=yFq)
                print(f"Confusion Matrix: {cm}")
                return cm
        else:
            # Discriminator mode: evaluate per-culture discrimination
            X = []
            for XC in Xt:
                X.extend(XC)
            y = []
            for yC in yT:
                y.extend(yC)
            # Get predictions from discriminator
            yF = self.model.predict(np.asarray(X, dtype='int32'))
            # Get predicted culture
            yF = np.argmax(yF, axis=1)
            y = np.asarray(y)
            # Extract true culture labels (one-hot)
            y = y[:, 0:self.n_cultures]
            y = np.argmax(y, axis=1)
            if yF.any() != None:
                cm = confusion_matrix(y_true=y, y_pred=yF)
                print(f"Confusion Matrix: {cm}")
                return cm
        
    def get_model_from_weights(self, path="./"):
        self.model = tf.keras.models.load_model(path)

    def save_model(self, path="./"):
        self.model.save(path)
        



    def explain(
        self,
        validation_data,
        class_index,
        layer_name=None,
        use_guided_grads=True,
        colormap=cv2.COLORMAP_VIRIDIS,
        image_weight=0.7,
    ):
        """
        Compute GradCAM for a specific class index.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            layer_name (str): Targeted layer for GradCAM. If no layer is provided, it is
                automatically infered from the model architecture.
            colormap (int): OpenCV Colormap to use for heatmap visualization
            image_weight (float): An optional `float` value in range [0,1] indicating the weight of
                the input image to be overlaying the calculated attribution maps. Defaults to `0.7`.
            use_guided_grads (boolean): Whether to use guided grads or raw gradients

        Returns:
            numpy.ndarray: Grid of all the GradCAM
        """
        images, _ = validation_data

        if layer_name is None:
            layer_name = self.infer_grad_cam_target_layer()

        outputs, grads = self.get_gradients_and_filters(images, layer_name, class_index, use_guided_grads)

        print(f"outputs in explain: {np.shape(outputs)}")
        print(f"grads in explain: {np.shape(grads)}")

        cams = self.generate_ponderated_output(outputs, grads)

        heatmaps = np.array(
            [
                # not showing the actual image if image_weight=0
                heatmap_display(cam.numpy(), image, colormap, image_weight)
                for cam, image in zip(cams, images)
            ]
        )

        grid = grid_display(heatmaps)

        return grid

    #@staticmethod
    def infer_grad_cam_target_layer(self):
        """
        Search for the last convolutional layer to perform Grad CAM, as stated
        in the original paper.

        Args:
            model (tf.keras.Model): tf.keras model to inspect

        Returns:
            str: Name of the target layer
        """
        for layer in reversed(self.model.layers):
            # Select closest 4D layer to the end of the network.
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError(
            "Model does not seem to contain 4D layer. Grad CAM cannot be applied."
        )

    #@staticmethod
    def get_gradients_and_filters(
        self, images, layer_name, class_index, use_guided_grads
    ):
        """
        Generate guided gradients and convolutional outputs with an inference.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            layer_name (str): Targeted layer for GradCAM
            class_index (int): Index of targeted class
            use_guided_grads (boolean): Whether to use guided grads or raw gradients

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
        """
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.layers[2].get_layer(layer_name).output, self.model.output]
        )
        print(f"layer name is {layer_name}")
        print(f"self.model.layers[2] = {self.model.layers[2]}")
        print(f"Created grad model with:\n inputs:{self.model.inputs};\n output:{[self.model.get_layer(layer_name).output, self.model.output]}")
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            print(f"input in get_gradients and filters: {np.shape(inputs)}")
            conv_outputs, predictions = grad_model(inputs)
            print(f"conv_outputs in get_gradients and filters: {np.shape(conv_outputs)}")
            print(f"predictions in get_gradients and filters: {np.shape(predictions)}")
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)

        if use_guided_grads:
            grads = (
                tf.cast(conv_outputs > 0, "float32")
                * tf.cast(grads > 0, "float32")
                * grads
            )

        return conv_outputs, grads

    #@staticmethod
    def generate_ponderated_output(self, outputs, grads):
        """
        Apply Grad CAM algorithm scheme.

        Inputs are the convolutional outputs (shape WxHxN) and gradients (shape WxHxN).
        From there:
            - we compute the spatial average of the gradients
            - we build a ponderated sum of the convolutional outputs based on those averaged weights

        Args:
            output (tf.Tensor): Target layer outputs, with shape (batch_size, Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (batch_size, Hl, Wl, Nf)

        Returns:
            List[tf.Tensor]: List of ponderated output of shape (batch_size, Hl, Wl, 1)
        """
        print(f"outputs inside function generated_ponderated_output = {np.shape(outputs)}")
        print(f"grads inside function generated_ponderated_output = {np.shape(grads)}")
        maps = [
            self.ponderate_output(output, grad)
            for output, grad in zip(outputs, grads)
        ]

        return maps

    #@staticmethod
    def ponderate_output(self, output, grad):
        """
        Perform the ponderation of filters output with respect to average of gradients values.

        Args:
            output (tf.Tensor): Target layer outputs, with shape (Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (Hl, Wl, Nf)

        Returns:
            tf.Tensor: Ponderated output of shape (Hl, Wl, 1)
        """

        print(f"output inside function ponderated_output = {output}")
        print(f"grad inside function ponderated_output = {grad}")

        weights = tf.reduce_mean(grad, axis=(0, 1))

        # Perform ponderated sum : w_i * output[:, :, i]
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

        return cam
    
    def save(self, img, outdir, name):
        cv2.imwrite(img, outdir + name + ".jpg")

    def test_gradcam(self,gradcam_layers, Xv, yv, out_dir):
        for name in gradcam_layers:
                    for class_index in range(2):
                        print(f"Shape of Xv is {np.shape(Xv)}")
                        print(f"Shape of yv is {np.shape(yv)}")
                        output = self.explain(validation_data=(Xv, yv),
                                                class_index=class_index,
                                                layer_name=name)
                        # Save output
                        self.save(output, out_dir, name)
