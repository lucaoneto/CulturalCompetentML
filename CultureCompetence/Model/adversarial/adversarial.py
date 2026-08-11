#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"

import os
import gc
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
from math import ceil
import math

# Import your base class
from Model.GeneralModel import GeneralModelClass

# Set seeds for reproducibility
seed_val = int(math.floor(datetime.now().timestamp()))
random.seed(seed_val)
np.random.seed(seed_val)
tf.random.set_seed(seed_val)
os.environ['PYTHONHASHSEED'] = str(seed_val)

# Suppress annoying TF warnings about while_loops on the A100
import logging
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AdversarialStandard(GeneralModelClass):
    def __init__(self, type="RESNET", points=50, kernel="linear", verbose_param=0,
                 learning_rate=1e-3, epochs=15, batch_size=1, weights=None,
                 imbalanced=0, class_division=0, only_imb_imgs=0,
                 save_discriminator=0, path='./', culture=0):
        
        GeneralModelClass.__init__(self, standard=1, adversarial=1, imbalanced=imbalanced)
        
        self.type = type
        self.points = points
        self.kernel = kernel
        self.verbose_param = verbose_param
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_division = class_division
        self.only_imb_imgs = only_imb_imgs
        self.save_discriminator = save_discriminator
        self.path = path
        self.culture = culture
        self.weights = weights if weights is not None else np.ones(self.n_cultures)
        self.model = None
        self.shape = None

    @tf.function
    def generate_adversarial_image_pgd(self, img, lbl, model, epsilon=0.1, alpha=0.005, num_iter=40):
        """
        Calculates PGD perturbations. 
        Uses training=False to bypass Random Layers (Flip, Rotation, Noise).
        """
        # Ensure we have a batch dimension (1, H, W, C)
        img_batch = tf.expand_dims(img, axis=0) 
        lbl_batch = tf.expand_dims(lbl, axis=0)
        x_adv = tf.identity(img_batch) 

        # Cast epsilon/alpha to float32 to match image tensor type
        eps_255 = tf.cast(epsilon * 255.0, dtype=tf.float32)
        alpha_255 = tf.cast(alpha * 255.0, dtype=tf.float32)

        for _ in range(num_iter):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                # CRITICAL: training=False ensures random layers are dormant
                prediction = model(x_adv, training=False)
                loss = tf.keras.losses.categorical_crossentropy(lbl_batch, prediction)
            
            gradients = tape.gradient(loss, x_adv)
            x_adv = x_adv + alpha_255 * tf.sign(gradients)
            
            # Project back onto the epsilon ball and valid image range
            x_adv = tf.clip_by_value(x_adv, img_batch - eps_255, img_batch + eps_255)
            x_adv = tf.clip_by_value(x_adv, 0.0, 255.0) 
            
        return x_adv

    def plot_culture_transition(self, original, adversarial, culture_idx, main_class, index=0):
        """
        Saves comparison plots. Clips values to [0,1] to avoid Matplotlib warnings.
        """
        div_status = "ClassDiv_ON" if self.class_division else "ClassDiv_OFF"
        specific_path = os.path.join(self.path, div_status, f"MainClass_{int(main_class)}", f"SourceCulture_{culture_idx}")
        
        if not os.path.exists(specific_path):
            os.makedirs(specific_path, exist_ok=True)
            
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # np.clip avoids "Clipping input data to the valid range" warnings
        axes[0].imshow(np.clip(original / 255.0, 0, 1))
        axes[0].set_title(f"Original (Culture {culture_idx})")
        axes[0].axis("off")
        
        axes[1].imshow(np.clip(adversarial / 255.0, 0, 1))
        axes[1].set_title(f"Adv (Main Class {int(main_class)})")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(specific_path, f"sample_{index}.jpg"))
        plt.close()

    def remove_data_aug(self, model):
        """
        Keras handles augmentation removal automatically during .predict(training=False).
        Manual graph reconstruction is disabled to prevent 'Empty batch_outputs' errors.
        """
        return model

    def LearningAdversarially(self, TS, VS, aug, path="./", eps=0.1, **kwargs):
        # 1. Shuffle
        idx_t = np.random.permutation(len(TS[0]))
        TS = ([TS[0][i] for i in idx_t], [TS[1][i] for i in idx_t])
        if VS:
            idx_v = np.random.permutation(len(VS[0]))
            VS = ([VS[0][i] for i in idx_v], [VS[1][i] for i in idx_v])

        if self.imbalanced:
            TS = self.ImbalancedTransformation(TS)

        adversarial_models = []

        # 2. Phase 1: Train Cultural Discriminators
        if self.class_division:
            for j in range(2): 
                tempTS = ([TS[0][i] for i in range(len(TS[0])) if TS[1][i][self.n_cultures] == j],
                          [TS[1][i] for i in range(len(TS[1])) if TS[1][i][self.n_cultures] == j])
                tempVS = ([VS[0][i] for i in range(len(VS[0])) if VS[1][i][self.n_cultures] == j],
                          [VS[1][i] for i in range(len(VS[1])) if VS[1][i][self.n_cultures] == j])
                
                if len(tempTS[0]) > 0:
                    self.ModelSelection(TS=tempTS, VS=tempVS, aug=aug, adv=1, eps=eps, path=path, **kwargs)
                    adversarial_models.append(self.model)
                else:
                    adversarial_models.append(None)
                
                self.model = None # Clear reference for next loop
                gc.collect()
        else:
            self.ModelSelection(TS=TS, VS=VS, aug=aug, adv=1, eps=eps, path=path, **kwargs)
            adversarial_models = self.model

        self.adversarial_model = adversarial_models

        # 3. Generate Adversarial Samples
        original_len = len(TS[0])
        # Generate for 25% of the dataset
        for i in range(original_len // 4):
            culture_vec = TS[1][i][0:self.n_cultures]
            main_label = TS[1][i][self.n_cultures]
            
            target_model = adversarial_models[int(main_label)] if self.class_division else adversarial_models
            if target_model is None: continue
            
            original_img = TS[0][i].copy()
            lbl = tf.cast(culture_vec, dtype=tf.float32)
            
            # Generate perturbed image
            adv_img_batch = self.generate_adversarial_image_pgd(
                tf.cast(original_img, tf.float32), lbl, target_model, epsilon=eps
            )
            
            # Detach from GPU graph with .numpy().copy()
            adv_img = adv_img_batch[0].numpy().copy() 
            
            TS[0].append(adv_img)
            TS[1].append(TS[1][i])
            
            if i < 40: # Visual audit for the first 20 samples
                self.plot_culture_transition(
                    original=original_img, 
                    adversarial=adv_img, 
                    culture_idx=np.argmax(culture_vec), 
                    main_class=main_label, 
                    index=i
                )

        # 4. Phase 2: Final Training for Binary Classification
        print("Starting Phase 2: Final Binary Classification...")
        self.ModelSelection(TS=TS, VS=VS, aug=aug, adv=0, eps=eps, path=path, **kwargs)
        tf.keras.backend.clear_session()

    def ModelSelection(self, TS, VS, aug, batches=[32], lrs=[1e-3], fine_lrs=[1e-5], epochs=15, fine_epochs=5, nDropouts=[0.3], adv=0, **kwargs):
        best_loss = np.inf
        best_params = {'b': batches[0], 'lr': lrs[0], 'f_lr': fine_lrs[0], 'drop': nDropouts[0]}

        for b in batches:
            for lr in lrs:
                for f_lr in fine_lrs:
                    for drop in nDropouts:
                        loss = self.DL(TS, VS, aug=aug, batch_size=b, lr=lr, fine_lr=f_lr, 
                                       epochs=epochs, fine_epochs=fine_epochs, nDropout=drop, adv=adv, **kwargs)
                        if loss < best_loss:
                            best_loss = loss
                            best_params = {'b': b, 'lr': lr, 'f_lr': f_lr, 'drop': drop}
        
        # Train final version with best params
        self.DL(TS, VS, aug=aug, batch_size=best_params['b'], lr=best_params['lr'], 
                fine_lr=best_params['f_lr'], epochs=epochs, fine_epochs=fine_epochs, 
                nDropout=best_params['drop'], val=False, adv=adv, **kwargs)

    def ImbalancedTransformation(self, TS):
        newX, newY = [], []
        for img, label in zip(TS[0], TS[1]):
            class_idx = np.argmax(label[:self.n_cultures])
            repeat = ceil(1.0 / self.weights[class_idx])
            for _ in range(repeat):
                newX.append(img.copy())
                newY.append(label)
        return (newX, newY)

    def DL(self, TS, VS, aug=False, batch_size=32, lr=1e-3, fine_lr=1e-5, epochs=1, fine_epochs=1, nDropout=0.2, g=0.1, val=True, adv=0, **kwargs):
        tf.keras.backend.clear_session()
        
        shape = np.shape(TS[0][0])
        self.shape = shape
        monitor = "val_loss" if val else "loss"

        def map_fn(img, y):
            # If adv=1: target is Culture Vector. If adv=0: target is Binary Scalar.
            label = y[0:self.n_cultures] if adv else tf.expand_dims(y[self.n_cultures], axis=-1)
            return tf.cast(img, tf.float32), label

        train_ds = tf.data.Dataset.from_tensor_slices((TS[0], TS[1])).map(map_fn).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((VS[0], VS[1])).map(map_fn).batch(batch_size).prefetch(tf.data.AUTOTUNE) if val else None

        base_model = keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_shape=shape)
        base_model.trainable = False

        inputs = keras.Input(shape=shape)
        x = inputs
        if aug:
            x = layers.RandomFlip("horizontal")(x)
            x = layers.RandomRotation(0.01)(x)
            x = layers.GaussianNoise(g)(x)
        
        x = layers.Rescaling(1./255.0)(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(nDropout)(x)
        
        output_dim = self.n_cultures if adv else 1
        activation = "softmax" if adv else "sigmoid"
        outputs = layers.Dense(output_dim, activation=activation)(x)
        
        self.model = keras.Model(inputs, outputs)
        
        loss_fn = keras.losses.CategoricalCrossentropy() if adv else keras.losses.BinaryCrossentropy()
        self.model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss_fn, metrics=['accuracy'])
        
        callbacks = [EarlyStopping(monitor=monitor, patience=3, restore_best_weights=True), 
                     ReduceLROnPlateau(monitor=monitor, factor=0.5)]
        
        self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=self.verbose_param)

        # Fine-tuning phase
        base_model.trainable = True
        self.model.compile(optimizer=keras.optimizers.Adam(fine_lr), loss=loss_fn, metrics=['accuracy'])
        history = self.model.fit(train_ds, epochs=fine_epochs, validation_data=val_ds, callbacks=callbacks, verbose=self.verbose_param)

        return history.history[monitor][-1]

    def fit(self, TS, VS=None, aug=0, g=0.1, eps=0.3, out_dir="./", **kwargs):
        if self.type in ["DL", "RESNET"]:
            self.LearningAdversarially(TS, VS, aug=aug, g=g, path=out_dir, eps=eps, **kwargs)