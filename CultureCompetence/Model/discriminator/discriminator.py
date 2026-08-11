#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys
import time

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(1, "../")
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import layers
from Model.GeneralModel import GeneralModelClass
import gc
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from datetime import datetime
#import keras_cv
from PIL import Image
from IPython.display import Image as IImage

random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())


class Discriminator(GeneralModelClass):
    def __init__(
        self,
        type="SVC",
        points=50,
        kernel="linear",
        verbose_param=0,
        learning_rate=1e-3,
        epochs=15,
        batch_size=1,
        weights=None,
        imbalanced=0,
        class_division=0,
        save_discriminator=0
    ):
        """
        Initialization function for modeling standard ML models.
        We have narrowed the problems to image classification problems.
        I have implemented SVM (with linear and gaussian kernel) and Random Forest, using scikit-learn library;
        ResNet using Tensorflow library.
        :param type: selects the algorithm "SVC", "RFC" and "RESNET" are possible values.
        :param points: n of points in gridsearch for SVC and RFC
        :param kernel: type of kernel for SVC: "linear" and "gaussian" are possible values.
        :param verbose_param: if enabled, the program logs more information
        :param learning_rate: hyperparameter for DL
        :param epochs: hyperparameter for DL
        :param batch_size: hyperparameter for DL
        """
        GeneralModelClass.__init__(
            self, standard=1, adversarial=1, imbalanced=imbalanced
        )
        self.type = type
        self.points = points
        self.kernel = kernel
        self.verbose_param = verbose_param
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = np.ones(self.n_cultures)
        self.class_division = class_division
        self.save_discriminator = save_discriminator
        self.reweighting = False
        if weights is not None:
            self.weights = weights

    def remove_data_aug(self, model :keras.Model):  
        input = keras.Input(shape=self.shape)
        x = model.layers[2](input)
        x = model.layers[3](x)
        x = model.layers[4](x)
        x = model.layers[5](x)
        y = model.layers[6](x)
        truncated_model = keras.Model(inputs = input, outputs = y)   
        truncated_model.summary()
        return truncated_model

    def LearningAdversarially(
        self,
        TS,
        VS,
        aug,
        show_imgs=False,
        batches=[4],
        lrs=[1e-2, 1e-3, 1e-4, 1e-5],
        fine_lrs=[1e-5],
        epochs=30,
        fine_epochs=10,
        nDropouts=[0.4],
        g=0.1,
        save=False,
        path="./",
        eps=0.1,
    ):
        #if aug:
        #    self.reweighting = True
        class_division = self.class_division

        if self.imbalanced:
                TS = self.ImbalancedTransformation(TS)
                VS = self.ImbalancedTransformation(VS) 
    

        if class_division:
            adversarial_model = []
            print(f"ADVERSARIAL USING CLASS DIVISION")
            for j in range(2):
                tempX = [
                    TS[0][i]
                    for i in range(len(TS[0]))
                    if TS[1][i][self.n_cultures] == j
                ]
                tempY = [
                    TS[1][i]
                    for i in range(len(TS[1]))
                    if TS[1][i][self.n_cultures] == j
                ]
                tempTS = (tempX, tempY)
                tempX = [
                    VS[0][i]
                    for i in range(len(VS[0]))
                    if VS[1][i][self.n_cultures] == j
                ]
                tempY = [
                    VS[1][i]
                    for i in range(len(VS[1]))
                    if VS[1][i][self.n_cultures] == j
                ]
                tempVS = (tempX, tempY)
                self.ModelSelection(
                    TS=tempTS,
                    VS=tempVS,
                    aug=aug,
                    show_imgs=show_imgs,
                    batches=batches,
                    lrs=lrs,
                    fine_lrs=fine_lrs,
                    epochs=epochs,
                    fine_epochs=fine_epochs,
                    nDropouts=nDropouts,
                    g=g,
                    save=save,
                    path=path,
                )
                if aug:
                    adversarial_model.append(self.remove_data_aug(self.model))
                else:
                    adversarial_model.append(self.model)

                if self.save_discriminator:
                    self.model.save(path=path + f'/class_discriminator={i}')
                self.model = None
            
        else:
            self.ModelSelection(
                TS=TS,
                VS=VS,
                aug=aug,
                show_imgs=show_imgs,
                batches=batches,
                lrs=lrs,
                fine_lrs=fine_lrs,
                epochs=epochs,
                fine_epochs=fine_epochs,
                nDropouts=nDropouts,
                g=g,
                save=save,
                path=path,
            )
            if aug:
                adversarial_model = self.remove_data_aug(self.model)
            else:
                adversarial_model = self.model

            if self.save_discriminator:
                    self.model.save(path=path + f'/class_discriminator={i}')

        tf.keras.backend.clear_session()

    def ModelSelection(
        self,
        TS,
        VS,
        aug,
        show_imgs=False,
        batches=[32],
        lrs=[1e-2, 1e-3, 1e-4, 1e-5],
        fine_lrs=[1e-5],
        epochs=30,
        fine_epochs=10,
        nDropouts=[0.4],
        g=0.1,
        save=False,
        path="./",
    ):

        if self.verbose_param:
            tf.get_logger().setLevel(4)
        best_loss = np.inf
        for b in batches:
            for lr in lrs:
                for fine_lr in fine_lrs:
                    for nDropout in nDropouts:
                        self.model = None
                        gc.collect()
                        tf.get_logger().info(
                            f"Training with: batch_size={b}, lr={lr}, fine_lr={fine_lr}, nDropout={nDropout}"
                        )
                        sys.stdout.write("\r")
                        loss = self.DL(
                            TS,
                            VS,
                            aug,
                            b,
                            lr,
                            fine_lr,
                            epochs,
                            fine_epochs,
                            nDropout,
                            g=g
                        )

                        if loss < best_loss:
                            best_loss = loss
                            best_bs = b
                            best_lr = lr
                            best_fine_lr = fine_lr
                            best_nDropout = nDropout

                        self.model = None
                        gc.collect()

        print(
            f"Best loss:{best_loss}, best batch size:{best_bs}, best lr:{best_lr}, best fine_lr:{best_fine_lr}, best_dropout:{best_nDropout}"
        )
        list(TS).append(VS)
        self.DL(
            TS,
            None,
            aug,
            best_bs,
            best_lr,
            best_fine_lr,
            epochs,
            fine_epochs,
            best_nDropout,
            val=False,
            g=g
        )
        tf.keras.backend.clear_session()
        if save:
            self.save(path)

    def ImbalancedTransformation(self, TS):
        newX = []
        newY = []
        X = TS[0]
        Y = TS[1]
        for i in range(len(X)):
            img = X[i]
            label = Y[i]
            for j in range(
                int(1 / self.weights[label.index(1.0)])
            ):  # I use the inverse of the total proportion for augmenting the dataset
                newX.append(np.asarray(img))  
                newY.append(np.asarray(label))

        del TS
        tf.keras.backend.clear_session()
        return (newX, newY)

    def DL(
        self,
        TS,
        VS,
        aug=False,
        batch_size=32,
        lr=1e-3,
        fine_lr=1e-5,
        epochs=1,
        fine_epochs=1,
        nDropout=0.2,
        g=0.1,
        val=True,
    ):
        with tf.device("/gpu:0"):
            shape = np.shape(TS[0][0])
            self.shape = shape

            if val:
                monitor_val = "val_loss"
            else:
                monitor_val = "loss"

            data_augmentation = keras.Sequential(
                [
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.01),
                    layers.GaussianNoise(g),
                    tf.keras.layers.RandomBrightness(0.01),
                    layers.RandomZoom(g, g),
                    layers.Resizing(shape[0], shape[1]),
                ]
            )

            validation_generator = None

            train_generator = tf.data.Dataset.from_tensor_slices(TS)
            # train_generator = tf.random.shuffle(int(train_generator.cardinality()/batch_size))
            
            
        
            train_generator = train_generator.map(
                lambda img, y: (
                    data_augmentation(img, training=aug),
                    y[0 : self.n_cultures],
                )
            )
            train_generator = (
                train_generator.cache().batch(batch_size).prefetch(buffer_size=10)
            )
            if val:
                validation_generator = tf.data.Dataset.from_tensor_slices(VS)

            
                validation_generator = validation_generator.map(
                    lambda img, y: (
                        data_augmentation(img, training=aug),
                        y[0 : self.n_cultures],
                    )
                )
                
                validation_generator = (
                    validation_generator.cache()
                    .batch(batch_size)
                    .prefetch(buffer_size=10)
                )

            # MODEL IMPLEMENTATION
            base_model = keras.applications.ResNet50V2(
                weights="imagenet",  # Load weights pre-trained on ImageNet.
                input_shape=shape,
                include_top=False,
            )  # Do not include the ImageNet classifier at the top.

            # Freeze the base_model
            base_model.trainable = False

            # Create  model on top
            inputs = keras.Input(shape=shape)
            scale_layer = keras.layers.Rescaling(scale=1 / 255.0)
            if aug:
                x = data_augmentation(inputs)  # Apply random data augmentation
                x = scale_layer(x)
            else:
                x = scale_layer(inputs)

            x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dropout(nDropout)(x)  # Regularize with dropout
            
            outputs = keras.layers.Dense(3, activation="softmax")(x)
            
            self.model = keras.Model(inputs, outputs)

            bcemetric = keras.losses.CategoricalCrossentropy(from_logits=True)
            train_acc_metric = keras.metrics.CategoricalAccuracy()
           

            lr_reduce = ReduceLROnPlateau(
                monitor=monitor_val,
                factor=0.2,
                patience=5,
                verbose=self.verbose_param,
                min_lr=1e-9,
            )
            early = EarlyStopping(
                monitor=monitor_val,
                min_delta=0.001,
                patience=10,
                verbose=self.verbose_param,
                mode="auto",
            )
            callbacks = [early, lr_reduce]

            # MODEL TRAINING
            self.model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss=bcemetric,
                metrics=[train_acc_metric],
                # run_eagerly=True
            )
            #if self.reweighting:
            #    class_weights = dict(enumerate(np.ones(self-n_cultures)/self.weights))
            #else:
            #    classweights = dict(enumerate(np.ones(self.n_cultures)))

            self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                verbose=self.verbose_param,
                callbacks=callbacks,
                #class_weight=class_weights
            )

            # FINE TUNING
            base_model.trainable = True
            # self.model.summary()

            self.model.compile(
                optimizer=keras.optimizers.Adam(fine_lr),
                loss=bcemetric,
                metrics=[train_acc_metric],
                # run_eagerly=True
            )

            history = self.model.fit(
                train_generator,
                epochs=fine_epochs,
                validation_data=validation_generator,
                verbose=self.verbose_param,
                callbacks=callbacks,
                #class_weight=class_weights
            )
            tf.keras.backend.clear_session()
            return history.history[monitor_val][-1]


    def fit(
        self,
        TS,
        VS=None,
        aug=0,
        g=0.1,
        eps=0.3,
        gradcam=0,
        out_dir="./",
        complete=0,
    ):
        """
        General function for implementing model selection
        :param TS: training set
        :param VS: validation set
        :param adversary: if enabled, adversarial training is enabled
        :param eps: if adversary enabled, step size of adversarial training
        :param gradcam: if enabled, gradcam callback is called
        :param out_dir: if gradcam enabled, output directory of gradcam heatmap
        :param complete: dummy argument
        """
        if self.type == "SVC":
            self.SVC(TS)
        elif self.type == "RFC":
            self.RFC(TS)
        elif self.type == "DL" or "RESNET":
            self.LearningAdversarially(TS, VS, aug=aug, g=g, path=out_dir, eps=eps)
            """self.DL_model_selection(
                TS, VS, adversary, eps, gradcam=gradcam, out_dir=out_dir
            )"""
        else:
            self.LearningAdversarially(TS, VS, aug=aug, g=g, path=out_dir, eps=eps)
