#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from Model.diffusion.diffusion_standard import DiffusionStandardModel

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
random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())



class StandardModels4GradCam(GeneralModelClass):
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
        diffusion=0,
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
        GeneralModelClass.__init__(self, standard=1, imbalanced=imbalanced)
        self.type = type
        self.points = points
        self.kernel = kernel
        self.verbose_param = verbose_param
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights=np.ones(self.n_cultures)
        self.diffusion = diffusion
        if weights is not None:
            self.weights=weights

    def SVC(self, TS):
        """
        This function performs the model selection on SVM for Classification
        :param TS: union between training and validation set
        :return the best model
        """
        if self.kernel == "rbf":
            logspaceC = np.logspace(-4, 3, self.points)  # np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(
                -4, 3, self.points
            )  # np.logspace(-2,2,self.points)
            grid = {"C": logspaceC, "kernel": [self.kernel], "gamma": logspaceGamma}
        if self.kernel == "linear":
            logspaceC = np.logspace(-4, 3, self.points)  # np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(
                -4, 3, self.points
            )  # np.logspace(-2,2,self.points)
            grid = {"C": logspaceC, "kernel": [self.kernel]}

        MS = GridSearchCV(
            estimator=SVC(),
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=10,
            verbose=self.verbose_param,
        )
        # training set is divided into (X,y)
        TS = np.array(TS, dtype=object)
        del TS
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        print("SVC TRAINING")
        H = MS.fit(X, y)
        # Check that C and gamma are not the extreme values
        print(f"C best param {H.best_params_['C']}")
        # print(f"gamma best param {H.best_params_['gamma']}")
        self.model = H

    def RFC(self, TS):
        """
        This function performs the model selection on Random Forest for Classification
        :param TS: union between training and validation set
        :return the best model
        """
        rfc = RandomForestClassifier(random_state=42)
        logspace_max_depth = []
        for i in np.logspace(0, 3, self.points):
            logspace_max_depth.append(int(i))
        param_grid = {
            "n_estimators": [500],  # logspace_n_estimators,
            "max_depth": logspace_max_depth,
        }

        CV_rfc = GridSearchCV(
            estimator=rfc, param_grid=param_grid, cv=5, verbose=self.verbose_param
        )
        # training set is divided into (X,y)
        TS = np.array(TS, dtype=object)
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        del TS
        print("RFC TRAINING")
        H = CV_rfc.fit(X, y)
        # print(CV_rfc.best_params_)
        self.model = H

    def create_adversarial_pattern(self, model, input_image, input_label):
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                tape.watch(input_image)
                prediction = model(input_image)
                loss = tf.keras.losses.categorical_crossentropy(input_label, prediction)
            
            gradient = tape.gradient(loss, input_image)
            signed_grad = tf.sign(gradient)
            return signed_grad

    # Create adversarial samples
    def generate_adversarial_samples(self, model, images, labels, shape, epsilon=0.1):
        with tf.device("/gpu:0"):
            adversarial_images = []
            for img, lbl in zip(images, labels):
                img = tf.convert_to_tensor(img.reshape((1, shape[0], shape[1], 3)))
                lbl = tf.convert_to_tensor(lbl.reshape((1, 1)))
                perturbations = self.create_adversarial_pattern(model, img, lbl)
                adversarial_img = img + epsilon * perturbations
                adversarial_img = tf.clip_by_value(adversarial_img, 0, 1)
                adversarial_images.append(adversarial_img.numpy())
            return tf.convert_to_tensor(adversarial_images)

    
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
           
            best_loss = np.inf
            for b in batches:
                for lr in lrs:
                    for fine_lr in fine_lrs:
                        for nDropout in nDropouts:
                                
                                self.model = None
                                gc.collect()
                                print(
                                    f"Training with: batch_size={b}, lr={lr}, fine_lr={fine_lr}, nDropout={nDropout}"
                                )
                                history = self.DL(
                                    TS,
                                    VS,
                                    aug,
                                    show_imgs,
                                    b,
                                    lr,
                                    fine_lr,
                                    epochs,
                                    fine_epochs,
                                    nDropout,
                                    g=g,
                                )
                                
                                loss = history.history["val_loss"][-1]
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
            TS = TS + VS
            self.DL(
                TS,
                None,
                aug,
                show_imgs,
                best_bs,
                best_lr,
                best_fine_lr,
                epochs,
                fine_epochs,
                best_nDropout,
                val=False,
                g=g,
            )

            if save:
                self.save(path)

    def ImbalancedTransformation(self, TS, data_augmentation, aug):
        newX = []
        newY = []
        X = TS[0]
        Y = TS[1]
        for i in range(len(X)):
            img = X[i]
            label = Y[i]
            for i in range(int(1/self.weights[label[0]])): # I use the inverse of the total proportion for augmenting the dataset
                im = np.asarray(data_augmentation(img, training=aug), dtype=object)
                newX.append(im) # I do not need culture for training 
                newY.append(label[1])
        del TS
        return (newX, newY)

    def DL(
        self,
        TS,
        VS,
        aug=False,
        show_imgs=False,
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
            n = np.shape(TS[0])

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

            if self.imbalanced:
                TS = self.ImbalancedTransformation(TS, data_augmentation, aug)
                train_datagen = ImageDataGenerator()
            else:
                train_datagen = ImageDataGenerator(
                    preprocessing_function=lambda img: data_augmentation(img, training=aug)
                )
            # Apply data augmentation to the training dataset
            X = tf.constant(TS[0], dtype="float32")
            y = tf.constant(TS[1], dtype="float32")
            train_generator = train_datagen.flow(x=X, y=y, batch_size=batch_size)
            # train_generator = train_datagen.flow(x=np.asarray(TS[0], dtype=object).astype('float32'),y=np.asarray(TS[1], dtype=object).astype('float32'), batch_size=32)
            validation_generator = None
            if val:
                val_datagen = ImageDataGenerator()
                Xv = tf.constant(VS[0], dtype="float32")
                yv = tf.constant(VS[1], dtype="float32")
                validation_generator = val_datagen.flow(x=Xv, y=yv, batch_size=batch_size)

            if show_imgs:
                # DISPLAY IMAGES
                # NOAUGMENTATION
                images = []
                for i in range(9):
                    idx = np.random.randint(0, len(TS[0]) - 1)
                    images.append((TS[0][idx], TS[1][idx]))
                plt.figure(figsize=(10, 10))
                for i, (image, label) in enumerate(images):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(image)
                    plt.title(int(label))
                    plt.axis("off")
                plt.show()

            # DIVIDE IN BATCHES
            # TS = TS.batch(batch_size).prefetch(buffer_size=10)
            # if val:
            #    VS = VS.batch(batch_size).prefetch(buffer_size=10)
            if aug:
                if show_imgs:
                    # DISPLAY IMAGES
                    # AUGMENTATION
                    idx = np.random.randint(0, len(TS) - 1)
                    images = []
                    images.append((TS[0][idx], TS[1][idx]))
                    for ims, labels in images:
                        plt.figure(figsize=(10, 10))
                        for i in range(9):
                            ax = plt.subplot(3, 3, i + 1)

                            augmented_image = data_augmentation(
                                tf.expand_dims(ims, 0), training=True
                            )
                            plt.imshow(augmented_image[0].numpy().astype("int32"))
                            plt.title(int(labels))
                            plt.axis("off")
                        plt.show()

            # MODEL IMPLEMENTATION

            # Create  model on top
            inputs = keras.Input(shape=shape)
            # Pre-trained Xception weights requires that input be scaled
            # from (0, 255) to a range of (-1., +1.), the rescaling layer
            # outputs: `(inputs * scale) + offset`
            # scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
            scale_layer = keras.layers.Rescaling(scale=1 / 255.0)
            if aug:
                x = data_augmentation(inputs)  # Apply random data augmentation
                x = scale_layer(x)
            else:
                x = scale_layer(inputs)

            base_model = keras.applications.ResNet50V2(
                weights="imagenet",  # Load weights pre-trained on ImageNet.
                input_shape=shape,
                input_tensor=x, 
                include_top=False,
            )  # Do not include the ImageNet classifier at the top.

            # Freeze the base_model
            base_model.trainable = False

            # The base model contains batchnorm layers. We want to keep them in inference mode
            # when we unfreeze the base model for fine-tuning, so we make sure that the
            # base_model is running in inference mode here.
            #x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(base_model.output)
            x = keras.layers.Dropout(nDropout)(x)  # Regularize with dropout
            outputs = keras.layers.Dense(1, activation="sigmoid")(x)
            self.model = keras.Model(inputs, outputs)

            #self.model.summary()

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

            # self.model.summary()
            # MODEL TRAINING
            self.model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()],
            )

            self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                verbose=self.verbose_param,
                callbacks=callbacks,
            )

            # FINE TUNING
            base_model.trainable = True
            # self.model.summary()

            self.model.compile(
                optimizer=keras.optimizers.Adam(fine_lr),  # Low learning rate
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()],
            )

            history = self.model.fit(
                train_generator,
                epochs=fine_epochs,
                validation_data=validation_generator,
                verbose=self.verbose_param,
                callbacks=callbacks,
            )
            tf.keras.backend.clear_session()
            return history


    def fit(
        self,
        TS,
        VS=None,
        adversary=0,
        eps=0.05,
        gradcam=False,
        out_dir="./",
        complete=0,
        aug=0,
        g=0.1,
        save=False,
    ):
        """
        General function for implementing model selection
        :param TS: training set
        :param VS: validation set
        :param adversary: if enabled, adversarial training is enabled
        :param eps: if adversary enabled, step size of adversarial training
        :param mult: if adversary enabled, multiplier of adversarial training
        :param gradcam: if enabled, gradcam callback is called
        :param out_dir: if gradcam enabled, output directory of gradcam heatmap
        :param complete: dummy argument
        """
        if self.type == "SVC":
            self.SVC(TS)
        elif self.type == "RFC":
            self.RFC(TS)
        elif self.type == "DL" or "RESNET":
            self.ModelSelection(TS, VS, aug=aug, g=g, save=save, path=out_dir)
        else:
            self.ModelSelection(TS, VS, aug=aug, g=g, save=save, path=out_dir)
