#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

from Model.diffusion.diffusion_standard import DiffusionStandardModel

sys.path.insert(1, "../")
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
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
from math import ceil
random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())



class StandardModels(GeneralModelClass):
    """
    Standard machine learning models without bias mitigation.
    
    Implements three model types:
    - SVM (Support Vector Machine): Linear or RBF kernel
    - RFC (Random Forest Classifier): Ensemble method
    - DL (Deep Learning): ResNet50V2 with ImageNet pre-training
    
    Used as baselines for comparison with mitigated strategies.
    """
    def __init__(
        self,
        type="SVC",
        points=10,
        kernel="linear",
        verbose_param=0,
        learning_rate=1e-3,
        epochs=15,
        batch_size=1,
        weights=None,
        imbalanced=0,
        diffusion=0,
        path = './'
    ):
        """
        Initialize standard model.
        
        Args:
            type (str): Model type - "SVC" (SVM), "RFC" (Random Forest), or "DL" (ResNet)
            points (int): GridSearch points for SVM/RFC hyperparameter tuning
            kernel (str): SVM kernel - "linear" or "rbf" (RBF/Gaussian kernel)
            verbose_param (int): Verbosity level for logging
            learning_rate (float): Learning rate for deep learning models
            epochs (int): Number of training epochs for deep learning
            batch_size (int): Batch size for deep learning
            weights (array): Optional sample weight vector
            imbalanced (int): Handle data imbalance
            diffusion (int): Use diffusion-augmented data
            path (str): Base path for model files
        
        Note: All models inherit from GeneralModelClass with standard=1
        """
        GeneralModelClass.__init__(self, standard=1, imbalanced=imbalanced)
        self.type = type
        self.points = points
        self.kernel = kernel
        self.verbose_param = verbose_param
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = np.ones(self.n_cultures)
        self.diffusion = diffusion
        self.path = path
        if weights is not None:
            self.weights=weights

    def SVC(self, TS):
        """
        This function performs the model selection on SVM for Classification
        :param TS: union between training and validation set
        :return the best model
        """
        logspaceC = np.logspace(-2, 4, self.points)  # np.logspace(-2,2,self.points)
        logspaceGamma = np.logspace(
                -4, 2, int(np.sqrt(self.points))
            )  # np.logspace(-2,2,self.points)
        if self.kernel == "rbf":
            grid = {"svc__C": logspaceC, "svc__kernel": [self.kernel], "svc__gamma": logspaceGamma}
        if self.kernel == "linear":
            grid = {"svc__C": logspaceC, "svc__kernel": [self.kernel]}

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC())
        ])


        MS = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            verbose=self.verbose_param,
            n_jobs=-1
        )
        # training set is divided into (X,y)
        X, y = TS
        print("SVC TRAINING")
        H = MS.fit(X, y)
        # Check that C and gamma are not the extreme values
        print(f"Best params {H.best_params_}")
        # print(f"gamma best param {H.best_params_['gamma']}")
        self.model = H

    def RFC(self, TS):
        """
        This function performs the model selection on Random Forest for Classification
        :param TS: union between training and validation set
        :return the best model
        """
        
        logspace_max_depth = []
        for i in np.logspace(1, 3, self.points):
            logspace_max_depth.append(int(i))
        param_grid = {
            "rfc__n_estimators": [300],  # logspace_n_estimators,
            "rfc__max_depth": logspace_max_depth,
        }

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rfc", RandomForestClassifier(random_state=42))
        ])

        CV_rfc = GridSearchCV(
            estimator=pipe, param_grid=param_grid,
            cv=KFold(n_splits=5, shuffle=True, random_state=42), verbose=self.verbose_param, n_jobs=-1
        )
        # training set is divided into (X,y)
        X, y = TS
        print("RFC TRAINING")
        H = CV_rfc.fit(X, y)
        print(CV_rfc.best_params_)
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

    def plot_images(self, images, g, num_rows=3, num_cols=6):
        def close_event():
            plt.close()
        # plot random generated images for visual evaluation of generation quality
        shape = np.shape(images[0])
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
        
        images = images[0:num_rows * num_cols]
        generated_images = data_augmentation(tf.constant(images), training=True)
            
        fig = plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index]/255.0)
                plt.axis("off")
                #plt.imsave(f"./Sample{index}", generated_images[index])
        plt.tight_layout()
        timer = fig.canvas.new_timer(interval = 2000) #creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(close_event)
        timer.start()
        if not os.path.exists(self.path):
                print(f"Making directory: {str(self.path)}")
                os.makedirs(self.path)
        plt.savefig(self.path + f'transformations.jpg')
        plt.show()
        plt.close()
    
    def ModelSelection(
        self,
        TS,
        VS,
        aug,
        show_imgs=False,
        batches=[32],
        lrs=[1e-3, 1e-4, 1e-5],
        fine_lrs=[1e-6],
        epochs=40,
        fine_epochs=15,
        nDropouts=[0.3, 0.4],
        g=0.1,
        save=False,
        path="./",
    ):
            # SHUFFLE DATA
            zipped_data = list(zip(*TS))
            # Shuffle the list of tuples
            random.shuffle(zipped_data)
            # Unzip back into separate lists
            TS = tuple(map(list, zip(*zipped_data)))
            zipped_data = list(zip(*VS))
            # Shuffle the list of tuples
            random.shuffle(zipped_data)
            # Unzip back into separate lists
            VS = tuple(map(list, zip(*zipped_data)))
            del zipped_data

            best_loss = np.inf
            TS = (list(np.array(TS[0], dtype=np.float32)), TS[1])
            if self.imbalanced:
                #print(f'Byes before imbalanced transformation: {pickle.dumps(TS)}')
                TS = self.ImbalancedTransformation(TS)
            if self.imbalanced:
                VS = (list(np.array(VS[0], dtype=np.float32)), list(np.asarray(VS[1], dtype=np.float32)[:,1]))
            else:
                VS = (list(np.array(VS[0], dtype=np.float32)), VS[1])
            if aug: 
                self.plot_images(VS[0], g, num_rows=3, num_cols=6)
            
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

    def ImbalancedTransformation(self, TS):
        newX = []
        newY = []
        X = TS[0]
        Y = TS[1]
        for i in range(len(X)):
            img = X[i]
            label = Y[i]
            if label[0]<len(self.weights):
             for j in range(ceil(1/self.weights[label[0]])): # I use the inverse of the total proportion for augmenting the dataset
                im = np.asarray(X[i])
                newX.append(im) 
                newY.append(label[len(label)-1]) # I do not need culture for training 
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
            # DIVIDE IN BATCHES
            # TS = TS.batch(batch_size).prefetch(buffer_size=10)
            # if val:
            #    VS = VS.batch(batch_size).prefetch(buffer_size=10)
            if aug:
                if show_imgs:
                    # DISPLAY IMAGES
                    # AUGMENTATION
                    for k in range(6):
                        idx = np.random.randint(0, len(TS) - 1)
                        img = TS[0][idx]
                        temp_ims = np.expand_dims(np.asarray(img).astype('float32'), 0)                            
                        plt.figure(figsize=(10, 10))
                        
                        plt.tick_params(
                                axis='both',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) # labels along the bottom edge are off
                        plt.imshow(temp_ims[0].astype("int32"))
                        plt.show()
                        for i, j in enumerate(np.logspace(-4, 0, 6)):
                            
                            ax = plt.subplot(3, 2, i + 1)
                            dt_aug =  keras.Sequential(
                                    [
                                        layers.RandomFlip("horizontal"),
                                        layers.RandomRotation(0.01),
                                        layers.GaussianNoise(j),
                                        tf.keras.layers.RandomBrightness(0.01),
                                        layers.RandomZoom(j, j),
                                        layers.Resizing(shape[0], shape[1]),
                                    ]
                                )

                            augmented_image = dt_aug(
                            tf.constant(temp_ims), training=True
                            )
                            plt.imshow(augmented_image[0].numpy().astype("int32"))
                                
                        plt.show()

            
                #VS = self.ImbalancedTransformation(VS)
                #print(f'Byes after imbalanced transformation: {pickle.dumps(TS)}')

            
            #train_generator = train_datagen.flow(x=tf.constant(TS[0], dtype="float32"), y=tf.constant(TS[1], dtype="float32"), batch_size=batch_size)
            train_generator = tf.data.Dataset.from_tensor_slices((tf.constant(TS[0], dtype=tf.float32), tf.constant(TS[1], dtype=tf.float32))).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
            
            del TS

            validation_generator = None
            if val:
                val_datagen = ImageDataGenerator()
                #validation_generator = val_datagen.flow(x=tf.constant(VS[0], dtype="float32"), y=tf.constant(VS[1]), batch_size=batch_size)
                validation_generator = tf.data.Dataset.from_tensor_slices((tf.constant(VS[0], dtype=tf.float32), tf.constant(VS[1], dtype=tf.float32))).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
                
                del VS

            tf.keras.backend.clear_session() 

            
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

            # The base model contains batchnorm layers. We want to keep them in inference mode
            # when we unfreeze the base model for fine-tuning, so we make sure that the
            # base_model is running in inference mode here.
            x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dropout(nDropout)(x)  # Regularize with dropout
            outputs = keras.layers.Dense(1, activation="sigmoid")(x)
            self.model = keras.Model(inputs, outputs)

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
                shuffle=True
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
                shuffle=True
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
