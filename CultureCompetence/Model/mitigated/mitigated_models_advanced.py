#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

from matplotlib import pyplot as plt

from Model.diffusion.diffusion_standard import DiffusionStandardModel

sys.path.insert(1, "../")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import layers
from Model.GeneralModel import GeneralModelClass
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import random
from datetime import datetime
from keras.regularizers import Regularizer
random.seed(int(datetime.now().timestamp()))
tf.random.set_seed(int(datetime.now().timestamp()))
import gc
import time

#tf.keras.backend.set_floatx('float32')

class CustomReg(Regularizer):
        def __init__(self, lamb, n_cultures):
            self.lamb = lamb
            self.n_cultures = n_cultures

        def __call__(self, x):
            
            mean = tf.reshape(tf.reduce_mean(x, axis=1),  [-1, 1])
            diff = tf.subtract(x, mean)
            reg = tf.reduce_sum(tf.square(diff))
            res = (self.lamb) * reg

            return res

class MitigatedModels(GeneralModelClass):
    def __init__(
        self,
        type="DL",
        culture=0,
        verbose_param=0,
        epochs=15,
        batch_size=1,
        learning_rate=1e-3,
        lambda_index=-1,
        n_cultures = 3,
        weights=None,
        imbalanced=0,
        diffusion=0,
        parify_batches_diffusion=0
    ):
        """
        Initialization function for modeling mitigated ML models.
        We have narrowed the problems to image classification problems.
        :param type: selects the algorithm even if up to now "RESNET" is the only possible value.
        :param culture: selects the majority culture
        :param verbose_param: if enabled, the program logs more information
        :param learning_rate: hyperparameter for DL
        :param epochs: hyperparameter for DL
        :param batch_size: hyperparameter for DL
        :param lambda_index: select the gain of the regularizer in a logspace(-3, 2, 31)
        """
        GeneralModelClass.__init__(self, standard=0, n_cultures=n_cultures, imbalanced=imbalanced)
        self.type = type
        self.culture = culture
        self.verbose_param = verbose_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weights=np.ones(self.n_cultures)
        self.diffusion=diffusion
        self.parify_batches_diffusion=parify_batches_diffusion
        if weights is not None:
            self.weights=weights

        if lambda_index >= 0:
            lambda_grid = np.logspace(-3, 2, 31)
            self.lamb = lambda_grid[lambda_index]
        else:
            self.lamb = 0

    def computeCIC(self, errs):
        tf.add(errs, -tf.math.reduce_min(errs))
        cic = tf.reduce_mean(errs)
        return cic

    def get_cic(self, valX, valY):
        losses = []
        valY = list(np.asarray(valY)[:, self.n_cultures])
        for out in range(self.n_cultures):
            yPred = self.model.predict(np.asarray(valX, dtype="int32"))
            yPred = list(np.asarray(yPred)[:, out])
            ls = tf.keras.losses.binary_crossentropy(valY, yPred)
            losses.append(ls)
        cic = float(self.computeCIC(losses))
        return cic

    def parify_batches(self, s, culture, batch_size):
        indeces_per_culture = []
        
        for i in range(self.n_cultures):
            c = np.zeros(self.n_cultures)
            c[i] = 1.0
            
            vals = np.where((np.asarray(s[1], dtype=object)[:, :self.n_cultures] == c).all(axis=1))[0]
            indeces_per_culture.append(vals)
            if i == culture:
                n_samples_majority = len(vals)
            
        B = []
        Blabel = []
        DS = []
        DSlabel = []
        for i in range(n_samples_majority):
            for j in range(self.n_cultures):
                if len(B)>= batch_size:
                    DS.append(np.asarray(B))
                    B = []
                    DSlabel.append(np.asarray(Blabel))
                    Blabel = []
                sample = np.asarray(s[0])[indeces_per_culture[j][i % len(indeces_per_culture[j])]]
                label = np.asarray(s[1])[indeces_per_culture[j][i % len(indeces_per_culture[j])]]
                B.append(sample)
                Blabel.append(label)

        if len(B)>0:
            for i in range(batch_size-len(B)):
                j = i % self.n_cultures
                rnd_index = np.random.randint(0, len(indeces_per_culture[j]))
                B.append(s[0][indeces_per_culture[j][rnd_index]])
                Blabel.append(s[1][indeces_per_culture[j][rnd_index]])
            DS.append(np.asarray(B))
            DSlabel.append(np.asarray(Blabel))

        return DS, DSlabel
  
    def parify_batches_fast(self, s, culture, batch_size):

        X = np.asarray(s[0])
        y = np.asarray(s[1])

        # Extract culture one-hot labels
        culture_labels = y[:, :self.n_cultures]

        # ---- 1. Compute indices per culture (vectorized) ----
        indices_per_culture = [
            np.where(culture_labels[:, i] == 1)[0]
            for i in range(self.n_cultures)
        ]

        # Number of samples in majority culture
        n_majority = len(indices_per_culture[culture])

        # ---- 2. Generate balanced indices using modular trick ----
        all_indices = []
        for i in range(self.n_cultures):
            idxs = indices_per_culture[i]
            # repeat idxs until reaching n_majority
            tiled = np.resize(idxs, n_majority)
            all_indices.append(tiled)

        # Shape becomes (n_cultures, n_majority)
        all_indices = np.stack(all_indices, axis=1)
        # Flatten → balanced sample order
        final_indices = all_indices.reshape(-1)

        # ---- 3. Shuffle to avoid culture blocks ----
        np.random.shuffle(final_indices)

        # ---- 4. Slice X and y using final indices ----
        X_balanced = X[final_indices]
        y_balanced = y[final_indices]

        return X_balanced, y_balanced


    import tensorflow as tf

    def custom_loss(self):
        n_cultures = self.n_cultures
        lamb = self.lamb
        model = self.model

        def reg():
            chinese_weights = model.get_layer('pred_dense_layer_0').kernel
            french_weights = model.get_layer('pred_dense_layer_1').kernel
            turkish_weights = model.get_layer('pred_dense_layer_2').kernel
            all_weights = tf.concat([chinese_weights, french_weights, turkish_weights], axis=0)
            mean_all_weights = tf.reduce_mean(all_weights)
            regularization_term = tf.reduce_sum(tf.square(all_weights - mean_all_weights))
            return regularization_term

        def loss(y_true, y_pred_list):
            # Concatenate outputs
            y_pred = tf.concat(y_pred_list, axis=1)  # shape: (batch_size, n_cultures)
            y_true_label = y_true[:, n_cultures]
            culture_selector = y_true[:, 0:n_cultures]

            # Select prediction
            selected_y_pred = tf.reduce_sum(culture_selector * y_pred, axis=1)

            bc = tf.keras.losses.binary_crossentropy(y_true_label, selected_y_pred)
            mean_bc = tf.reduce_mean(bc)

            # Total loss
            total_loss = mean_bc + lamb * reg()
            return total_loss

        return loss

    def custom_accuracy(self):
        #self.prev_accs = tf.ones(self.n_cultures)
        n_cultures = self.n_cultures
        #@tf.function
        def accuracy(y_true, y_pred):
            ## accuracy metric 
            y_pred = tf.concat(y_pred, axis=1)  # shape: (batch_size, n_cultures)
            y_true_label = y_true[:, n_cultures]
            culture_selector = y_true[:, 0:n_cultures]

            # Select prediction
            selected_y_pred = tf.reduce_sum(culture_selector * y_pred, axis=1)

            acc = tf.keras.metrics.binary_accuracy(y_true_label, selected_y_pred)
            
            return acc

        return accuracy
    
    def ImbalancedTransformation(self, TS):
        newX = []
        newY = []
        X = TS[0]
        Y = TS[1]
        for i in range(len(X)):
            img = X[i]
            label = Y[i]
            label = label[0:self.n_cultures]
            if np.sum(label)>0:
                label = np.argmax(label)
                for j in range(int(1/self.weights[label])): # I use the inverse of the total proportion for augmenting the dataset
                    newX.append(img) 
                    newY.append(Y[i])
        del TS
        return (newX, newY)

    #@tf.function
    def regularizer(self, w):
        sum = tf.constant(0.0, dtype="float32")

        mean = tf.reduce_mean(w, axis=1)
        for i in range(self.n_cultures):
            sum += tf.math.square(tf.norm(w[:, i] - mean))

        res = (self.lamb) * sum
        return res

    def get_best_idx(self, losses: list, cics: list, tau=0.15):
        tmp_losses = losses.copy()
        n_ls = math.ceil(len(losses) * tau)

        pairs = []
        tmp_cics = []

        for i in range(n_ls):
            val = min(tmp_losses)
            idx = tmp_losses.index(val)
            pairs.append((val, idx))
            tmp_losses.remove(val)
            tmp_cics.append(cics[idx])

        mincic = min(tmp_cics)
        for i in range(n_ls):
            if mincic == cics[i]:
                idx = i

        return pairs[i][1]
    
    def create_adversarial_pattern(self, model, input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = tf.keras.losses.categorical_crossentropy(input_label, prediction)
        
        gradient = tape.gradient(loss, input_image)
        signed_grad = tf.sign(gradient)
        return signed_grad

    # Create adversarial samples
    def generate_adversarial_samples(self, model, images, labels, shape, epsilon=0.1):
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
        lrs=[8e-5, 8e-4],
        fine_lrs=[1e-6],
        epochs=[38],
        fine_epochs=12,
        nDropouts=[0.35],
        g=0.1,
        save=False,
        path="./"
    ):
        losses = []
        cics = []

    
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
        
        TS = (list(np.array(TS[0], dtype=np.float32)), TS[1])
        VS = (list(np.array(VS[0], dtype=np.float32)), VS[1])

        lambdas = np.logspace(-2, 1, 3)
        
        hyperparameters = []

        for lmb in lambdas:
            self.lamb = lmb
            for b in batches:
                for lr in lrs:
                  for ep in epochs:
                    for fine_lr in fine_lrs:
                        for nDropout in nDropouts:
                                self.model = None
                                tf.keras.backend.clear_session()
                                gc.collect()
                                print(
                                    f"Training with: lamb={lmb}, batch_size={b}, lr={lr}, fine_lr={fine_lr}, nDropout={nDropout}"
                                )
                                history = self.DL(
                                    TS,
                                    VS,
                                    aug,
                                    show_imgs,
                                    b,
                                    lr,
                                    fine_lr,
                                    ep,
                                    fine_epochs,
                                    nDropout,
                                    g=g,
                                )
                                #loss = history.history["val_loss"][-1]
                                err_0 = 1-history.history["val_pred_dense_layer_0_accuracy"][-1]
                                err_1 = 1-history.history["val_pred_dense_layer_1_accuracy"][-1]
                                err_2 = 1-history.history["val_pred_dense_layer_2_accuracy"][-1]
                                loss = (err_0 + err_1 + err_2)/3

                                CIC = self.get_cic(VS[0], VS[1])
                                print(f"loss is {loss}")
                                losses.append(loss)
                                cics.append(CIC)
                                hyperparameters.append({
                                    'batch_size': b,
                                    'lr': lr,
                                    'fine_lr': fine_lr,
                                    'nDropout': nDropout,
                                    'lambda': lmb,
                                    'epochs': ep
                                })
                                
                                self.model = None
                                gc.collect()

        idx = self.get_best_idx(losses, cics)
        Hstar = hyperparameters[idx]
        best_loss = losses[idx]
        best_CIC = cics[idx]
        best_fine_lr = hyperparameters[idx]['fine_lr']
        best_lr = hyperparameters[idx]['lr']
        best_bs = hyperparameters[idx]['batch_size']
        best_lmb = hyperparameters[idx]['lambda']
        best_epochs = hyperparameters[idx]['epochs']
        best_nDropout = hyperparameters[idx]['nDropout']

        self.lamb = best_lmb
        print(
            f"loss*:{best_loss}, batch size*:{best_bs} lr*:{best_lr}, fine_lr*:{best_fine_lr}, dropout*:{best_nDropout}, lambda*={best_lmb}, epochs*={best_epochs}, CIC*={best_CIC}"#, best CIC={best_CIC}"
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
            best_epochs,
            fine_epochs,
            best_nDropout,
            val=False,
            g=g,
        )

        if save:
                self.save(path)
  

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
            print(f"Shape of data is {shape}")
            tf.keras.backend.clear_session()

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
                    plt.title(label)
                    plt.axis("off")
                plt.show()

            if val:
                monitor_val = "val_loss"
            else:
                monitor_val = "loss"

            data_augmentation = keras.Sequential(
                [
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.01),
                    layers.GaussianNoise(0.01),
                    #tf.keras.layers.RandomBrightness(0.01),
                    layers.RandomZoom(g, g),
                    layers.Resizing(shape[0], shape[1]),
                ]
            )

            print(f"aug is {aug}")
            seed = int(time.time() % 2**31)
            
                        
            if self.imbalanced:
                #print(f'Byes before imbalanced transformation: {pickle.dumps(TS)}')
                TS = self.ImbalancedTransformation(TS)
                #print(f'Byes after imbalanced transformation: {pickle.dumps(TS)}')
                     
            #train_generator = train_datagen.flow(x=tf.constant(TS[0], dtype="float32"), y=tf.constant(TS[1], dtype="float32"), batch_size=batch_size)
            
            if self.parify_batches_diffusion:
                Xb, yb = self.parify_batches_fast(TS, self.culture, batch_size)
                train_generator = tf.data.Dataset.from_tensor_slices((Xb, yb)) \
                                            .batch(batch_size) \
                                            .prefetch(tf.data.AUTOTUNE)
                #train_generator = tf.data.Dataset.from_tensor_slices(self.parify_batches(TS, self.culture, batch_size)).prefetch(tf.data.AUTOTUNE).cache()
            else:
    
                #train_generator = tf.data.Dataset.from_tensor_slices((tf.constant(TS[0], dtype=tf.float32), tf.constant(TS[1], dtype=tf.float32))).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
                train_generator = tf.data.Dataset.from_tensor_slices(
                            (tf.convert_to_tensor(TS[0]), tf.convert_to_tensor(TS[1]))
                        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            del TS

            validation_generator = None
            if val:
                #val_datagen = ImageDataGenerator()
                #validation_generator = val_datagen.flow(x=Xv, y=yv, batch_size=batch_size)
                if self.parify_batches_diffusion:
                    Xb, yb = self.parify_batches_fast(VS, self.culture, batch_size)
                    
                    validation_generator = tf.data.Dataset.from_tensor_slices((Xb, yb)) \
                                            .batch(batch_size) \
                                            .prefetch(tf.data.AUTOTUNE)
                    #validation_generator = tf.data.Dataset.from_tensor_slices(self.parify_batches(VS, self.culture, batch_size)).prefetch(tf.data.AUTOTUNE).cache()
                else:
                    #validation_generator = tf.data.Dataset.from_tensor_slices((tf.constant(VS[0], dtype=tf.float32), tf.constant(VS[1], dtype=tf.float32))).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
                    validation_generator = tf.data.Dataset.from_tensor_slices(
                            (tf.convert_to_tensor(VS[0]), tf.convert_to_tensor(VS[1]))
                        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

                del VS
         

            # DIVIDE IN BATCHES
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
            base_model = keras.applications.ResNet50V2(
                weights="imagenet",  # Load weights pre-trained on ImageNet.
                input_shape=shape,
                include_top=False,
            )  # Do not include the ImageNet classifier at the top.

            # Freeze the base_model
            base_model.trainable = False

            # Create  model on top
            inputs = keras.Input(shape=shape)
            # Pre-trained Xception weights requires that input be scaled
            # from (0, 255) to a range of (-1., +1.), the rescaling layer
            # outputs: `(inputs * scale) + offset`
            scale_layer = keras.layers.Rescaling(scale=1 / 255.0)
            if aug:
                x = data_augmentation(inputs)   # Apply random data augmentation
                x = scale_layer(x)
            else:
                x = scale_layer(inputs)

            # The base model contains batchnorm layers. We want to keep them in inference mode
            # when we unfreeze the base model for fine-tuning, so we make sure that the
            # base_model is running in inference mode here.
            x = base_model(x, training=False)
            
            y = keras.layers.GlobalAveragePooling2D()(x)

            
            #y = keras.layers.Dropout(nDropout)(y)  # Regularize with dropout
            y = keras.layers.Flatten()(y)
            outputs = []
            for i in range(self.n_cultures):
                outputs.append(keras.layers.Dense(1, activation='sigmoid', name=f'pred_dense_layer_{i}')(y))
            
            self.model = keras.Model(inputs, outputs = outputs)


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

            
            self.model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss=self.custom_loss(),
                metrics=self.custom_accuracy(),
                #run_eagerly=True
            )

            #ws = np.linalg.norm(self.model.layers[-1].weights[0])
            self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                verbose=self.verbose_param,
                callbacks=callbacks,
                shuffle=True
            )
            #ws2 = np.linalg.norm(self.model.layers[-1].weights[0])
            #print(f"Same = {ws2==ws}")

            # FINE TUNING
            base_model.trainable = True
            # self.model.summary()

            self.model.compile(
                optimizer=keras.optimizers.Adam(fine_lr),  # Low learning rate
                loss=self.custom_loss(),
                metrics=self.custom_accuracy(),
                #run_eagerly=True
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
        save=False,
        aug=0,
        g=0.1,
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

    def get_model_from_weights(self, size, adversary=0, eps=0.05, mult=0.2, path="./"):
        self.model = tf.keras.models.load_model(path)