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
random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())
import gc

#tf.keras.backend.set_floatx('float32')

class CustomReg(Regularizer):
    """
    Custom regularizer for culture-bias mitigation.
    
    Minimizes the variance of culture-specific weight matrices by penalizing
    deviations from the mean weight vector. This encourages the model to learn
    similar decision boundaries for all cultures.
    
    Mathematical Formulation:
        L_reg = λ * Σ_i ||w_i - mean(w)||²
    
    Where:
        - w_i: weight matrix for culture i
        - mean(w): average weight matrix across all cultures
        - λ: regularization strength parameter
    
    This strategy aims to reduce per-culture accuracy disparities.
    """
    def __init__(self, lamb, n_cultures):
        """
        Initialize the custom regularizer.
        
        Args:
            lamb (float): Regularization strength (λ parameter)
            n_cultures (int): Number of cultures in dataset
        """
        self.lamb = lamb
        self.n_cultures = n_cultures

    def __call__(self, x):
        """
        Compute regularization penalty for weight matrix.
        
        Args:
            x (Tensor): Weight matrix of shape [features, n_cultures]
        
        Returns:
            Tensor: Scalar regularization penalty
        """
        # Compute mean weight across cultures
        mean = tf.reshape(tf.reduce_mean(x, axis=1), [-1, 1])
        # Compute deviations from mean
        diff = tf.subtract(x, mean)
        # Sum of squared deviations
        reg = tf.reduce_sum(tf.square(diff))
        # Apply regularization strength
        res = (self.lamb) * reg

        return res

class MitigatedModels(GeneralModelClass):
    """
    Mitigated models with custom bias reduction strategies.
    
    Implements culture-aware training using:
    - Custom regularization to reduce per-culture weight variance
    - Per-culture output heads for culture-specific predictions
    - Culture-Inclusive loss (CIC) for fairness
    
    Aims to reduce cultural bias while maintaining overall performance.
    """
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
        Initialize mitigated model with bias mitigation.
        
        Args:
            type (str): Model architecture - "DL" for ResNet50V2 (only option currently)
            culture (int): Index of majority culture
            verbose_param (int): Verbosity level
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Adam optimizer learning rate
            lambda_index (int): Regularization strength index in [0, 31]
                - Corresponds to logspace(-3, 2, 31)
                - Higher index = stronger regularization
            n_cultures (int): Number of cultures in dataset
            weights (array): Optional sample weights
            imbalanced (int): Handle imbalanced data
            diffusion (int): Use diffusion-augmented data
            parify_batches_diffusion (int): Use culture-balanced batches
        
        Note: All mitigated models inherit from GeneralModelClass with standard=0
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
  
    def custom_loss(self):
        """
        This function implements the loss and the regularizer of the mitigation stratyegy
        :param out: related to the corresponding output to be optimized
        :return loss function
        """
        n_cultures = self.n_cultures
        lamb = self.lamb
        #@tf.function
        def loss(y_true, y_pred):
            
            bc = tf.keras.losses.binary_crossentropy(y_true[:, n_cultures], tf.einsum('ij,ij->i', y_true[:, 0:n_cultures], y_pred))
            return bc
        return loss

    def custom_accuracy(self):
        #self.prev_accs = tf.ones(self.n_cultures)
        n_cultures = self.n_cultures
        #@tf.function
        def accuracy(y_true, y_pred):
            ## accuracy metric 
            yc = y_true[:, 0:n_cultures]  # Assume yc is of shape [batch_size, n_cultures]
            yt = y_true[:, n_cultures]    # Assume yt is of shape [batch_size, 1]
            # Get indices where yc has the maximum value (class with highest probability)
            preds = tf.einsum('ij,ij->i', yc, y_pred)
            acc = tf.keras.metrics.binary_accuracy(yt, preds)
            
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
        lrs=[1e-3, 1e-4, 1e-5],
        fine_lrs=[1e-6],
        epochs=[40],
        fine_epochs=15,
        nDropouts=[0.3, 0.4],
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

        lambdas = np.logspace(-3, 1, 4)
        
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
                                loss = history.history["val_loss"][-1]
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
                    layers.GaussianNoise(g),
                    #tf.keras.layers.RandomBrightness(0.01),
                    layers.RandomZoom(g, g),
                    layers.Resizing(shape[0], shape[1]),
                ]
            )

            print(f"aug is {aug}")
            
                        
            if self.imbalanced:
                #print(f'Byes before imbalanced transformation: {pickle.dumps(TS)}')
                TS = self.ImbalancedTransformation(TS)
                #print(f'Byes after imbalanced transformation: {pickle.dumps(TS)}')
                     
            #train_generator = train_datagen.flow(x=tf.constant(TS[0], dtype="float32"), y=tf.constant(TS[1], dtype="float32"), batch_size=batch_size)
            
            if self.parify_batches_diffusion:
                train_generator = tf.data.Dataset.from_tensor_slices(self.parify_batches(TS, self.culture, batch_size)).prefetch(tf.data.AUTOTUNE).cache()
            else:
                train_generator = tf.data.Dataset.from_tensor_slices((tf.constant(TS[0], dtype=tf.float32), tf.constant(TS[1], dtype=tf.float32))).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()

            del TS

            validation_generator = None
            if val:
                #val_datagen = ImageDataGenerator()
                #validation_generator = val_datagen.flow(x=Xv, y=yv, batch_size=batch_size)
                if self.parify_batches_diffusion:
                    validation_generator = tf.data.Dataset.from_tensor_slices(self.parify_batches(VS, self.culture, batch_size)).prefetch(tf.data.AUTOTUNE).cache()
                else:
                    validation_generator = tf.data.Dataset.from_tensor_slices((tf.constant(VS[0], dtype=tf.float32), tf.constant(VS[1], dtype=tf.float32))).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
                
                del VS

            tf.keras.backend.clear_session()            

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
            output = keras.layers.Dense(3, activation='sigmoid', name=f'pred_dense_layer', 
                                        kernel_regularizer=CustomReg(self.lamb, self.n_cultures))(y)
            
            self.model = keras.Model(inputs, output)


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
                loss=[self.custom_loss()],
                metrics=[self.custom_accuracy()],
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

            tf.keras.backend.clear_session()

            # FINE TUNING
            base_model.trainable = True
            # self.model.summary()

            self.model.compile(
                optimizer=keras.optimizers.Adam(fine_lr),  # Low learning rate
                loss=[self.custom_loss()],
                metrics=[self.custom_accuracy()],
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
