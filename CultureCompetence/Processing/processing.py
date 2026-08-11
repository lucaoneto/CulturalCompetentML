#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../")
import tensorflow as tf
tf_version = tf.__version__
# Split version string into major, minor, and patch numbers
version_tuple = tuple(map(int, tf_version.split(".")))
from Model.diffusion.diffusion_standard import DiffusionStandardModel
# Check if the version is less than 2.15
"""if version_tuple[1] < 15:
    from Model.diffusion.diffusion_standard import DiffusionStandardModel
else:
    from Model.diffusion.diffusion_standard_new_tf import DiffusionStandardModel"""
from Model.mitigated.mitigated_models import MitigatedModels
from Model.mitigated.mitigated_models_advanced import MitigatedModels as MitigatedModelsAdvanced
from Model.standard.standard_models import StandardModels
from Model.standard.gradcam_standard import StandardModels4GradCam
from Model.adversarial.adversarial import AdversarialStandard
from Model.discriminator.discriminator import Discriminator
from Utils.Data.Data import DataClass
from Utils.FileManager.FileManager import FileManagerClass
from Utils.Results.Results import ResultsClass
from Utils.Data.deep_paths import DeepStrings
from Utils.Data.shallow_paths import ShallowStrings
from Utils.Data.Data import PreprocessingClass
import numpy as np

import os
import gc
import cv2
import time
import random

class NullWriter:
    def write(self, _): pass

def suppress_output():
    sys.stdout = NullWriter()

def restore_output():
    sys.stdout = sys.__stdout__

def random_culture(n_cultures, culture):
    choices = [i for i in range(n_cultures) if i != culture]  # Exclude `culture`
    return random.choice(choices) if choices else None  # Return None if no valid choices

class ProcessingClass:
    """
    Main pipeline orchestrator for data processing, model training, and evaluation.
    
    This class coordinates the complete workflow:
    1. Data loading and preprocessing
    2. Optional augmentation (classical or diffusion-based)
    3. Model training with various strategies
    4. Testing and evaluation
    5. Result logging
    
    Supports multiple model types (standard, mitigated, discriminator, adversarial)
    and augmentation strategies for bias mitigation research.
    
    Attributes:
        dataobj: DataClass instance for data management
        shallow (bool): Shallow learning (SVM) vs deep learning (ResNet)
        lamp (bool): Use lamp dataset vs carpet dataset
        basePath (str): Base output directory
    """

    def __init__(
        self, shallow, lamp, gpu=False, memory_limit=2700, basePath="./"
    ) -> None:
        """
        Initialize the processing pipeline.
        
        Args:
            shallow (bool): 
                True = shallow learning (SVM, RFC with flattened grayscale images)
                False = deep learning (ResNet with RGB images)
            lamp (bool):
                True = use lamp dataset
                False = use carpet dataset
            gpu (bool): Enable GPU acceleration for TensorFlow
            memory_limit (int): GPU memory limit in MB
            basePath (str): Base path for output files
        
        Sets up dataset paths, GPU configuration, and initializes DataClass
        for image loading and preprocessing.
        """
        # Load dataset paths based on learning type
        if shallow:
            strObj = ShallowStrings()
            if lamp:
                paths = strObj.lamp_paths
            else:
                # Shallow learning for carpets not implemented
                paths = None
        else:
            # Deep learning uses RGB images
            strObj = DeepStrings()
            if lamp:
                paths = strObj.lamp_paths
            else:
                paths = strObj.carpet_paths_str
        
        if paths:
            self.dataobj = DataClass(paths)
        else:
            raise Exception("Carpet Problem has not been tackled in shallow learning")
        
        self.shallow = shallow
        self.lamp = lamp
        self.basePath = basePath
        
        # Configure GPU if requested
        if gpu:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                # Set memory limit for GPU to prevent OOM errors
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_limit
                            )
                        ],
                    )
                    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                    print(
                        len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                    )

                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(e)
            else:
                print("no gpus")
        else:
            # Disable GPU: use CPU only
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def parify_batches(self, s, culture, hot_encoding, size):
        """
        Create culture-balanced batches for training.
        
        This method ensures that each batch contains samples from all cultures in proportion
        to the majority culture. This addresses cultural imbalance by forcing the model
        to see all cultures equally during each training step.
        
        Args:
            s (tuple): Data tuple (X, y) where X are images and y are labels
            culture (int): Index of majority culture
            hot_encoding (bool): Whether labels are one-hot encoded
            size (int): Output image size (square images)
        
        Returns:
            tuple: (batches_X, batches_y) - balanced batches of images and labels
        
        Strategy:
            1. Extract indices for each culture from one-hot or label format
            2. Sample culture-balanced examples: for each sample from majority culture,
               include one sample from each other culture
            3. Group samples into batches of 64
            4. If any culture runs out of samples, oversample using modular indexing
        """
        data_x = np.asarray(s[0])
        data_y = np.asarray(s[1])
        
        indeces_per_culture = []
        batch_size = 64
        
        # 1. Extract indices for each culture
        if hot_encoding:
            for i in range(self.n_cultures):
                # Create the one-hot vector to match against
                c = np.zeros(self.n_cultures)
                c[i] = 1.0
                # Match rows where the first n_cultures columns equal the one-hot vector
                idx = np.where((data_y[:, :self.n_cultures] == c).all(axis=1))[0]
                indeces_per_culture.append(idx)
        else:
            # Assume data_y is a 1D array of class integers or labels are in a specific column
            # Adjust the index data_y[:, 0] if your labels are elsewhere
            labels = data_y if data_y.ndim == 1 else data_y[:, 0]
            for i in range(self.n_cultures):
                idx = np.where(labels == i)[0]
                indeces_per_culture.append(idx)

        # Determine how many samples we take (based on the requested majority culture)
        n_samples_majority = len(indeces_per_culture[culture])
        
        if n_samples_majority == 0:
            return [], []

        DS = []
        DSlabel = []
        B = []
        Blabel = []

        # 2. Build balanced batches
        # We iterate based on the majority culture count to ensure parity
        for i in range(n_samples_majority):
            for j in range(self.n_cultures):
                # Modular indexing handles cultures with fewer samples (oversampling)
                current_culture_indices = indeces_per_culture[j]
                if len(current_culture_indices) == 0: continue
                
                # Sample from culture j, cycling back to start if needed (oversampling)
                target_idx = current_culture_indices[i % len(current_culture_indices)]
                
                sample = data_x[target_idx]
                label = data_y[target_idx]
                
                # Consistent resizing to target size
                resized_sample = cv2.resize(sample, (size, size), interpolation=cv2.INTER_CUBIC)
                
                B.append(resized_sample)
                Blabel.append(label)

                # Check if batch is full
                if len(B) == batch_size:
                    DS.append(np.array(B))
                    DSlabel.append(np.array(Blabel))
                    B, Blabel = [], []

        # 3. Handle remaining samples to fill the last batch to batch_size
        if len(B) > 0:
            while len(B) < batch_size:
                # Fill with random samples from any culture to maintain batch shape
                j = len(B) % self.n_cultures
                if len(indeces_per_culture[j]) > 0:
                    rnd_idx = np.random.choice(indeces_per_culture[j])
                    B.append(cv2.resize(data_x[rnd_idx], (size, size), interpolation=cv2.INTER_CUBIC))
                    Blabel.append(data_y[rnd_idx])
                else:
                    # Fallback if a culture is completely empty
                    rnd_idx = np.random.randint(0, len(data_x))
                    B.append(cv2.resize(data_x[rnd_idx], (size, size), interpolation=cv2.INTER_CUBIC))
                    Blabel.append(data_y[rnd_idx])
                    
            DS.append(np.array(B))
            DSlabel.append(np.array(Blabel))

        return DS, DSlabel


    def prepare_data(
        self,
        standard,
        culture,
        percent=0,
        val_split: float = 0.2,
        test_split: float = 0.2,
        n: int = 1000,
        augment=0,
        gaug = 0.01,
        adversarial=0,
        imbalanced=0,
        discriminator=0,
        diffusion = 0,
        aug = 0,
        weights = 0,
        only_minority_diffusion=0,
        parify_batches_diffusion=0,
        plt_imgs = True
    ):
        """
        Prepare training data with optional augmentation and bias mitigation strategies.
        
        Loads data, applies culture-stratified splitting, and optionally applies:
        - Classical augmentation (rotation, noise, brightness)
        - Diffusion-based synthetic data generation for minority cultures
        - Parified batches for culture-balanced training
        
        Args:
            standard (int): 0=mitigated strategy, 1=standard model
            culture (int): Index of majority culture (0 to n_cultures-1)
            percent (float): Percentage of minority culture data to include (0-1)
            val_split (float): Validation set ratio (0-1)
            test_split (float): Test set ratio (0-1)
            n (int): Maximum images per culture/class
            augment (int): Enable classical augmentation
            gaug (float): Classical augmentation gain
            adversarial (int): Enable adversarial training
            imbalanced (int): Enable imbalanced learning strategies
            discriminator (int): Prepare for discriminator training
            diffusion (int): Enable diffusion-based augmentation
            only_minority_diffusion (int): Generate only for non-majority cultures
            parify_batches_diffusion (int): Create culture-balanced batches
            plt_imgs (bool): Plot sample images
        
        Data Preparation Pipeline:
            1. Load images from disk for specified culture distribution
            2. Split into train/validation/test sets (stratified by culture)
            3. Optionally augment training data
            4. Optionally generate synthetic images with diffusion
            5. Store in self.dataobj for use in model training
        """
        # Load and split data with specified cultural composition
        
        self.dataobj.prepare(
            standard=standard,
            culture=culture,
            percent=percent,
            shallow=self.shallow,
            val_split=val_split,
            test_split=test_split,
            n=n,
            adversarial=adversarial or discriminator,
            imbalanced=imbalanced,

        )
        if augment:
            print("Training Augmentation...")
            suppress_output()
            prepObj = PreprocessingClass()
            X_augmented = prepObj.classical_augmentation(
                X=self.dataobj.X, g=gaug, 
            )
        
            self.dataobj.X.extend(X_augmented)
            self.dataobj.y.extend(self.dataobj.y)
            restore_output()
            del X_augmented
            del prepObj 
        
        if diffusion==1 and not discriminator:
            print(f"Diffusion")
            size = 100
            n_imgs = len(self.dataobj.X)//25
            diff_model = DiffusionStandardModel(image_size=size)
            init_shape = np.shape(self.dataobj.X[0])[0:2]

            bpath = "./"
            if standard==0:
                bpath = bpath + '/MIT/'
            if parify_batches_diffusion:
                bpath = bpath + '/PAR_BS/'
            if only_minority_diffusion:
                bpath = bpath + '/ONLY_MIN/'

            fObj = FileManagerClass(bpath)
            del fObj

            fObj = FileManagerClass(bpath+'/GeneratedImages/')
            del fObj

            is_parify = parify_batches_diffusion
            is_only_min = only_minority_diffusion

            for j in range(2):
                tempX, tempXv, tempY, tempYv, tempXt = [], [], [], [], []
                
                # Filter Training Data
                for i in range(len(self.dataobj.X)):
                    # Extract the categorical label (j) based on format
                    label_val = self.dataobj.y[i] if standard else self.dataobj.y[i][self.n_cultures]
                    
                    # Check conditions: Correct category AND (if only_min, must NOT be the majority culture)
                    match_cat = (label_val == j)
                    match_culture = True
                    if is_only_min:
                        if standard:
                            # Assuming standard mode stores culture in a secondary structure or index 0
                            match_culture = (self.dataobj.y[i][0] != culture) if isinstance(self.dataobj.y[i], list) else True
                        else:
                            match_culture = (np.argmax(self.dataobj.y[i][0:self.n_cultures]) != culture)

                    if match_cat and match_culture:
                        img = cv2.resize(self.dataobj.X[i], (size, size), interpolation=cv2.INTER_CUBIC)
                        tempX.append(img)
                        tempY.append(self.dataobj.y[i])

                # Filter Validation Data (Repeat logic for Xv)
                for i in range(len(self.dataobj.Xv)):
                    label_val_v = self.dataobj.yv[i] if standard else self.dataobj.yv[i][self.n_cultures]
                    if label_val_v == j:
                        img_v = cv2.resize(self.dataobj.Xv[i], (size, size), interpolation=cv2.INTER_CUBIC)
                        tempXv.append(img_v)
                        tempYv.append(self.dataobj.yv[i])

                for k in range(len(self.dataobj.Xt)):
                    tempXt.append([])
                    for i in range(len(self.dataobj.Xt[k])):
                        label_val_v = self.dataobj.yt[k][i] if standard else self.dataobj.yt[k][i][self.n_cultures]
                        if label_val_v == j:
                            img_v = cv2.resize(self.dataobj.Xt[k][i], (size, size), interpolation=cv2.INTER_CUBIC)
                            tempXt[k].append(img_v)

                # 2. Apply Parify Batches if activated
                if is_parify and len(tempX) and (not (is_only_min)) > 0:
                    # Pass True for adversarial (not standard) to match your logic
                    tempX, _ = self.parify_batches((tempX, tempY), culture, (not standard), size)
                    tempXv, _ = self.parify_batches((tempXv, tempYv), culture, (not standard), size)

                # 3. Model Training / Image Generation
                if len(tempX) > 0:
                    images = diff_model.learn_on_custom_dataset(
                        tempX, tempXv, n_images=n_imgs, plot_imgs=plt_imgs, aug=aug, 
                        percent=percent, lamp=self.lamp, culture=culture, category=j, 
                        imb=imbalanced, parify_batches_diffusion=is_parify, 
                        base_path=bpath, onlymin=is_only_min,
                        test_set=tempXt
                    )

                    # 4. Post-processing and Appending
                    for img in images:
                        img = cv2.resize(np.asarray(img, dtype=np.float32), init_shape, interpolation=cv2.INTER_CUBIC)
                        self.dataobj.X.append(img)
                        
                        if standard and (not adversarial):
                            self.dataobj.y.append(j)
                        else:
                            # Generate adversarial multi-label
                            c = random_culture(self.n_cultures, culture)
                            lbl = list(np.zeros(self.n_cultures))
                            lbl[c] = 1.0
                            lbl.append(j)
                            self.dataobj.y.append(lbl)
                else: 
                    print(f"\n\n\n\nnPay Attention len tempX is 0!!!!\n\n\n\n")

            del diff_model   
        
        

    def prepare_test(
        self,
        augment=0,
        g_rot: float = 0.1,
        g_noise: float = 0.1,
        g_bright: float = 0.1,
        adversary=0,
        culture=None,
        eps=0.3,
        nt=None,
    ):
        """
        This function prepares the data for testing
        :param augment: if enabled, we augment the dataset
        :param g_rot: if augment is enabled, is the gain of random rotation
        :param g_noise: if augment is enabled, is the gain of gaussian noise
        :param g_bright: if augment is enabled, is the gain of random brightness
        :param adversary: if enabled, we augment the dataset using adversary samples
        :param culture: if adversary is enabled, we need the output information for implementing
        fast gradient method
        :param eps: is adversary is enabled, it is the gain of fast gradient method
        :param nt: is the number of images to use for testing
        """
        self.Xt_totaug = []
        self.Xt_adv = []
        self.Xt_aug = []
        if nt != None and nt < len(self.dataobj.Xt):
            self.dataobj.Xt = self.dataobj.Xt[0:nt]
        for culture in range(3):
            if augment:
                if adversary:
                    if self.model != None and culture != None:
                            print("Preparing Tot Aug for Testing...")
                            prepObj = PreprocessingClass()
                            Xt_aug = prepObj.classical_augmentation(
                                X=self.dataobj.Xt[culture],
                                g_rot=g_rot,
                                g_noise=g_noise,
                                g_bright=g_bright,
                            )
                            self.Xt_totaug.append(
                                prepObj.adversarial_augmentation(
                                    X=Xt_aug,
                                    y=self.dataobj.yt[culture],
                                    model=self.model,
                                    culture=culture,
                                    eps=eps,
                                )
                            )
                            del prepObj
                    else:
                        raise Exception(
                            "Incorrect call for prepare_test, missing model or culture"
                        )
                else:
                    
                        print("Preparing Aug for Testing...")
                        prepObj = PreprocessingClass()
                        self.Xt_aug.append(
                            prepObj.classical_augmentation(
                                X=self.dataobj.Xt[culture],
                                g_rot=g_rot,
                                g_noise=g_noise,
                                g_bright=g_bright,
                            )
                        )
                        del prepObj
            else:
                if adversary:
                    if self.model != None and culture != None:
                        print("Preparing Adv for Testing...")
                        
                        prepObj = PreprocessingClass()
                        self.Xt_adv.append(
                            prepObj.adversarial_augmentation(
                                X=self.dataobj.Xt[culture],
                                y=self.dataobj.yt[culture],
                                model=self.model,
                                culture=culture,
                                eps=eps,
                            )
                        )
                        del prepObj
                    else:
                        raise Exception(
                            "Incorrect call for prepare_test, missing model or culture"
                        )

    def process(
        self,
        standard,
        type="DL",
        points=50,
        kernel="linear",
        verbose_param=0,
        learning_rate=0.001,
        epochs=15,
        batch_size=2,
        lambda_index=0,
        culture=0,
        percent=0,
        val_split: float = 0.2,
        test_split: float = 0.1,
        n: int = 1000,
        augment=0,
        gaug: float = 0.1,
        discriminator=0,
        adversary=0,
        eps=0.3,
        gradcam=False,
        complete=0,
        n_cultures=3,
        imbalanced=0,
        class_division=0,
        only_imb_imgs=0,
        diffusion=0,
        only_minority_diffusion=0,
        parify_batches_diffusion=0,
        mitigation_type=0,
        just_preprare = False
    ):
        """
        process function prepares the data and fit the model

        This function prepares the data for training
        :param standard: if enabled, we prepare the dataset for
        standard ML, else our mitigation strategy
        :param type: select the algorithm, possible values: (SVM and DL/RESNET)
        :param points: if the selected algorithm is SVM, this value sets the number of points used in the grid
        :param kernel: if the selected algorithm is SVM, this value sets the kernel (linear or gaussian)
        :param verbose_param: sets the verbose mode
        :param learning_rate: if the selected algorithm is DL, this value sets the gain of the step
        :param epochs: if the selected algorithm is DL, this value sets the number of epochs
        :param lambda_index: if we are in our Mitigation Strategy mode, it selectes the gain of the regularizer
        :param batchs_size: if the selected algorithm is DL, this value sets the batch size
        :param culture: culture is an integer number from 0 to |C|-1,
        that represents the majority culture used for training the dataset
        :param percent: is the percentage of images from their dataset of the minority cultures
        :param val_split: is the proportion of the Validation Set w.r.t the union of the Learning and Validation sets
        :param test_split: is the proprtion of the Test Set w.r.t the whole dataset
        :param n: is the maximum number of images contained in each cultural dataset for each class
        :param augment: if enabled, we augment the dataset
        :param g_rot: if augment is enabled, is the gain of random rotation
        :param g_noise: if augment is enabled, is the gain of gaussian noise
        :param g_bright: if augment is enabled, is the gain of random brightness
        :param culture: if adversary is enabled, we need the output information for implementing
        fast gradient method
        :param eps: is adversary is enabled, it is the gain of fast gradient method
        :param nt: is the number of images to use for testing
        :param gradcam: if enabled, we extrapolate the GradCAM during training for explainability
        """
        weights = np.ones(n_cultures) * 1/2 #percent
        weights[culture] = 1  # these are the proportions in the dataset
        self.n_cultures = n_cultures
        self.prepare_data(
            standard=standard,
            culture=culture,
            percent=percent,
            val_split=val_split,
            test_split=test_split,
            n=n,
            augment=augment,
            adversarial=adversary,
            imbalanced=imbalanced,
            discriminator=discriminator,
            diffusion = diffusion,
            gaug = gaug,
            weights = weights,
            only_minority_diffusion=only_minority_diffusion,
            parify_batches_diffusion=parify_batches_diffusion
        )
        self.model = None
        if just_preprare:
            return
        
        # Base path:
        # - STD/MIT
        # - model: SVC, RFC, DL
        # - culture: LC, LF, LT, CI, CJ, CS
        # - augment in TS: NOAUG, STDAUG, ADV, TOTAUG
        # - lambda index: -1, 0, 1, ...
        # Complete path:
        # - augment in Test: TNOAUG, TSTDAUG, TADV, TTOTAUG
        if discriminator:
            self.basePath = self.basePath + "/DISCR/STD/" + type
        else:
            if standard:
                self.basePath = self.basePath + "STD/" + type
                if type!="DL":
                    if type=="SVC":
                        self.basePath = self.basePath + f"/K={kernel}/P={points}/"
                    else:
                        self.basePath = self.basePath + f"/P={points}/"
            else:
                self.basePath = self.basePath + "MIT/" + type

        if imbalanced:
            self.basePath = self.basePath + "/IMB/"
        else:
            self.basePath = self.basePath + "/BAL/"
        if self.lamp:
            if culture == 0:
                c = "/LC/"
            elif culture == 1:
                c = "/LF/"
            elif culture == 2:
                c = "/LT/"
            else:
                c = "/LC/"
        else:
            if culture == 0:
                c = "/CI/"
            elif culture == 1:
                c = "/CJ/"
            elif culture == 2:
                c = "/CS/"
            else:
                c = "/CI/"
        self.basePath = self.basePath + c + str(percent) + "/"
        if diffusion: 
            self.basePath = self.basePath + "DIFFUSION/"
            if only_minority_diffusion:
                self.basePath = self.basePath + "ONLY_MIN/"
        if parify_batches_diffusion:
                self.basePath = self.basePath + "PAR_BS/"
        if augment:
            if adversary:
                if only_imb_imgs:
                    aug = f"ADD_TOTAUG/g={gaug}/eps={eps}/"
                else:
                    aug = f"TOTAUG/g={gaug}/eps={eps}/"
                if class_division:
                    aug = aug + "/CLSDIV/"
                else:
                    aug = aug + "/NOCLSDIV/"
           

            else:
                aug = f"STDAUG/g={gaug}/"
        else:
            if adversary :
                if only_imb_imgs:
                    aug = f"ADD_AVD/eps={eps}/"
                else:
                    aug = f"AVD/eps={eps}/"
                if class_division:
                    aug = aug + "/CLSDIV/"
                else:
                    aug = aug + "/NOCLSDIV/"
            
            else:
                aug = "NOAUG/"

        self.basePath = self.basePath + aug
        if (not standard) and (not complete):
            self.basePath = self.basePath + str(lambda_index) + "/"

        if discriminator:
            self.model = Discriminator(
                type=type,
                points=points,
                kernel=kernel,
                verbose_param=verbose_param,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                weights=weights,
                imbalanced=imbalanced,
                class_division=class_division,
                
            )
        else:
            if standard:
                if adversary:
                    self.model = AdversarialStandard(
                        type=type,
                        points=points,
                        kernel=kernel,
                        verbose_param=verbose_param,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        batch_size=batch_size,
                        weights=weights,
                        imbalanced=imbalanced,
                        class_division=class_division,
                        only_imb_imgs=only_imb_imgs,
                        path = self.basePath,
                        culture = culture
                    )
                    
                else:
                    if gradcam:
                        self.model = StandardModels4GradCam(
                            type=type,
                            points=points,
                            kernel=kernel,
                            verbose_param=verbose_param,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            batch_size=batch_size,
                            weights=weights,
                            imbalanced=imbalanced,
                            diffusion=diffusion,
                        )
                    else:
                        self.model = StandardModels(
                            type=type,
                            points=points,
                            kernel=kernel,
                            verbose_param=verbose_param,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            batch_size=batch_size,
                            weights=weights,
                            imbalanced=imbalanced,
                            diffusion=diffusion,    
                            path = self.basePath,                        
                        )
            else:
                if mitigation_type==1:
                    self.model = MitigatedModelsAdvanced(
                        type=type,
                        culture=culture,
                        verbose_param=verbose_param,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        lambda_index=lambda_index,
                        n_cultures=n_cultures,
                        imbalanced=imbalanced,
                        diffusion=diffusion,
                        weights=weights,
                        parify_batches_diffusion=parify_batches_diffusion
                    )
                else:   
                    self.model = MitigatedModels(
                        type=type,
                        culture=culture,
                        verbose_param=verbose_param,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        lambda_index=lambda_index,
                        n_cultures=n_cultures,
                        imbalanced=imbalanced,
                        diffusion=diffusion,
                        weights=weights,
                        parify_batches_diffusion=parify_batches_diffusion
                    )

        self.model.standard = standard
        self.model.fit(
            TS=(self.dataobj.X, self.dataobj.y),
            VS=(self.dataobj.Xv, self.dataobj.yv),
            eps=eps,
            gradcam=gradcam,
            out_dir=self.basePath,
            complete=complete,
            aug=augment,
            g=gaug,
        )
        if adversary:
            self.prepare_test()
            if class_division:
                for j in range(2):
                    self.discriminator_test( 0, imbalanced, self.model.adversarial_model[j], j)
            else:
                self.discriminator_test( 0, imbalanced, self.model.adversarial_model)
        self.imbalanced = imbalanced
        
        del c
        del aug


    def discriminator_test(self, augment, imbalanced, model, j=-1):
        discriminator_model = Discriminator(imbalanced=imbalanced)
        discriminator_model.model = model
        if augment:
                cm = discriminator_model.get_model_stats(
                    self.Xt_aug, self.dataobj.yt, discriminator=1
                )
                testaug = f"TSTDAUG/G_AUG={gaug}/"
        else:
                cm = discriminator_model.get_model_stats(
                    self.dataobj.Xt, self.dataobj.yt, discriminator=1
                )
                testaug = f"TNOAUG/"
        testaug = testaug + f"CULTURE/"
        path = self.basePath + testaug + f"res_scrimin={j}.csv"
        self.save_results(cm, path, discriminator=1)

    def test(
        self,
        standard,
        culture=0,
        augment=0,
        gaug=0.1,
        adversary=0,
        eps=0.3,
        nt=None,
        discriminator=0,
    ):
        """
        This function is used for testing the model

        :param augment: if enabled, we augment the dataset
        :param g_rot: if augment is enabled, is the gain of random rotation
        :param g_noise: if augment is enabled, is the gain of gaussian noise
        :param g_bright: if augment is enabled, is the gain of random brightness
        :param adversary: if enabled, we augment the dataset using adversary samples
        :param culture: if adversary is enabled, we need the output information for implementing
        fast gradient method
        :param eps: is adversary is enabled, it is the gain of fast gradient method
        :param nt: is the number of images to use for testing

        :return -1 is the model is not trained, 0 if end the testing phase
        """
        if self.model:
            self.prepare_test(
                augment=augment,
                g_rot=gaug,
                g_noise=gaug,
                g_bright=gaug,
                adversary=adversary,
                culture=culture,
                eps=eps,
            )
        else:
            print("Pay attention: no model information given for tests")
            return -1
        if discriminator==0:
            for culture in range(3):
                if standard:
                    if augment:
                        if adversary:
                            cm = self.model.get_model_stats(
                                self.Xt_totaug[culture],
                                self.dataobj.yt[culture],
                                discriminator=discriminator,
                            )
                            testaug = f"TTOTAUG/G_AUG={gaug}/EPS={eps}/"
                        else:
                            cm = self.model.get_model_stats(
                                self.Xt_aug[culture],
                                self.dataobj.yt[culture],
                                discriminator=discriminator,
                            )
                            testaug = f"TSTDAUG/G_AUG={gaug}/"
                    else:
                        if adversary:
                            cm = self.model.get_model_stats(
                                self.Xt_adv[culture],
                                self.dataobj.yt[culture],
                                discriminator=discriminator,
                            )
                            testaug = f"TAVD/EPS={eps}/"
                        else:
                            cm = self.model.get_model_stats(
                                self.dataobj.Xt[culture],
                                self.dataobj.yt[culture],
                                discriminator=discriminator,
                            )
                            testaug = f"TNOAUG/"
                    testaug = testaug + f"CULTURE{culture}/"
                    path = self.basePath + testaug + "res.csv"
                    self.save_results(cm, path, discriminator=discriminator)
                else:
                    for i in range(3):
                        if augment:
                            if adversary:
                                cm = self.model.get_model_stats(
                                    self.Xt_totaug[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TTOTAUG/G_AUG={gaug}/EPS={eps}/"
                            else:
                                cm = self.model.get_model_stats(
                                    self.Xt_aug[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TSTDAUG/G_AUG={gaug}/"
                        else:
                            if adversary:
                                cm = self.model.get_model_stats(
                                    self.Xt_adv[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TAVD/EPS={eps}/"
                            else:
                                cm = self.model.get_model_stats(
                                    self.dataobj.Xt[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TNOAUG/"
                        testaug = testaug + f"CULTURE{culture}/"
                        path = self.basePath + testaug + "out " + str(i) + ".csv"
                        print(f"Path is {path}")
                        self.save_results(cm, path, discriminator=discriminator)
                        del path
                        del testaug
        else:
            self.discriminator_test( augment, self.imbalanced, self.model)

        return

    def save_results(self, cm, path, discriminator=0):
        """
        :param cm: is the confusion matrix to be saved
        :param path: is the path in which we want to save the confusion matrix
        """
        print(f"Path is {path}")
        fObj = FileManagerClass(path)
        fObj.writecm(cm, discriminator=discriminator)
        del fObj

    def partial_clear(self, basePath=None):
        """
        Partially clear the space for avoiding memory issues
        """
        tf.keras.backend.clear_session()
        self.model = None
        del self.model
        self.dataobj.clear()
        self.Xt_totaug = None
        del self.Xt_totaug
        self.Xt_adv = None
        del self.Xt_adv
        self.Xt_aug = None
        del self.Xt_aug
        self.basePath = basePath

        gc.collect()
