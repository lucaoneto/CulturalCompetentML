#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import os
import pathlib
import cv2
import random
import time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import sys

from tensorflow import keras
from keras import layers
random.seed(datetime.now().timestamp())


## DataClass should:
# Collect all images of a given directory and prepare the labels
# Hp1: we have some cultures, which should be initially separated
# Hp2: we are dealing with a classification task
# The structure of the dataset is:
# List per Cultures
# List per Label
# Then we need to build a function that prepares the dataset
# in different contexts (shallow, deep, or )
class DataClass:
    """DataClass is a util class for managing data


    DataClass initialization contains only the dataset got from
    the folder.
    Time by time a function is called in order to prepare
    Tr, V and Te division inside variables (X,y), (Xv, yv)
    and (Xt, yt) respectively.

    Attributes:
        dataset: is a list of datasets per culture. each dataset
        contains a list of data subdivided per labels in the form (X,y)
        X contains the image in RGB
        y contains the label in the form (culture, class)
    """

    def __init__(self, paths) -> None:
        """
        init function gets the images from the folder
        images are stored inside self.dataset variable

        :param paths: contains the paths from which the program gets
        the images
        :return None
        """
        self.dataset = []
        for j, path in enumerate(paths):
            labels = self.get_labels(path)
            imgs_per_culture = []
            for i, label in enumerate(labels):
                images = self.get_images(path + "/" + label)
                X = []
                for k in range(len(images)):
                    X.append([images[k], [j, i]])
                # print(f"Culture is {j}, label is {i}")
                # plt.imshow(images[0])
                # plt.show()
                imgs_per_culture.append(X)
                del images
                del X
            self.dataset.append(imgs_per_culture)

    def get_labels(self, path):
        """
        get_labels returns a list of the labels in a directory

        :param path: directory in which search of the labels
        :return list of labels
        """
        dir_list = []
        for file in os.listdir(path):
            d = os.path.join(path, file)
            if os.path.isdir(d):
                d = d.split("\\")
                if len(d) == 1:
                    d = d[0].split("/")
                d = d[-1]
                dir_list.append(d)
        return dir_list

    def get_images(self, path, n=1000, rescale=False):
        """
        get_images returns min(n, #images contained in a directory)

        :param path: directory in which search for images
        :param n: maximum number of images

        :return list of images
        """
        images = []
        types = ("*.png", "*.jpg", "*.jpeg")
        paths = []
        for typ in types:
            paths.extend(pathlib.Path(path).glob(typ))
        paths = paths[0 : min(len(paths), n)]
        for i in paths:
            im = cv2.imread(str(i)) 
            if rescale:
                im = im  /255
            im = im[..., ::-1]
            images.append(im)
        return images

    # STANDARD ML
    # Standard ML does not need culture information
    # Standard ML does not need culture division for Tr and V sets
    def prepare(
        self,
        standard,
        culture,
        percent=0,
        shallow=0,
        val_split=0.2,
        test_split=0.2,
        n=1000,
        n_cultures=3,
        adversarial=0,
        imbalanced=0
    ):
        """
        this function prepares time by time the sets for training
        and evaluating the models
        :standard standard ML does not need culture information in labels
        :param culture: states which is the majority culture
        :param percent: indicates which percentage of minority culture is
        put inside Tr and V sets
        :param shallow: if shallow learning images are converted
        to Greyscale and then flattened
        :param val_split: indicates the validation percentage with respect
        to the sum between Tr and V sets
        :param test_split: indicates the test percentage with respect to the
        whole dataset
        :param n: number of images for each class and culture
        :return None
        """
        random.seed(time.time_ns())

        self.X = []
        self.y = []
        self.Xv = []
        self.yv = []
        self.Xt = []
        self.yt = []
        for c, cDS in enumerate(self.dataset):
            cultureXt = []
            cultureyT = []
            for lb, lDS in enumerate(cDS):
                random.shuffle(lDS)
                Xds = []
                yds = []

                for img, label in lDS:
                    if standard and not adversarial and not imbalanced:
                        label = int(label[1])
                    else:
                        if not standard or adversarial:
                            a = np.zeros(n_cultures)
                            a[c] = 1
                            a = np.append(a, label[1])
                            label=list(a) #label is {0,..0,1,0...0, original_label}  with 0,..,0,1,0..,0 is one hot encoding
                        
                    if shallow:
                        img = img[0::]
                        img = img.flatten()
                    Xds.append(img)
                    yds.append(label)
                nt = int(n * test_split)
                cultureXt.extend(Xds[0:nt])
                cultureyT.extend(yds[0:nt])

                Xds = Xds[nt : n - 1]
                yds = yds[nt : n - 1]
                if percent != 0:
                    if c != culture:
                        Xds = Xds[0 : int(percent * len(Xds))]
                        yds = yds[0 : int(percent * len(yds))]
                    nv = int(val_split * len(Xds))
                    self.Xv.extend(Xds[0:nv])
                    self.yv.extend(yds[0:nv])
                    self.X.extend(Xds[nv : len(Xds)])
                    self.y.extend(yds[nv : len(yds)])
                else:
                    if c == culture:
                        nv = int(val_split * len(Xds))
                        self.Xv.extend(Xds[0:nv])
                        self.yv.extend(yds[0:nv])
                        self.X.extend(Xds[nv : len(Xds)])
                        self.y.extend(yds[nv : len(yds)])
                del Xds
                del yds
            self.Xt.append(cultureXt)
            self.yt.append(cultureyT)



        

    def clear(self):
        """
        clear empty all the dataset divisions
        """
        self.X = None
        self.y = None
        self.Xv = None
        self.yv = None
        self.Xt = None
        self.yt = None
        del self.X
        del self.y
        del self.Xv
        del self.yv
        del self.Xt
        del self.yt


class NullWriter:
    def write(self, _): pass

def suppress_output():
    sys.stdout = NullWriter()

def restore_output():
    sys.stdout = sys.__stdout__
## Preprocessing Class should:
# given a dataset it should perform standard data augmentation
# given a dataset and a model it should perform adversarial data augm
class PreprocessingClass:
    def classical_augmentation(self, X, g=0.1, n=-1):
        """
        this function gets a set of images and return them augmented
        param: X: the set of images
        param: g_rot: is the gain of random rotation
        param: g_noise: is the gain of random noise
        param_ g_bright: is the gain of random brightness
        param_ n: number of images to perform the augmentation
        """
        if n <= 0 or n == None:
            n = len(X)
        X = X[0:n]
        X = np.asarray(X)

        shape = np.shape(X[0])
        suppress_output()
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
        
        X_augmented = data_augmentation(X, training=True)
        restore_output()
        return np.asarray(X_augmented)

    def adversarial_augmentation(self, X, y, model, culture, eps=0.3):
        """
        adversarial_augmentation creates adversarial samples, i.e.
        samples that are created using samples in the dataset and the
        gradient for testing/enforce the robustness of a ML model

        :param X: samples to be used as starting point
        :param y: labels of the respective samples
        :param model: model used for computing the gradient
        :param culture: used because in our mitigation strategy we select the output using
        the culture from which the image derives
        :param eps: gain of the fast gradient method

        return: a set X_augmented of adversarial samples of the same size of X

        """
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        X_augmented = []
        for i in range(len(X)):
            x = X[i]
            y_i = y[i]
            if len(np.shape(y_i)) > 0:
                if np.shape(y_i)[0] > 1:
                    y_i = y_i[1]
            y_i = y_i * 2 - 1
            y_i = np.asarray([y_i], dtype=float)
            y_i = y_i[None, ...]
            X_augmented.append(
                self.my_fast_gradient_method(
                    model,
                    np.asarray(x[None, ...]),
                    eps,
                    np.inf,
                    y=y_i,
                    culture=culture,
                    loss_fn=bce,
                )
            )
        return X_augmented

    @tf.function
    def my_compute_gradient(self, model_fn, loss_fn, x, y, targeted, culture=0):
        """
        my_compute_gradient computes the gradient of a model

        :param model_fn: model with respect to compute the gradient
        :param loss_fn: loss function used for computing the gradient
        :param x: samples
        :param y: label of samples
        :param targeted: if targeted, minimize loss of target label rather than maximize loss of correct label
        :param culture: culture that selects the corresponding output

        :return gradient
        """
        with tf.GradientTape() as g:
            g.watch(x)
            # Compute loss
            yP = model_fn(x)
            if tf.size(yP) > 1:
                yP = yP[culture]
            else:
                yP = yP[None, ...]

            loss = loss_fn(y, yP)
            if (
                targeted
            ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
                loss = -loss

        # Define gradient of loss wrt input
        grad = g.gradient(loss, x)
        return grad

