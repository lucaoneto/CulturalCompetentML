#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

import cv2

sys.path.insert(1, "../")
from GradCam.gradCam import GradCAM
from Utils.FileManager.FileManager import FileManagerClass
from Processing.processing import ProcessingClass
from math import floor
import tensorflow as tf
import os
import gc
import random
from datetime import datetime
import numpy as np

random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())

# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_asyn"

# tf.config.set_soft_device_placement(True)

memory_limit = 3800
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
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
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
else:
    print("no gpus")


percents = [0.05]
standard = 0
# lamp = 1

verbose_param = 1
n = 1000
class_divisions = [ 0]
imbalances = [0]
g_gaugs = np.logspace(-4, -1, 4)
eps = np.logspace(-3, -1, 3)
g_aug = g_gaugs[0]
cs = [2, 1, 0]
lamps = [1, 0]

ep = eps[0]
imb = 0

diffusion = 1
adversary = 0
k = 1


basePath = "./advanced/"
for i in range(2):
 for percent in percents:
    for lamp in lamps:
          for c in cs:
              for cl_div in class_divisions:
                procObj = ProcessingClass(
                    shallow=0,
                    lamp=lamp,
                    gpu=False,
                    memory_limit=memory_limit,
                    basePath=basePath,
                )
                model = None
                print(f"Training->aug={k%2};adv={floor(k/2)}")
                procObj.process(
                    standard=standard,
                    type="DL",
                    verbose_param=verbose_param,
                    culture=c,
                    percent=percent,
                    n=n,
                    augment=k % 2,
                    gaug=g_aug,
                    adversary=adversary,
                    eps =ep,
                    class_division=cl_div,
                    imbalanced=imb, 
                    diffusion = diffusion,
                    only_minority_diffusion=1-i
                )
                # NoAUg
                print(f"Testing->aug={0};adv={0}")
                procObj.test(
                    standard=standard,
                    culture=c,
                    augment=0,
                    gaug=0,
                    adversary=0,
                )
                procObj.partial_clear(basePath)
                    
                        