#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys


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
import cv2

random.seed(datetime.now().timestamp())
tf.random.set_seed(datetime.now().timestamp())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# tf.config.set_soft_device_placement(True)

memory_limit = 9000
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


percent = 0.05
standards = [0, 1]
# lamp = 1

verbose_param = 1
n = 1000
class_divisions = [0, 1]
imb = 0
g_augs = np.logspace(-2, -1, 2)
eps = np.logspace(-2, -1, 2)
cs = [ 0, 1, 2]
lamps = [0, 1]

diffusions = [1, 0]
ks = [0, 1]
adv = 0
parify_batches_diffusion = 0
only_mins = [0]

basePath = "./try2/"
for lamp in lamps:
  for diffusion in diffusions:
   for standard in standards:
    for adv in [1]:
     for k in ks:
        if (adv and not standard) or (diffusion and not k) or (diffusion and imb) or (diffusion and adv) or (not diffusion and not adv) or (k and adv) or (adv and parify_batches_diffusion):
          break
        else:
            for c in cs:
                for only_min in only_mins:
                    for cl_div in class_divisions:
                        for g_aug in g_augs:
                            for ep in eps:
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
                                    augment=k,
                                    gaug=g_aug,
                                    adversary=adv,
                                    eps =ep,
                                    class_division=cl_div,
                                    imbalanced=imb, 
                                    diffusion = diffusion,
                                    only_minority_diffusion=only_min,
                                    parify_batches_diffusion=parify_batches_diffusion,
                                    mitigation_type=1
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
                                    
                                        