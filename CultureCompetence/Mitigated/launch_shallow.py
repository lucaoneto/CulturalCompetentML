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


# tf.config.set_soft_device_placement(True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


percents = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
standard = 1
tps = ["DL"]
kernels = ["linear", "rbf"]
points = 7
# lamp = 1

verbose_param = 1
n = 1000
cs = [0, 1, 2]
lamps = [1, 0]


basePath = "./first_comparing_with_teacher/"
for i in range(5):
   for percent in percents:
      for lamp in lamps:
        for c in cs:
          for tp in tps:
                if tp == "DL":
                   shallow=0
                else:
                   shallow=1
                procObj = ProcessingClass(
                    shallow=shallow,
                    lamp=lamp,
                    gpu=True,
                    memory_limit=8000,
                    basePath=basePath,
                )
                print(f"Training->Model={tp}")
                procObj.process(
                    standard=standard,
                    type=tp,
                    points = points,
                    verbose_param=verbose_param,
                    culture=c,
                    percent=percent,
                    n=n,
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
                del procObj
                gc.collect()
                tf.keras.backend.clear_session()
                    
                        