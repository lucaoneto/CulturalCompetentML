#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"

import sys
import os
import gc
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.insert(1, "../")
from Processing.processing import ProcessingClass

# --- GPU Configuration ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

memory_limit = 6000
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
        )
    except RuntimeError as e:
        print(e)

# --- Hyperparameters ---
percent = 0.05
n = 1000
g_aug = 0.1  
ep = 0.2
basePath = "./try4/"
verbose_param = 1

# --- Mapping Table ---
# (standard, lamp, culture, diffusion, only_min, parify, augment)
# standard=0 (MIT) | standard=1 (Control)
# Requirement: If DIFF=1, then Augment=1
todo_configs = [
    # --- STD Group (standard=0) ---
    (1, 0, 0, 0, 0, 0, 0), # STD -> CI 
    (1, 0, 1, 0, 0, 0, 0), # STD -> CJ 
    (1, 0, 2, 0, 0, 0, 0), # STD -> CS 
    (1, 1, 0, 0, 0, 0, 0), # STD -> LC 
    (1, 1, 1, 0, 0, 0, 0), # STD -> LF 
    (1, 1, 2, 0, 0, 0, 0), # STD -> LT 
]

todo_configs = todo_configs[::-1]

# --- Execution Loop ---
for i in range(2): 
 for cls_div in [0,1]:
    random.seed(int(datetime.now().timestamp()))
    tf.random.set_seed(int(datetime.now().timestamp()))
    
    for std, lp, cult, diff, omin, par, aug in todo_configs:
        print(f"\n[Iteration {i}] Std:{std} | L:{lp} | C:{cult} | Diff:{diff} | OMin:{omin} | Aug:{aug}")
        
        procObj = ProcessingClass(
            shallow=0,
            lamp=lp,
            gpu=False,
            memory_limit=memory_limit,
            basePath=basePath,
        )

        procObj.process(
            standard=std,
            type="DL",
            verbose_param=verbose_param,
            culture=cult,
            percent=percent,
            n=n,
            augment=0,
            gaug=0,
            adversary=1,
            eps=ep,
            class_division=cls_div,
            imbalanced=0, 
            diffusion=0,
        )

        procObj.test(
            standard=std,
            culture=cult,
            augment=0,
            gaug=0,
            adversary=0,
        )

        
        # Cleanup
        procObj.partial_clear(basePath)
        del procObj
        gc.collect()
        tf.keras.backend.clear_session()

print("\n--- All 16 targeted experiments finished. ---")