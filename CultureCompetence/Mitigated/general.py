#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import sys

sys.path.insert(1, "../")
from Processing.processing import ProcessingClass
from math import floor
import tensorflow as tf

tf.config.set_soft_device_placement(True)


def main():
    # percents = [0.05, 0.1]
    # standards = [0, 1]
    n = 1000
    bs = 2
    g_aug = 0.1
    test_g_augs = [0.01, 0.05, 0.1]
    verbose_param = 1
    learning_rate = 5e-4
    eps = 0.03
    test_eps = [0.0005, 0.001, 0.005]
    val_split = 0.2
    test_split = 0.1
    mult = 0.25
    epochs = 15
    procObj = ProcessingClass(shallow=0, lamp=1, gpu=True)

    standard = sys.argv[1]
    l = sys.argv[2]
    per = sys.argv[3]
    c = sys.argv[4]
    adv = sys.argv[5]

    with tf.device("/CPU:0"):
        print(f"Training->aug={adv%2};adv={floor(adv/2)}")
        procObj.process(
            standard=standard,
            type="DL",
            verbose_param=verbose_param,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=bs,
            lambda_index=l,
            culture=c,
            percent=per,
            val_split=val_split,
            test_split=test_split,
            n=n,
            augment=adv % 2,
            g_rot=g_aug,
            g_noise=g_aug,
            g_bright=g_aug,
            adversary=floor(adv / 2),
            eps=eps,
            mult=mult,
        )
        # NoAUg
        print(f"Testing->aug={0};adv={0}")
        procObj.test(
            standard=standard,
            culture=c,
            augment=0,
            g_rot=g_aug,
            g_noise=g_aug,
            g_bright=g_aug,
            adversary=0,
            eps=test_eps,
        )
        for t_g_aug in test_g_augs:
            procObj.test(
                standard=standard,
                culture=c,
                augment=1,
                g_rot=t_g_aug,
                g_noise=t_g_aug,
                g_bright=t_g_aug,
                adversary=0,
                eps=None,
            )
        for test_ep in test_eps:
            procObj.test(
                standard=standard,
                culture=c,
                augment=0,
                g_rot=None,
                g_noise=None,
                g_bright=None,
                adversary=1,
                eps=test_ep,
            )
        for t, t_g_aug in enumerate(test_g_augs):
            for test_ep in test_eps:
                procObj.test(
                    standard=standard,
                    culture=c,
                    augment=1,
                    g_rot=t_g_aug,
                    g_noise=t_g_aug,
                    g_bright=t_g_aug,
                    adversary=1,
                    eps=test_ep,
                )

        procObj.partial_clear()


if __name__ == "__main__":
    main()
