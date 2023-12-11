import sys

sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings
import numpy as np
import gc
import time

def run(culture, paths, file_name, i, percent):
    cc = ClassificatorClass(culture,
                                0,
                                paths,
                                batch_size=4,
                                fileName=file_name,
                                verbose=0,
                                plot=0,
                                validation_split=0.2,
                                epochs=40,
                                learning_rate=6e-5,
                                lambda_index=i,
                                times=10,
                                percent=percent)
    cc.execute()
    cc = None
    time.sleep(2)

strings = Strings()
paths = strings.lamp_paths
file_name = 'l_chin.csv'
culture = 0
for i in range(5, 25):
        percent = 0.05
        run(culture, paths, file_name, i, percent)
        
