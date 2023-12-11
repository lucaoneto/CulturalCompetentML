import sys

sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings
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
                                times=12,
                                percent=percent)
    cc.execute()
    cc = None
    time.sleep(2)

strings = Strings()
paths = strings.lamp_paths
file_name = 'l_fren.csv'
space = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9]
culture = 1

for i in range(9, 25):
        percent = space[j]
        run(culture, paths, file_name, i, percent)