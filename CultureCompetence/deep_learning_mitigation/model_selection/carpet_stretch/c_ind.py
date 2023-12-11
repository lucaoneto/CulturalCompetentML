import sys
sys.path.insert(1, '../../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings
import time

strings = Strings()
paths = strings.carpet_paths_str
space = [0.0, 0.05, 0.1]
folders = ['0,1']
j = space[0]
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=folders[0] + '/c_ind.csv', verbose = 0, validation_split=0.3, epochs=7, learning_rate=4e-4, percent=j, times=10)
cc.execute_model_selection()
cc = None
time.sleep(5)
j = space[1]
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=folders[0] + '/c_ind.csv', verbose = 0, validation_split=0.3, epochs=7, learning_rate=4e-4, percent=j, times=10)
cc.execute_model_selection()
cc = None
time.sleep(5)
j = space[2]
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=folders[0] + '/c_ind.csv', verbose = 0, validation_split=0.3, epochs=7, learning_rate=4e-4, percent=j, times=10)
cc.execute_model_selection()
cc = None
time.sleep(5)
'''
for i in range(0,len(space)):
    j = space[i]
    cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=folders[i] + '/l_chin.csv', verbose = 0, validation_split=0.2, epochs=6, learning_rate=4e-4, percent=j, times=1)
    cc.execute_model_selection()
'''
