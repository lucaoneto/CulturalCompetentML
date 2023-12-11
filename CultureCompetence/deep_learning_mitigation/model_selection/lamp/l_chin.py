import sys
sys.path.insert(1, '../../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.lamp_paths
space = [0.1]
folders = ['0,1']
j = space[0]
cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=folders[0] + '/l_chin.csv', verbose = 0, validation_split=0.2, epochs=6, learning_rate=4e-4, percent=j, times=8)
cc.execute_model_selection()
'''
for i in range(0,len(space)):
    j = space[i]
    cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=folders[i] + '/l_chin.csv', verbose = 0, validation_split=0.2, epochs=6, learning_rate=4e-4, percent=j, times=1)
    cc.execute_model_selection()
'''
