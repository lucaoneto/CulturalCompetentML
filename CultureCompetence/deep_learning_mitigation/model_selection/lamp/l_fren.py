import sys
sys.path.insert(1, '../../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.lamp_paths
space = [0.0, 0.05, 0.1]
for i in range(1,len(space)):
    j = space[i]
    cc = ClassificatorClass(1,0,paths,batch_size=4, fileName='l_fren.csv', verbose = 0, validation_split=0.2, epochs=10, learning_rate=4e-4, percent=j)
    cc.execute_model_selection()