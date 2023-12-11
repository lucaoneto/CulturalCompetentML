import sys
sys.path.insert(1, '../../')

from deep_learning_mitigation.classificator import ClassificatorClass
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.carpet_paths_str
file_name = 'c_ind.csv'
for i in range(25):
    cc = ClassificatorClass(0, 0, paths,batch_size=4, fileName=file_name + str(i), lambda_index=i, verbose = 1, validation_split=0.2, epochs=15, learning_rate=4e-5)
    cc.execute()