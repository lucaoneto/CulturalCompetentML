import sys
sys.path.insert(1, '../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings

strings = Strings()
paths = strings.carpet_paths_str

cc = ClassificatorClass(1,0,paths,batch_size=2, fileName='c_jap.csv',
                         verbose = 0, validation_split=0.3, epochs=20,
                          learning_rate=5e-4, gpu=False, times=25)
cc.execute()