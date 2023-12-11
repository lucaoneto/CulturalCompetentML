import sys
sys.path.insert(1, '../../')

from deep_learning.classificator import ClassificatorClass
from deep_learning.strings import Strings

strings = Strings()
paths = strings.lamp_paths

cc = ClassificatorClass(2,0,paths,batch_size=4, fileName='l_tur.csv', verbose = 1, validation_split=0.2, epochs=1, learning_rate=4e-4, times=2)
cc.execute()