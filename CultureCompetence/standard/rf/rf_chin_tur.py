import sys
sys.path.insert(1, '../../')

from standard.classificator import ClassificatorClass
from standard.strings import Strings

strings = Strings()
paths = strings.paths

cc = ClassificatorClass(0,1,paths,'RFC',30, fileName='rf_chin_tur.csv')
cc.execute_mixed([0,2])