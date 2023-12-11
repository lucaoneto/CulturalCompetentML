import sys
sys.path.insert(1, '../../')

from standard.classificator import ClassificatorClass
from standard.strings import Strings

strings = Strings()
paths = strings.paths

cc = ClassificatorClass(0,1,paths,'SVC',30,'rbf', times = 30, fileName='rbf_fren_tur.csv')
cc.execute_mixed([1,2])