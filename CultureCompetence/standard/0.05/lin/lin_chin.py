import sys
sys.path.insert(1, '../../../')

from standard.classificator import ClassificatorClass
from standard.strings import Strings

strings = Strings()
paths = strings.paths

cc = ClassificatorClass(0,1,paths,'SVC',20,'linear', times = 10, fileName='lin_chin.csv')
cc.executenineone(percent=0.05)