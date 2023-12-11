import sys
sys.path.insert(1, '../../../')

from standard.classificator import ClassificatorClass
from standard.strings import Strings

strings = Strings()
paths = strings.paths

cc = ClassificatorClass(2,1,paths,'SVC',20,'rbf',times=6, fileName='rbf_tur.csv')
cc.executenineone()