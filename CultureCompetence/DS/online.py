import sys
sys.path.insert(1, '../')
import DS.ds
import numpy as np
from matplotlib import pyplot as plt

paths = ['C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\turkish\\35\\Greyscale', 
         'C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\chinese\\35\\Greyscale',
         'C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\french\\35\\Greyscale']

obj = DS.ds.DSClass()
obj.build_dataset(paths, 1)

TS = obj.TS
XY_pair_sample = TS[0]
X = XY_pair_sample[0][0]
Y = XY_pair_sample[0][1]

print(np.shape(TS))
print(np.shape(TS[0]))
print(np.shape(TS[0][0]))
print(np.shape(TS[0][0][0]))
print(np.shape(TS[0][0][1]))
print(f'The image is classified as {Y}')


obj2 = DS.ds.DSClass()
obj2.mitigation_dataset(paths, 1)
obj2.nineonedivision(0)
y = obj2.TS[0][0][1]
print(f'y of the mitigation dataset is {y}')