import sys
sys.path.insert(1, '../')
from Utils import utils
import csv
import os
import pathlib
import cv2

def get_paths(path):
        types = ('*.png', '*.jpg', '*.jpeg')
        paths = []
        rel_paths = []
        for typ in types:
            paths.extend(pathlib.Path(path).glob(typ))
        for path in paths:
            rel_paths.append(os.path.relpath(path, start = os.curdir))
        return rel_paths
try:
    os.remove('LAMP.csv')
    os.remove('CARPET.csv')
except:
    ...
# Create csv for LAMP dataset
first_row = ['Culture', 'Image', 'Class']
try:
    with open('LAMP.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_row)
except Exception as e:
    print(f'Error in writing file LAMP.CSV due to Exception:/n{e}')
# Create csv for CARPET dataset
try:
    with open('CARPET.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_row)
except Exception as e:
    print(f'Error in writing file LAMP.CSV due to Exception:/n{e}')

ds_paths = 'C:/Users/enzop/Desktop/FINALDS/originals'
path_to_lamps = ds_paths + '/lamps'
path_to_carpets = ds_paths + '/carpets'

chinese_n = get_paths(path_to_lamps + '/chinese/on')
chinese_f = get_paths(path_to_lamps + '/chinese/off')

french_n = get_paths(path_to_lamps + '/french/on')
french_f = get_paths(path_to_lamps + '/french/off')

turkish_n = get_paths(path_to_lamps + '/turkish/on')
turkish_f = get_paths(path_to_lamps + '/turkish/off')

indian_p = get_paths(path_to_carpets + '/indian/with')
indian_a = get_paths(path_to_carpets + '/indian/without')

japanese_p = get_paths(path_to_carpets + '/japanese/with')
japanese_a = get_paths(path_to_carpets + '/japanese/without')

scandinavian_p = get_paths(path_to_carpets + '/scandinavian/with')
scandinavian_a = get_paths(path_to_carpets + '/scandinavian/without')

try:
    with open('LAMP.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for c in chinese_n:
            writer.writerow(['C', c, 1])
        for c in chinese_f:
            writer.writerow(['C', c, 0])
        for c in french_n:
            writer.writerow(['F', c, 1])
        for c in french_f:
            writer.writerow(['F', c, 0])
        for c in turkish_n:
            writer.writerow(['T', c, 1])
        for c in turkish_f:
            writer.writerow(['T', c, 0])
except Exception as e:
    print(f'Error in writing file LAMP.CSV due to Exception:/n{e}')

try:
    with open('CARPET.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for c in indian_p:
            writer.writerow(['C', c, 1])
        for c in indian_a:
            writer.writerow(['C', c, 0])
        for c in japanese_p:
            writer.writerow(['F', c, 1])
        for c in japanese_a:
            writer.writerow(['F', c, 0])
        for c in scandinavian_p:
            writer.writerow(['T', c, 1])
        for c in scandinavian_a:
            writer.writerow(['T', c, 0])
except Exception as e:
    print(f'Error in writing file LAMP.CSV due to Exception:/n{e}')





