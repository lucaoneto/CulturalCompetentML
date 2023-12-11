#! /usr/bin/env python3

import pathlib
import numpy as np
import skimage.color
import cv2
import math
import random
import time
import os, shutil
from PIL import Image
import random
from collections import deque
import tensorflow as tf

n_ims = 1000


class DSClass:
    # utils
    def options(self, options=[], default=1):
        if options == None:
            raise Exception('Options is None')
        if len(options) == 0:
            raise Exception('Options has length 0')
        print('Which one would you like to choose?')
        for i, opt in enumerate(options):
            print(f'{i+1}) {opt}')
        x = input('')
        for i, option in enumerate(options):
            if x.lower() == option.lower():
                return i + 1
        try:
            x = int(x)
            if x >= 1 and x <= len(options):
                return x
            else:
                return default
        except:
            return default

    def accept(self, text):
        x = input(text)
        # check if there is yes or no
        x = x.lower()
        if x == 'yes' or x == 'y':
            return True
        if x == 'no' or x == 'n':
            return False
        try:
            x = int(x)
            if x == 1:
                return True
            return False
        except:
            return False

    # OFFLINE
    # prepare path and images
    def delete_folder_content(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def mkdir(self, dir):
        if not os.path.exists(dir):
            print(f'Making directory: {str(dir)}')
            os.makedirs(dir)

    def paths(self):
        self.starting = input('Enter the starting path: ')
        self.destination = input('Enter the destination path: ')
        self.mkdir(self.destination)
        n = input('Enter the number of the labels, default is 2 (0,1): ')
        self.labels = []
        try:
            n = int(n)
            if n > 1:
                for i in range(n):
                    self.labels.append(input('Enter label name '))
            else:
                n = 2
                self.labels = ['0', '1']

        except:
            n = 2
            self.labels = ['0', '1']

    def acquire_images(self, path, grayscale=False):
        images = []
        types = ('*.png', '*.jpg', '*.jpeg')
        paths = []
        for typ in types:
            paths.extend(pathlib.Path(path).glob(typ))
        paths = paths[0:n_ims]
        for i in paths:
            if grayscale:
                im = cv2.imread(str(i), cv2.IMREAD_GRAYSCALE)
            else:
                im = cv2.imread(str(i))
            images.append(im)
        return images

    # transform images
    # assumption, starting point is rgb
    def modify_color(self, img):
        img = img[:, :, ::-1]
        # HSV color
        if self.color == 2:
            img = skimage.color.rgb2hsv(img)
        # grayscale is default color
        if self.color != 1 and self.color != 2:
            img = skimage.color.rgb2gray(img)
        return img

    def strc(self, img):
        im_dim = np.shape(img)
        dim = (self.size, self.size)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img

    def get_max_and_index(self, l):
        value = max(l)
        for i, el in enumerate(l):
            if el == value:
                index = i
        return index, value

    def get_dimensions(self, height, width):
        list_size = []
        list_size.append(math.floor((self.size - height) / 2))
        list_size.append(math.ceil((self.size - height) / 2))
        list_size.append(math.floor((self.size - width) / 2))
        list_size.append(math.ceil((self.size - width) / 2))
        return list_size

    def fill(self, img):
        im_dim = np.shape(img)
        index, value = self.get_max_and_index(im_dim)
        if index:
            dim = (math.floor((im_dim[0] * self.size) / im_dim[1]), self.size)
        else:
            dim = (self.size, math.floor((im_dim[1] * self.size) / im_dim[0]))
        img = cv2.resize(img, (dim[1], dim[0]), interpolation=cv2.INTER_AREA)
        dimensions = img.shape
        tblr = self.get_dimensions(dimensions[0], dimensions[1])
        img = cv2.copyMakeBorder(img,
                                 tblr[0],
                                 tblr[1],
                                 tblr[2],
                                 tblr[3],
                                 cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])
        return img

    def modify_size(self, img):
        if self.stretch:
            img = self.strc(img)
        else:
            img = self.fill(img)
        return img

    def modify(self, dataset, label):
        for i, img in enumerate(dataset):
            img = self.modify_size(img)
            img = self.modify_color(img)
            dest = self.destination + '/' + str(
                self.size) + '/' + self.opts[self.color - 1] + '/' + label
            self.mkdir(dest)
            im = Image.fromarray(np.uint8(img))
            im.save(dest + '/im' + str(i) + '.jpeg')
            #cv2.imwrite(dest + '/' + str(i) + '.jpg', img)

    def preferences(self):
        # get color and size and preferences
        self.opts = ['RGB', 'HSV', 'Greyscale']
        self.color = self.options(self.opts, default=3)
        size = input('Enter the size (default 33): ')
        try:
            self.size = int(size)
        except:
            self.size = 33
        self.stretch = self.accept(
            'Do you want to stretch the image (default, fill with white pixels) '
        )

    def prepare(self, arg=[], auto=False):
        self.paths()
        # prepare path and images
        if auto:
            if arg:
                self.color = arg['color']
                self.size = arg['size']
                self.stretch = arg['stretch']
            else:
                self.color = 3
                self.size = 33
                self.stretch = 0
        else:
            self.preferences()
        # transform images
        for label in self.labels:
            dataset = self.acquire_images(self.starting + '/' + label)
            self.modify(dataset, label)

    # ONLINE
    def get_labels(self, path):
        dir_list = []
        for file in os.listdir(path):
            d = os.path.join(path, file)
            if os.path.isdir(d):
                d = d.split('\\')
                if len(d) == 1:
                    d = d[0].split('/')
                d = d[-1]
                dir_list.append(d)
        return dir_list

    def splitting(self, path, i, label):
        images = self.acquire_images(path + '/' + label, self.greyscale)
        training = []
        test = []
        if len(images) > 1:
            for image in images[0:int(len(images) * self.proportion)]:
                #image = image.flatten()
                if self.greyscale:
                    image = image[0::]
                if self.flat:
                    image = image.flatten()
                training.append([image / 255, i])
            for image in images[int(len(images) *
                                    self.proportion):len(images) - 1]:
                #image = image.flatten()
                if self.greyscale:
                    image = image[0::]
                if self.flat:
                    image = image.flatten()
                test.append([image / 255, i])
        else:
            print(f'{path} is empty')
        return training, test
        #self.TS.append(training)
        #self.TestS.append(test)

    def build_dataset(self, paths, greyscale=0, flat=1, proportion = 0.8):
        random.seed(time.time_ns())
        self.proportion = proportion
        self.greyscale = greyscale
        self.flat = flat
        # for each path build a dataset
        # the dataset is divided into training and test set
        self.TS = []
        self.TestS = []
        tempTS = []
        tempTestS = []
        if paths != None:
            for path in paths:
                labels = self.get_labels(path)
                for i, label in enumerate(labels):
                    training, test = self.splitting(path, i, label)
                    tempTS = tempTS + training
                    tempTestS = tempTestS + test
                random.shuffle(tempTS)
                random.shuffle(tempTestS)
                self.TS.append(tempTS)
                self.TestS.append(tempTestS)
                tempTS = []
                tempTestS = []

        else:
            print('Paths is None')

    def nineonedivision(self, culture, percent=0.1, divide=0):
        if not divide:
            for i, ts in enumerate(self.TS):
                if i != culture:
                    self.TS[culture] = self.TS[culture] + ts[
                        0:int(len(ts) * percent)]
            for ts in self.TS:
                random.shuffle(ts)
        else:
            TS = self.TS
            for i, ts in enumerate(TS):
                if i != culture:
                    TS[i] = ts[0:int(len(ts) * percent)]
            self.TS = TS

    def mitigation_dataset(self, paths, greyscale=0, flat=1):
        self.build_dataset(paths, greyscale, flat)
        cultureID = 0
        for cultureTS in self.TS:
            for sample in cultureTS:
                y = (cultureID, sample[1])
                sample[1] = y
            cultureID = cultureID + 1
        cultureID = 0
        for cultureTestS in self.TestS:
            for sample in cultureTestS:
                y = (cultureID, sample[1])
                sample[1] = y
            cultureID = cultureID + 1

    def indipendent_dataset(self, culture, percent=0.1):
        TS = self.TS
        for i, ts in enumerate(TS):
            if i == culture:
                TS[i] = ts[0:int(len(ts) * percent)]
            else:
                TS[i] = 0
        self.TS = TS

