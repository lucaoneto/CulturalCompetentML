import sys

sys.path.insert(1, '../')
import DS.ds
from Utils.utils import FileClass, ResultsClass
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras import layers, optimizers
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
import keras.backend as K
from keras.models import Model
import gc
import os
import time


class ClassificatorClass:
    def __init__(self,
                 culture=0,
                 greyscale=0,
                 paths=None,
                 times=30,
                 fileName='results.csv',
                 validation_split=0.1,
                 batch_size=1,
                 epochs=10,
                 learning_rate=1e-3,
                 verbose=0,
                 plot = False,
                 gpu = True, 
                 percent = 0.1):
        self.culture = culture
        self.greyscale = greyscale
        self.paths = paths
        self.times = times
        self.fileName = fileName
        self.resultsObj = ResultsClass()
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose_param = verbose
        self.plot = plot
        self.gpu = gpu
        self.percent = percent
        if self.gpu:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
            # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2800)])
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(e)
            else:
                print('no gpus')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def prepareDataset(self, paths):
        datasetClass = DS.ds.DSClass()
        datasetClass.build_dataset(paths)
        self.TS = datasetClass.TS
        self.TestSet = datasetClass.TestS

    def quantize(self, yF):
        values = []
        for y in yF:
            if y>0.5:
                values.append(1)
            else:
                values.append(0)
        return values

    def test(self, model, testSet):
        testSet = np.array(testSet, dtype=object)
        XT = list(testSet[:, 0])
        yT = list(testSet[:, 1])
        XT = tf.stack(XT)
        yT = tf.stack(yT)

        yF = model.predict(XT)
        yF = self.quantize(yF)
        cm = confusion_matrix(yT, yF)
        return cm

    def save_cm(self, fileName, cm):
        f = FileClass(fileName)
        f.writecm(cm)

    def get_results(self, fileName):
        f = FileClass(fileName)
        return f.readcms()

    def plot_training(self):
        train_acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        train_loss = self.history.history['loss']
        train_acc_x = range(len(train_acc))
        val_acc_x = range(len(train_acc))
        train_loss_x = range(len(train_acc))
        val_loss_x = range(len(train_acc))

        plt.plot(train_acc_x, train_acc, marker = 'o', color = 'blue', markersize = 10, 
                        linewidth = 1.5, label = 'Training Accuracy')
        plt.plot(val_acc_x, val_acc, marker = '.', color = 'red', markersize = 10, 
                        linewidth = 1.5, label = 'Validation Accuracy')

        plt.title('Training Accuracy and Testing Accuracy w.r.t Number of Epochs')

        plt.legend()

        plt.figure()

        plt.plot(train_loss_x, train_loss, marker = 'o', color = 'blue', markersize = 10, 
                        linewidth = 1.5, label = 'Training Loss')
        plt.plot(val_loss_x, val_acc, marker = '.', color = 'red', markersize = 10, 
                        linewidth = 1.5, label = 'Validation Loss')

        plt.title('Training Loss and Testing Loss w.r.t Number of Epochs')

        plt.legend()

        plt.show()

    def train(self, TS):
        size = np.shape(TS[0][0])
        input = Input(size)
        x = tf.keras.Sequential([
            ResNet50(input_shape=size, weights='imagenet', include_top=False)
        ])(input)
        x = Flatten()(x)
        x = (Dense(1, activation='sigmoid'))(x)
        model = Model(inputs=input,
                    outputs = x,
                    name = 'model')
        model.trainable = True
        for layer in model.layers[1].layers:
            layer.trainable = False
        for layer in model.layers[1].layers[-1:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        lr_reduce = ReduceLROnPlateau(monitor='val_accuracy',
                                      factor=0.2,
                                      patience=3,
                                      verbose=self.verbose_param,
                                      mode='max',
                                      min_lr=1e-8)
        early = EarlyStopping(monitor='val_accuracy',
                              min_delta=0.001,
                              patience=8,
                              verbose=self.verbose_param,
                              mode='auto')
        adam = optimizers.Adam(self.learning_rate)
        optimizer = adam
        model.compile(loss="binary_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])

        TS = np.array(TS, dtype=object)
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        X_val = X[int(len(X) * (1-self.validation_split)):len(X) - 1]
        y_val = y[int(len(y) * (1-self.validation_split)):len(y) - 1]
        X = X[0:int(len(X) * (1-self.validation_split))]
        y = y[0:int(len(y) * (1-self.validation_split))]
        X = tf.stack(X)
        y = tf.stack(y)
        X_val = tf.stack(X_val)
        y_val = tf.stack(y_val)
        self.history = model.fit(X,y,
                                 epochs=self.epochs,
                                 validation_data=(X_val,y_val),
                                 callbacks=[early, lr_reduce],
                                 verbose=self.verbose_param,
                                 batch_size = self.batch_size)
        if self.plot:
            self.plot_training()
        return model

    def execute(self):
        for i in range(self.times):
            gc.collect()
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale, 0)
            # I have to select a culture
            TS = obj.TS[self.culture]
            # I have to test on every culture
            TestSets = obj.TestS
            # Name of the file management for results
            fileNames = []
            for l in range(len(TestSets)):
                name = self.fileName.split('.')[0] + str(
                    l) + '.csv'
                fileNames.append(name)

            model = self.train(TS)

            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                self.save_cm(fileNames[k], cm)
                cms.append(cm)
            # Reset Memory each time
            #gc.collect()

        #results = np.array(results, dtype = object)
        for i in range(len(obj.TS)):
            #result = results[:,i]
            result = self.get_results(fileNames[i])
            result = np.array(result, dtype=object)
            print(f'RESULTS OF CULTURE {i}')
            tot = self.resultsObj.return_tot_elements(result[0])
            pcm_list = self.resultsObj.calculate_percentage_confusion_matrix(
                result, tot)
            statistic = self.resultsObj.return_statistics_pcm(pcm_list)
            for j in statistic:
                print(j)
            accuracy = statistic[0][0][0] + statistic[0][1][1]
            print(f'Accuracy is {accuracy} %')

    def execute_mixed(self, cultures=[1]):
        # Name of the file management
        for i in range(self.times):
            print(f'CICLE {i}')
            gc.collect()
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale, 0)
            # I have to mix the cultures
            TS = []
            MixedTestSet = []
            TestSets = []
            cultureName = ""
            fileNames = []
            
            for culture in cultures:
                TS = TS + obj.TS[culture]
                MixedTestSet = MixedTestSet + obj.TestS[culture]
                cultureName = cultureName + str(culture)
            mixedName = self.fileName.split(
                '.')[0] + cultureName + self.fileName.split('.')[1]
            for i in range(len(obj.TestS)):
                if i not in cultures:
                    TestSets.append(obj.TestS[i])
                    name = self.fileName.split('.')[0] + str(
                        i) + self.fileName.split('.')[1]
                    fileNames.append(name)
            model = self.train(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                cms.append(cm)
                self.save_cm(fileNames[k], cm)
            self.save_cm(mixedName, self.test(model, MixedTestSet))

            # Reset Memory each time
            gc.collect()

        for i in range(len(fileNames)):
            print(f'RESULTS OF REMAIN CULTURE {i}')
            result = self.get_results(fileNames[i])
            result = np.array(result, dtype=object)
            tot = self.resultsObj.return_tot_elements(result[0])
            pcm_list = self.resultsObj.calculate_percentage_confusion_matrix(
                result, tot)
            statistic = self.resultsObj.return_statistics_pcm(pcm_list)
            for j in statistic:
                print(j)
            accuracy = statistic[0][0][0] + statistic[0][1][1]
            print(f'Accuracy is {accuracy} %')

        print('MIXED RESULTS')
        mixedResults = self.get_results(mixedName)
        print(np.shape(mixedResults))
        mixedResults = np.array(mixedResults, dtype = object)
        tot = self.resultsObj.return_tot_elements(mixedResults[0])
        pcm_list = self.resultsObj.calculate_percentage_confusion_matrix(
            mixedResults, tot)
        statistic = self.resultsObj.return_statistics_pcm(pcm_list)
        for j in statistic:
            print(j)
        accuracy = statistic[0][0][0] + statistic[0][1][1]
        print(f'Accuracy is {accuracy} %')
    

    def executenineone(self):
        for i in range(self.times):
            gc.collect()
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale, 0)
            obj.nineonedivision(self.culture, percent=self.percent)
            # I have to select a culture
            TS = obj.TS[self.culture]
            # I have to test on every culture
            TestSets = obj.TestS
            # Name of the file management for results
            fileNames = []
            for l in range(len(TestSets)):
                name = self.fileName.split('.')[0] + str(
                    l) + self.fileName.split('.')[1]
                fileNames.append(name)

            model = self.train(TS)

            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                self.save_cm(fileNames[k], cm)
                cms.append(cm)
            
            # Reset Memory each time
            gc.collect()

    def executenineone_model_selection(self):
        for i in range(self.times):
            gc.collect()
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale, 0)
            obj.nineonedivision(self.culture, percent=self.percent)
            # I have to select a culture
            TS = obj.TS[self.culture]
            # I have to test on every culture
            TestSets = obj.TestS
            print(np.shape(TestSets))
            # Name of the file management for results
            fileNames = []
            for l in range(len(TestSets)):
                name = self.fileName.split('.')[0] + str(
                    l) + self.fileName.split('.')[1]
                fileNames.append(name)

            model = self.model_selection(TS)

            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                self.save_cm(fileNames[k], cm)
                cms.append(cm)
            
            # Reset Memory each time
            gc.collect()

        #results = np.array(results, dtype = object)
        for i in range(len(obj.TS)):
            #result = results[:,i]
            result = self.get_results(fileNames[i])
            result = np.array(result, dtype=object)
            print(f'RESULTS OF CULTURE {i}')
            tot = self.resultsObj.return_tot_elements(result[0])
            pcm_list = self.resultsObj.calculate_percentage_confusion_matrix(
                result, tot)
            statistic = self.resultsObj.return_statistics_pcm(pcm_list)
            for j in statistic:
                print(j)
            accuracy = statistic[0][0][0] + statistic[0][1][1]
            print(f'Accuracy is {accuracy} %')

    def model_selection(self, TS):
        size = np.shape(TS[0][0])
        input = Input(size)
        x = tf.keras.Sequential([
            ResNet50(input_shape=size, weights='imagenet', include_top=False)
        ])(input)
        x = Flatten()(x)
        x = (Dense(1, activation='sigmoid'))(x)
        lr_reduce = ReduceLROnPlateau(monitor='val_accuracy',
                                      factor=0.2,
                                      patience=3,
                                      verbose=self.verbose_param,
                                      mode='max',
                                      min_lr=1e-8)
        early = EarlyStopping(monitor='val_accuracy',
                              min_delta=0.001,
                              patience=8,
                              verbose=self.verbose_param,
                              mode='auto')
        
        bs_list = [1,2,4] # batch size list
        lr_list = np.logspace(-5.5,-3,10)
        act_val_acc = 0
        for bs in bs_list:
            for lr in lr_list:
                if self.verbose_param:
                    print(f'For batch size = {bs} and learning rate = {lr}')
                model = Model(inputs=input,
                    outputs = x,
                    name = 'model')
                model.trainable = True
                for layer in model.layers[1].layers:
                    layer.trainable = False
                for layer in model.layers[1].layers[-1:]:
                    if not isinstance(layer, layers.BatchNormalization):
                        layer.trainable = True
                adam = optimizers.Adam(lr)
                optimizer = adam
                model.compile(loss="binary_crossentropy",
                            optimizer=optimizer,
                            metrics=["accuracy"])

                TS = np.array(TS, dtype=object)
                X = list(TS[:, 0])
                y = list(TS[:, 1])
                X_val = X[int(len(X) * (1-self.validation_split)):len(X) - 1]
                y_val = y[int(len(y) * (1-self.validation_split)):len(y) - 1]
                X = X[0:int(len(X) * (1-self.validation_split))]
                y = y[0:int(len(y) * (1-self.validation_split))]
                X = tf.stack(X)
                y = tf.stack(y)
                X_val = tf.stack(X_val)
                y_val = tf.stack(y_val)
                self.history = model.fit(X,y,
                                        epochs=self.epochs,
                                        validation_data=(X_val,y_val),
                                        callbacks=[early, lr_reduce],
                                        verbose=self.verbose_param,
                                        batch_size = bs)
                val_acc = self.history.history['val_accuracy']
                if act_val_acc < max(val_acc):
                    best_model = model
                if self.plot:
                    self.plot_training()
                model = None
                time.sleep(5)
        return best_model

    def execute_indipendent(self):
        for i in range(self.times):
            gc.collect()
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale, 0)
            obj.indipendent_dataset(self.culture, percent=self.percent)
            TS = obj.TS[self.culture]
            # I have to test on every culture
            TestSets = obj.TestS
            print(np.shape(TestSets))
            # Name of the file management for results
            fileNames = []
            for l in range(len(TestSets)):
                name = self.fileName.split('.')[0] + str(
                    l) + self.fileName.split('.')[1]
                fileNames.append(name)

            model = self.model_selection(TS)

            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                self.save_cm(fileNames[k], cm)
                cms.append(cm)
            
            # Reset Memory each time
            gc.collect()

        #results = np.array(results, dtype = object)
        for i in range(len(obj.TS)):
            #result = results[:,i]
            result = self.get_results(fileNames[i])
            result = np.array(result, dtype=object)
            print(f'RESULTS OF CULTURE {i}')
            tot = self.resultsObj.return_tot_elements(result[0])
            pcm_list = self.resultsObj.calculate_percentage_confusion_matrix(
                result, tot)
            statistic = self.resultsObj.return_statistics_pcm(pcm_list)
            for j in statistic:
                print(j)
            accuracy = statistic[0][0][0] + statistic[0][1][1]
            print(f'Accuracy is {accuracy} %')
