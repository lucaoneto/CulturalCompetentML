import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(1, '../')
import DS.ds
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from Utils.utils import FileClass, ResultsClass


class ClassificatorClass:
    def __init__(self,
                 culture=0,
                 greyscale=0,
                 paths=None,
                 type='SVC',
                 points=50,
                 kernel='linear',
                 times=10,
                 fileName='results.csv'):
        self.culture = culture
        self.greyscale = greyscale
        self.paths = paths
        self.type = type
        self.points = points
        self.kernel = kernel
        self.times = times
        self.fileName = fileName
        self.resultsObj = ResultsClass()

    def prepareDataset(self, paths):
        datasetClass = DS.ds.DSClass()
        datasetClass.build_dataset(paths)
        self.TS = datasetClass.TS
        self.TestSet = datasetClass.TestS

    def SVC(self, TS):
        if self.kernel == 'rbf':
            logspaceC = np.logspace(
                -4, 3, self.points)  #np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(
                -4, 3, self.points)  #np.logspace(-2,2,self.points)
            grid = {
                'C': logspaceC,
                'kernel': [self.kernel],
                'gamma': logspaceGamma
            }
        if self.kernel == 'linear':
            logspaceC = np.logspace(
                -4, 3, self.points)  #np.logspace(-2,2,self.points)
            logspaceGamma = np.logspace(
                -4, 3, self.points)  #np.logspace(-2,2,self.points)
            grid = {'C': logspaceC, 'kernel': [self.kernel]}

        MS = GridSearchCV(estimator=SVC(),
                          param_grid=grid,
                          scoring='balanced_accuracy',
                          cv=10,
                          verbose=0)
        # training set is divided into (X,y)
        TS = np.array(TS, dtype=object)
        X = list(TS[:, 0])
        y = list(TS[:, 1])
        print('SVC TRAINING')
        H = MS.fit(X, y)
        # Check that C and gamma are not the extreme values
        print(f"C best param {H.best_params_['C']}")
        #print(f"gamma best param {H.best_params_['gamma']}")

        return H

    def RFC(self, TS):
        rfc = RandomForestClassifier(random_state=42)
        logspace_max_depth = []
        for i in np.logspace(0, 3, self.points):
            logspace_max_depth.append(int(i))
        param_grid = {
            'n_estimators': [500],  #logspace_n_estimators,
            'max_depth': logspace_max_depth,
        }

        CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        # training set is divided into (X,y)
        TS = np.array(TS, dtype=object)
        X = list(TS[:, 0])
        y = list(TS[:, 1])

        print('RFC TRAINING')
        H = CV_rfc.fit(X, y)

        print(CV_rfc.best_params_)

        return H

    def test(self, model, testSet):
        testSet = np.array(testSet, dtype=object)
        XT = list(testSet[:, 0])
        yT = list(testSet[:, 1])

        yF = model.predict(XT)
        cm = confusion_matrix(yT, yF)
        return cm

    def save_cm(self, fileName, cm):
        f = FileClass(fileName)
        f.writecm(cm)

    def get_results(self, fileName):
        f = FileClass(fileName)
        return f.readcms()

    def execute(self):
        for i in range(self.times):
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale)
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
            if self.type == 'SVC':
                model = self.SVC(TS)
            elif self.type == 'RFC':
                model = self.RFC(TS)
            else:
                model = self.SVC(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                self.save_cm(fileNames[k], cm)
                cms.append(cm)
            #results.append(cms)

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
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale)
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

            if self.type == 'SVC':
                model = self.SVC(TS)
            elif self.type == 'RFC':
                model = self.RFC(TS)
            else:
                model = self.SVC(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                cms.append(cm)
                self.save_cm(fileNames[k], cm)

            self.save_cm(mixedName, self.test(model, MixedTestSet))

        for i in range(len(fileNames)):
            print(f'RESULTS OF CULTURE {i}')
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
        mixedResults = np.array(mixedResults, dtype=object)
        tot = self.resultsObj.return_tot_elements(mixedResults[0])
        pcm_list = self.resultsObj.calculate_percentage_confusion_matrix(
            mixedResults, tot)
        statistic = self.resultsObj.return_statistics_pcm(pcm_list)
        for j in statistic:
            print(j)
        accuracy = statistic[0][0][0] + statistic[0][1][1]
        print(f'Accuracy is {accuracy} %')

    def executenineone(self, percent=0.1):
        for i in range(self.times):
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale)
            obj.nineonedivision(self.culture, percent=percent)
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
            if self.type == 'SVC':
                model = self.SVC(TS)
            elif self.type == 'RFC':
                model = self.RFC(TS)
            else:
                model = self.SVC(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                self.save_cm(fileNames[k], cm)
                cms.append(cm)
            #results.append(cms)

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

    def execute_indipendent(self, percent=0.1):
        for i in range(self.times):
            print(f'CICLE {i}')
            obj = DS.ds.DSClass()
            obj.build_dataset(self.paths, self.greyscale)
            obj.indipendent_dataset(self.culture, percent=percent)
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
            if self.type == 'SVC':
                model = self.SVC(TS)
            elif self.type == 'RFC':
                model = self.RFC(TS)
            else:
                model = self.SVC(TS)
            cms = []
            for k, TestSet in enumerate(TestSets):
                cm = self.test(model, TestSet)
                self.save_cm(fileNames[k], cm)
                cms.append(cm)
            #results.append(cms)

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
