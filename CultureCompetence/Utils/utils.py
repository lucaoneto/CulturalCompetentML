import csv
import numpy as np
import os


class FileClass:
    def __init__(self, name):
        self.name = name
        dir = os.path.dirname(name)
        self.mkdir(dir)

    def mkdir(self, dir):
        try:
            if not os.path.exists(dir):
                print(f'Making directory: {str(dir)}')
                os.makedirs(dir)
        except Exception as e:
            print(f'{dir} Not created')

    def readrows(self):
        csvlist = []
        try:
            with open(self.name, 'r') as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    csvlist.append(row)
        except:
            print(f'Error in reading file {self.name}')
        return csvlist

    def writerow(self, row):
        try:
            with open(self.name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
        except Exception as e:
            print(f'Error in writing file {self.name} due to Exception:\n{e}')

    def writecm(self, cm):
        row = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        self.writerow(row)

    def readcms(self):
        cms = []
        rows = self.readrows()
        for row in rows:
            cm = [[int(row[0]), int(row[1])], [int(row[2]), int(row[3])]]
            cms.append(cm)
        return cms


class ResultsClass:
    def calculate_percentage_confusion_matrix(self, confusion_matrix_list,
                                              tot):
        pcms = []
        for i in confusion_matrix_list:
            true_negative = (i[0, 0] / tot) * 100
            false_negative = (i[1, 0] / tot) * 100
            true_positive = (i[1, 1] / tot) * 100
            false_positive = (i[0, 1] / tot) * 100
            pcm = np.array([[true_negative, false_positive],
                            [false_negative, true_positive]])
            pcms.append(pcm)
        return pcms

    def return_tot_elements(self, cm):
        tot = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]
        return tot

    def return_statistics_pcm(self, pcms):
        max_true_negative = 0
        max_false_negative = 0
        max_true_positive = 0
        max_false_positive = 0
        min_true_negative = 100
        min_false_negative = 100
        min_true_positive = 100
        min_false_positive = 100
        count_true_negative = 0
        count_false_negative = 0
        count_true_positive = 0
        count_false_positive = 0
        for i in pcms:
            true_negative = i[0, 0]
            false_negative = i[1, 0]
            true_positive = i[1, 1]
            false_positive = i[0, 1]

            count_true_negative += true_negative
            count_false_negative += false_negative
            count_false_positive += false_positive
            count_true_positive += true_positive

            if true_negative > max_true_negative:
                max_true_negative = true_negative
            if false_negative > max_false_negative:
                max_false_negative = false_negative
            if true_positive > max_true_positive:
                max_true_positive = true_positive
            if false_positive > max_false_positive:
                max_false_positive = false_positive

            if true_negative < min_true_negative:
                min_true_negative = true_negative
            if false_negative < min_false_negative:
                min_false_negative = false_negative
            if true_positive < min_true_positive:
                min_true_positive = true_positive
            if false_positive < min_false_positive:
                min_false_positive = false_positive

        mean_true_negative = count_true_negative / len(pcms)
        mean_false_negative = count_false_negative / len(pcms)
        mean_true_positive = count_true_positive / len(pcms)
        mean_false_positive = count_false_positive / len(pcms)

        mean_matrix = np.array([[mean_true_negative, mean_false_positive],
                                [mean_false_negative, mean_true_positive]])
        max_matrix = np.array([[max_true_negative, max_false_positive],
                               [max_false_negative, max_true_positive]])
        min_matrix = np.array([[min_true_negative, min_false_positive],
                               [min_false_negative, min_true_positive]])

        matrix = []
        matrix.append(mean_matrix)
        matrix.append(max_matrix)
        matrix.append(min_matrix)
        return matrix
