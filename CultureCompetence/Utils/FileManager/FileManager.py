#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
import csv
import os
import pandas as pd


class FileManagerClass:
    """
    FileManagerClass is a class useful for writing and reading
    confusion matrices in files 
    """
    def __init__(self, name, create=True):
        """
        init functions stores the file name and, by defaults, makes directory
        if the directory does not exist
        :param name: name of the file to be used
        :param create: if enabled, it make the directory of the file if this does
        not exist
        """
        self.name = name
        dir = os.path.dirname(name)
        if create:
            self.mkdir(dir)

    def mkdir(self, dir):
        """
        makes the directory if this path does not exists
        :param dir: directory path
        """
        try:
            if not os.path.exists(dir):
                print(f"Making directory: {str(dir)}")
                os.makedirs(dir)
        except Exception as e:
            print(f"{dir} Not created")

    def readrows(self):
        """
        readrows opens a file and returns the rows of the file
        return: list of rows
        """
        csvlist = []
        try:
            with open(self.name, "r") as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    csvlist.append(row)                
                file.close()
        except:
            pass
            #print(f"Error in reading file {self.name}")
        return csvlist

    def writerow(self, row, discriminator=0):
        """
        writerow opens a file and write in it a row
        :param row: row to be saved in the file
        """
        try:
            with open(self.name, "a", newline="") as file:
                if discriminator:
                    row.to_csv(file)
                    file.write('\n')
                else:
                    writer = csv.writer(file)
                    writer.writerow(row)
                del row
                file.close()
        except Exception as e:
            print(f"Error in writing file {self.name} due to Exception:\n{e}")

    def writecm(self, cm, discriminator=0):
        """
        writecm creates a row from confusion matrix
        :param cm: confusion matrix to be stored
        """
        if discriminator:
            class_labels = ['Class 0', 'Class 1', 'Class 2']
            # Convert confusion matrices to pandas DataFrames
            row = pd.DataFrame(cm, columns=class_labels, index=class_labels)
            # Add a label to identify the matrix
            row['Matrix'] = 'CM'
            # Move the 'Matrix' column to the first position
            row = row[['Matrix'] + class_labels]
        else:
            row = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        self.writerow(row, discriminator=discriminator)
        del row

    def readcms(self):
        """
        readcms transform the rows in a file to confusion matrices
        :return list of confusion matrices in a file
        """
        cms = []
        rows = self.readrows()
        for row in rows:
            cm = [[int(row[0]), int(row[1])], [int(row[2]), int(row[3])]]
            cms.append(cm)
            del cm
        return cms

