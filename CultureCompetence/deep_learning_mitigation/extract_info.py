import sys
sys.path.insert(1, '/../')
from classificator import ClassificatorClass
import numpy as np

def print_accuracies_of_percent(base, root):
    cc = ClassificatorClass()
    for lambda_index in range(-1, 25):
        fileNames = []
        print(f'FOR LAMBDA INDEX EQUAL {lambda_index}')
        for l in range(3):
                    fileNamesOut = []
                    for o in range(3):
                        name = root + str(lambda_index) + '/' + base + str(
                            l) + f'/out{o}.csv'
                        
                        fileNamesOut.append(name)
                    fileNames.append(fileNamesOut)
        for i in range(3):
            for o in range(3):
                result = cc.get_results(fileNames[i][o])
                result = np.array(result, dtype=object)
                print(f'RESULTS ON CULTURE {i}, out {o}')
                tot = cc.resultsObj.return_tot_elements(result[0])
                pcm_list = cc.resultsObj.calculate_percentage_confusion_matrix(
                    result, tot)
                statistic = cc.resultsObj.return_statistics_pcm(pcm_list)
                accuracy = statistic[0][0][0] + statistic[0][1][1]
                print(f'Accuracy is {accuracy} %')

def get_accuracies_of_percent(base, root):
    cc = ClassificatorClass()
    accs = []
    for lambda_index in range(-1, 25):
        try:
            fileNames = []
            #print(f'FOR LAMBDA INDEX EQUAL {lambda_index}')
            for l in range(3):
                        fileNamesOut = []
                        for o in range(3):
                            name = root + str(lambda_index) + '/' + base + str(
                                l) + f'/out{o}.csv'
                            
                            fileNamesOut.append(name)
                        fileNames.append(fileNamesOut)
            acc_per_model = []
            for i in range(3):
                acc_per_out = []
                for o in range(3):
                    result = cc.get_results(fileNames[i][o])
                    result = np.array(result, dtype=object)
                    #print(f'RESULTS ON CULTURE {i}, out {o}')
                    tot = cc.resultsObj.return_tot_elements(result[0])
                    pcm_list = cc.resultsObj.calculate_percentage_confusion_matrix(
                        result, tot)
                    statistic = cc.resultsObj.return_statistics_pcm(pcm_list)
                    accuracy = statistic[0][0][0] + statistic[0][1][1]
                    #print(f'Accuracy is {accuracy} %')
                    acc_per_out.append(accuracy)
                acc_per_model.append(acc_per_out)
            accs.append(acc_per_model)
        except:
             pass
    return accs

def get_max_acc_per_lambda(accs):
    accs = np.asarray(accs, dtype=object)
    cul_acc = []
    for i in range(np.shape(accs)[1]):
        cul_acc.append(max(accs[:,i,i]))
    
    print(cul_acc)
    return cul_acc


base = 'l_fren'
root = 'lamp/'
space = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9]
for j in range(0,6):
    percent = space[j]
    rootact = root + 'percent' + str(percent).replace('.', ',')[0:3] + '/'
    accs = get_accuracies_of_percent(base, rootact)
    get_max_acc_per_lambda(accs)
