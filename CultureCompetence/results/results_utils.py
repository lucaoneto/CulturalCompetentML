import sys

sys.path.insert(1, '../')
from numpy import abs
import numpy as np
import os
import pathlib
from Utils.utils import FileClass, ResultsClass
import glob
import colorama
from colorama import Fore


def l_value(index):
    ls = np.logspace(0, 2, 25)
    return ls[index]


## CIC formula is:
# CIC = Ehat_C {|ERR^C - min_C'\inCit{ERR^C'}|}
# Ehat_C = 1/|Cit| * sum over C\inCit
def calc_CIC(ERRs):
    factor = 1 / len(ERRs)
    CIC = 0
    for i in range(len(ERRs)):
        CIC += abs(ERRs[i] - min(ERRs))
    CIC = factor * CIC
    return CIC


def calc_ERR(acc):
    return 1 - acc


def calc_std_dev(errors):
    return np.std(errors)


def is_out(path, model):
    l = path.split(model)
    n = l[1][0]
    out = l[1][1:len(l[1])]
    return n in out


def retrieve_accs_lamb(path, model, ns):
    paths = []
    accs = []
    for j in range(3):
        paths.extend(glob.glob(f'{path}/{model}{j}/' + '*.csv'))
    sorted(paths)
    rc = ResultsClass()
    for p in paths:
        if model in p:
            if is_out(p, model):
                fc = FileClass(p)
                accs_temp = fc.readcms()
                accs_temp = np.asarray(accs_temp[0:ns], dtype=object)
                tot = rc.return_tot_elements(accs_temp[0])
                accs_temp = rc.calculate_percentage_confusion_matrix(
                    accs_temp, tot)
                temp = []
                for i in range(ns):
                    temp.append(accs_temp[i][0][0] + accs_temp[i][1][1])
                accs.append(temp)
    return accs


def retrieve_accs_pu(path, model, ns):
    lamb_accs = []
    for lamb in range(0, 25):
        try:
            pth = path + f'/{lamb}'
            lamb_accs.append(retrieve_accs_lamb(pth, model, ns))
        except:
            print(f'Missing data for lamb index = {lamb}')
    return lamb_accs


def retrievs_accs(path, model, ns, pu):
    pu_accs = []
    for p in pu:
        try:
            pth = path + f'/percent{p}'
            pu_accs.append(retrieve_accs_pu(pth, model, ns))
        except:
            print(f'Missing data for pu value = {p}')
    return pu_accs


def retrieve_mean_dev_std(l):
    m = np.mean(l)
    s = 0
    for i in l:
        s = s + (i - m) * (i - m)
    var = s / len(l)
    std = np.sqrt(var)
    return m, std


def get_err_std_for_every_lambda(accs_pu_cult):
    lerrs = []  # errors for each lambda
    lstds = []  # stds for each lambd
    try:
        for i in range(len(accs_pu_cult)):  # for each lambda
            rrs, stds = [], []
            for j in range(3):  # for each culture
                acc, std = retrieve_mean_dev_std(accs_pu_cult[i][j])
                err = calc_ERR(acc / 100)
                rrs.append(err)
                stds.append(std / 100)
            lerrs.append(rrs)
            lstds.append(stds)
    except:
        ...
    return lerrs, lstds


def get_CIC_std_for_every_lambda(accs_pu_cult):
    lCIC = []  # CIC for each lambda
    lstds = []  # stds for each lambd
    try:
        for i in range(len(accs_pu_cult)):  # for each lambda
            accs = np.asarray(accs_pu_cult[i], dtype=object).T
            CICs = []
            for j in range(len(accs)):  # for each sample
                errs = []
                for k in range(3):
                    er = calc_ERR(accs[j][k] / 100)
                    errs.append(er)
                c = calc_CIC(errs)
                CICs.append(c)

            CIC, std = retrieve_mean_dev_std(CICs)
            lCIC.append(CIC)
            lstds.append(std)
    except:
        ...
    return lCIC, lstds


def get_errs_stds_for_every_lambda(accs_pu, pu):
    clerrs = []  # errors for each lambda for each culture
    clstds = []  # stds for each lambda for each cultures
    accs_pu = np.asarray(accs_pu, dtype=object)
    try:
        for i in range(len(pu)):
            lerrs, lstds = get_err_std_for_every_lambda(accs_pu[i])
            clerrs.append(lerrs)
            clstds.append(lstds)
    except:
        ...
    return clerrs, clstds


def get_CICs_stds_for_every_lambda(accs_pu, pu):
    clCICs = []  # errors for each lambda for each culture
    clstds = []  # stds for each lambda for each cultures
    accs_pu = np.asarray(accs_pu, dtype=object)
    try:
        for i in range(len(pu)):
            lCICs, lstds = get_CIC_std_for_every_lambda(accs_pu[i])
            clCICs.append(lCICs)
            clstds.append(lstds)
    except:
        ...
    return clCICs, clstds


def get_lamb_for_min_CIC(errs_pu):
    clerrs = []  # errors for each lambda for each culture
    clstds = []  # stds for each lambda for each cultures
    errs_pu = np.asarray(errs_pu, dtype=object)
    errs_pu = list(errs_pu.T)
    for i in range(3):
        lerrs, lstds = get_err_std_for_every_lambda(errs_pu[i])
        clerrs.append(lerrs)
        clstds.append(lstds)
    CICs = []
    for i in range(len(errs_pu[0])):
        errors = []
        for j in range(3):
            errors.append(errs_pu[j][i])
        CIC = calc_CIC(errors)
        CICs.append(CIC)
    minCIC = min(CICs)
    lambda_index = CICs.index(minCIC)
    return lambda_index


def get_lamb_for_min_err(errs_pu, culture):
    errs_pu = np.asarray(errs_pu, dtype=object)
    errs_pu = list(errs_pu.T)
    errs_pu_c = list(errs_pu[culture])
    minimum = min(errs_pu_c)
    lambda_index = errs_pu_c.index(minimum)
    return lambda_index


def print_stats(errs_pu, stds_pu, lamb, CICs, stdsCIC, tau):
    print(
        f'LAMBDA INDEX = {lamb} LAMBDA VALUE = {l_value(lamb)} AND TAU = {tau}'
    )
    errors = []
    for j in range(3):
        #acc, _ = retrieve_mean_dev_std_accs_pu(accs[j][lamb])
        #print(f'Accuracy is {acc:.1f}+-{stds_pu[j][lamb]:.1f} on Culture {j}')
        print(
            f'Error is {errs_pu[lamb][j]*100:.1f}%+-{stds_pu[lamb][j]*100:.1f}% on Culture {j}'
        )
        errors.append(errs_pu[lamb][j])
    #CIC = calc_CIC(errors)
    print(f'Mean error is {np.mean(errs_pu[lamb])*100:.1f}')
    print(
        f'CIC is {CICs[lamb]*100:.1f}%+-{stdsCIC[lamb]*100:.1f}% on culture {j}\n'
    )


def get_lamb_metric(errs_pu, tau, CICs):
    lerrs = []
    for i in range(len(errs_pu)):  # per each lambda
        m = np.mean(errs_pu[i])  # ERR metric
        lerrs.append(m)
    #lambdas = np.argwhere(lerrs < (1 + tau) * min(lerrs))
    values = [x for x in lerrs if x < tau]
    lambdas = []
    for value in values:
        lambdas.append(lerrs.index(value))
    #lambdas = np.argwhere( )
    #print(tau)
    #print(lerrs)
    cs = []
    if len(lambdas) > 0:
        #if len(lambdas[0]) > 0:
            for l in lambdas:  # select the CICs that are in the ball desiderd
                cs.append(CICs[l])
            optIndex = np.argmin(cs)
            optIndex = list(CICs).index(cs[optIndex])
            #optIndex = np.where(CICs == cs[optIndex])
            optValue = CICs[optIndex]
            #print(f'lerrs are {lerrs}')
            #print(f'minimum err is {min(lerrs)}')
            #print(f'lambdas are {lambdas}')
            #print(f'rrs are {cs}')
            #print(f'optIndex is {optIndex}')
            #print(f'optValue is {optValue}')
            return optIndex
    return -1


def retrieve_statistics(p, model, ns, pu):
    print(Fore.WHITE + f'MODEL IS {model}')
    accs = retrievs_accs(p, model, ns, pu)
    clerrs, clstds = get_errs_stds_for_every_lambda(accs, pu)
    CICs, stdsCIC = get_CICs_stds_for_every_lambda(accs, pu)
    #print(CICs)
    #print(clerrs)
    for i in range(len(pu)):
        print(f'\nREFERRING TO PU={pu[i]}')
        # I want lambda for min errs (for each culture) and min CIC and their values
        accs_pu = accs[i]
        errs_pu, stds_pu = clerrs[i], clstds[i]
        CICs_pu, stds_CIC_pu = CICs[i], stdsCIC[i]
        lambdas = []
        taus = [0.1, 0.15, 0.2, 0.5,
                0.8]  # 5%, 10% and 20% of the minimum error
        if len(errs_pu) > 0:
            # Calculate the minimum CIC in the first tau% of errors
            for tau in taus:
                lamb = get_lamb_metric(errs_pu, tau, CICs_pu)
                if lamb >= 0:
                    print_stats(errs_pu, stds_pu, lamb, CICs_pu, stds_CIC_pu, tau)
                else:
                    print(f'For tau = {tau} no ERR detected')
            '''
            for j in range(3):
                print(f'LAMBDA FOR MINIMUM ERROR ON CULTURE {j}')
                l = get_lamb_for_min_err(errs_pu, j)
                print_stats(errs_pu, stds_pu, l, accs_pu)
                lambdas.append(l)
            l = get_lamb_for_min_CIC(errs_pu)
            lambdas.append(l)
            print('LAMBDA FOR MINIMUM CIC')
            print_stats(errs_pu, stds_pu, l, accs_pu)

            print('PRINT LAMBDA FOR SOME FIXED VALUES OF LAMBDA')
            try:
                print_stats(errs_pu, stds_pu, 0, accs_pu)
            except:
                ...
            try:
                print_stats(errs_pu, stds_pu, 13, accs_pu)
            except:
                ...
            try:
                print_stats(errs_pu, stds_pu, 25, accs_pu)
            except:
                ...'''
        print('\n')


print_results = True
if print_results:
    '''
    print(Fore.RED + '\n\nMITIGATION PART\n\n')
    print(Fore.BLUE + 'LAMPS\n')
    p = '../deep_learning_mitigation/lamp'
    ns = 10
    pu = ['0,05', '0,1']
    # CHIN
    model = 'l_chin'
    retrieve_statistics(p, model, ns, pu)
    # FREN
    model = 'l_fren'
    retrieve_statistics(p, model, ns, pu)
    # TUR
    model = 'l_tur'
    retrieve_statistics(p, model, ns, pu)
    '''


    print(Fore.BLUE + '\CARPETS STRETCHED\n')
    p = '../deep_learning_mitigation/carpet_stretch'
    ns = 10
    pu = ['0,05', '0,1']
    # CHIN
    model = 'c_ind'
    retrieve_statistics(p, model, ns, pu)
    # FREN
    model = 'c_jap'
    retrieve_statistics(p, model, ns, pu)
    # TUR
    model = 'c_scan'
    retrieve_statistics(p, model, ns, pu)



    # TEST FUNCTIONS ANALYSIS PART
print(Fore.RED + '\n\n\nANALYSIS PART \n', Fore.WHITE)


def retrieve_accs_standard(path, model, ns):
    paths = []
    accs = []
    for j in range(3):
        paths.extend(glob.glob(f'{path}/{model}{j}.csv'))
    sorted(paths)
    rc = ResultsClass()
    for p in paths:
        fc = FileClass(p)
        accs_temp = fc.readcms()
        accs_temp = np.asarray(accs_temp[0:ns], dtype=object)
        tot = rc.return_tot_elements(accs_temp[0])
        accs_temp = rc.calculate_percentage_confusion_matrix(accs_temp, tot)
        temp = []
        for i in range(ns):
            temp.append(accs_temp[i][0][0] + accs_temp[i][1][1])
        accs.append(temp)
    return accs


def print_errors_CIC(p, model, ns):
    print(Fore.WHITE + f'MODEL IS {model}')
    accs = retrieve_accs_standard(p, model, ns)
    CICs = []
    if len(accs) > 0:
        for j in range(ns):  # for each sample
            errs = []
            for k in range(3):
                er = calc_ERR(accs[k][j] / 100)
                errs.append(er)
            c = calc_CIC(errs)
            CICs.append(c)
        CIC, CICstd = retrieve_mean_dev_std(CICs)
        errs = []
        for j in range(3):
            l = np.multiply(accs[j], 0.01)
            acc, std = retrieve_mean_dev_std(l)
            #print(f'Acc: {acc:.1f} on Culture {j}')
            er = calc_ERR(acc)
            errs.append(er)
            print(f'Error is {er*100:.1f}%+-{std*100:.1f}% on Culture {j}')
        if len(errs) >= 3:
            #cic = calc_CIC(errs)
            #print(f'CIC for this model is {cic*100:.1f}%')
            print(f'CIC for the model is {CIC*100:.1f}%+-{CICstd*100:.1f}%\n')

print_results=True
if print_results:    
    '''
    print(Fore.BLUE + '\nLAMPS\n')
    # SVM
    p = '../standard'
    # pu = 0 and LIN
    print(Fore.WHITE + '\nPU = 0')
    print('LSVM')
    pt = p + '/lin'
    model = 'lin_chin'
    print_errors_CIC(pt, model, ns)
    model = 'lin_fren'
    print_errors_CIC(pt, model, ns)
    model = 'lin_tur'
    print_errors_CIC(pt, model, ns)

    # pu = 0 and RBF
    print('GSVM')
    pt = p + '/rbf'
    model = 'rbf_chin'
    print_errors_CIC(pt, model, ns)
    model = 'rbf_fren'
    print_errors_CIC(pt, model, ns)
    model = 'rbf_tur'
    print_errors_CIC(pt, model, ns)
    # pu = 0 and DL
    print(f'pu = 0')
    print('DL')
    pt = '../deep_learning' + '/lamp'
    model = 'l_chin'
    print_errors_CIC(pt, model, ns)
    model = 'l_fren'
    print_errors_CIC(pt, model, ns)
    model = 'l_tur'
    print_errors_CIC(pt, model, ns)
    print('\nPU = 0.1')
    # pu = 0.1 and LIN
    print('LSVM')
    pt = p + '/9010' + '/lin'
    model = 'lin_chin'
    print_errors_CIC(pt, model, ns)
    model = 'lin_fren'
    print_errors_CIC(pt, model, ns)
    model = 'lin_tur'
    print_errors_CIC(pt, model, ns)

    # pu = 0.1 and RBF
    print('GSVM')
    pt = p + '/9010' + '/rbf'
    model = 'rbf_chin'
    print_errors_CIC(pt, model, ns)
    model = 'rbf_fren'
    print_errors_CIC(pt, model, ns)
    model = 'rbf_tur'
    print_errors_CIC(pt, model, ns)

    # pu = 0.1 and DL
    print(f'pu = 0.1')
    print('DL')
    pt = '../deep_learning' + '/9010/lamp/percent0,1'
    model = 'l_chin'
    print_errors_CIC(pt, model, ns)
    model = 'l_fren'
    print_errors_CIC(pt, model, ns)
    model = 'l_tur'
    print_errors_CIC(pt, model, ns)
    print('\nPU = 0.05')
    # pu = 0.05 and LIN
    print('LSVM')
    pt = p + '/50' + '/lin'
    model = 'lin_chin'
    print_errors_CIC(pt, model, ns)
    model = 'lin_fren'
    print_errors_CIC(pt, model, ns)
    model = 'lin_tur'
    print_errors_CIC(pt, model, ns)

    # pu = 0.1 and RBF
    print('GSVM')
    pt = p + '/50' + '/rbf'
    model = 'rbf_chin'
    print_errors_CIC(pt, model, ns)
    model = 'rbf_fren'
    print_errors_CIC(pt, model, ns)
    model = 'rbf_tur'
    print_errors_CIC(pt, model, ns)
    # pu = 0.05 and DL
    print(f'pu = 0.05')
    print('DL')
    pt = '../deep_learning' + '/9010/lamp/percent0,05'
    model = 'l_chin'
    print_errors_CIC(pt, model, ns)
    model = 'l_fren'
    print_errors_CIC(pt, model, ns)
    model = 'l_tur'
    print_errors_CIC(pt, model, ns)
    '''

    print(Fore.BLUE + '\CARPETS STRETCHED\n')
    print(Fore.WHITE + 'DL')
    print('\nPU = 0.0')
    pt = '../deep_learning' + '/carpet_stretch'
    model = 'c_ind'
    print_errors_CIC(pt, model, ns)
    model = 'c_jap'
    print_errors_CIC(pt, model, ns)
    model = 'c_scan'
    print_errors_CIC(pt, model, ns)
    print('\nPU = 0.1')
    pt = '../deep_learning' + '/9010/carpet_stretch/percent0,1'
    model = 'c_ind'
    print_errors_CIC(pt, model, ns)
    model = 'c_jap'
    print_errors_CIC(pt, model, ns)
    model = 'c_scan'
    print_errors_CIC(pt, model, ns)
    print('\nPU = 0.05')
    pt = '../deep_learning' + '/9010/carpet_stretch/percent0,05'
    model = 'c_ind'
    print_errors_CIC(pt, model, ns)
    model = 'c_jap'
    print_errors_CIC(pt, model, ns)
    model = 'c_scan'
    print_errors_CIC(pt, model, ns)


