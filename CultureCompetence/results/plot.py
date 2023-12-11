import sys

sys.path.insert(1, '../')
import results_utils
from matplotlib import pyplot as plt
import colorama
from colorama import Fore
from Utils.utils import FileClass, ResultsClass
import glob
import numpy as np

def plot_cic_err_ref(cics, cicstds, errs, errstds, ref):
    fig, ax = plt.subplots()
    plt.errorbar(errs, cics, cicstds, errstds, linestyle='None', marker='^')
    for tau in ref:
        circle = plt.Circle((0, 0), tau, color='r')
        ax.add_patch(circle)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

def plot_cic_err_gamma(p, model, ns, pu):
    accs = results_utils.retrievs_accs(p, model, ns, pu)
    clerrs, clstds = results_utils.get_errs_stds_for_every_lambda(accs, pu)
    CICs, stdsCIC = results_utils.get_CICs_stds_for_every_lambda(accs, pu)

def plot_cic_err_tau(p, model, ns, pu):
    accs = results_utils.retrievs_accs(p, model, ns, pu)
    clerrs, clstds = results_utils.get_errs_stds_for_every_lambda(accs, pu)
    CICs, stdsCIC = results_utils.get_CICs_stds_for_every_lambda(accs, pu)
    taus = [0.01, 0.05, 0.1, 0.2, 0.5,
            0.8]  # 5%, 10% and 20% of the minimum error
    for i in range(len(pu)):
        print(f'\nREFERRING TO PU={pu[i]}')
        # I want lambda for min errs (for each culture) and min CIC and their values
        accs_pu = accs[i]
        errs_pu, stds_pu = clerrs[i], clstds[i]
        CICs_pu, stds_CIC_pu = CICs[i], stdsCIC[i]
        lambs = []
        if len(errs_pu) > 0:
            # Calculate the minimum CIC in the first tau% of errors
            for tau in taus:
                lamb = results_utils.get_lamb_metric(errs_pu, tau, CICs_pu)
                lambs.append(lamb)
                #results_utils.print_stats(errs_pu, stds_pu, lamb, CICs_pu,
                                          #stds_CIC_pu, tau)
            ERRS = []
            ERRSSTD = []
            CICS = []
            CICSSTD = []
            for lam in lambs:
                ERRS.append(errs_pu[lam])
                ERRSSTD.append(stds_pu[lam])
                CICS.append(CICs_pu[lam])
                CICSSTD.append(stds_CIC_pu[lam])
            plot_cic_err_ref(CICS, CICSSTD, ERRS, ERRSSTD, taus)


print_results = True
if print_results:
    print(Fore.RED + '\n\nMITIGATION PART\n\n')
    print(Fore.BLUE + 'LAMPS\n' + Fore.WHITE)
    p = '../deep_learning_mitigation/lamp'
    ns = 10
    pu = ['0,05', '0,1']
    # CHIN
    model = 'l_chin'
    plot_cic_err_tau(p, model, ns, pu)
    # FREN
    model = 'l_fren'
    plot_cic_err_tau(p, model, ns, pu)
    # TUR
    model = 'l_tur'
    plot_cic_err_tau(p, model, ns, pu)

    print(Fore.BLUE + '\CARPETS STRETCHED\n' + Fore.WHITE)
    p = '../deep_learning_mitigation/carpet_stretch'
    ns = 10
    pu = ['0,05', '0,1']
    # CHIN
    model = 'c_ind'
    plot_cic_err_tau(p, model, ns, pu)
    # FREN
    model = 'c_jap'
    plot_cic_err_tau(p, model, ns, pu)
    # TUR
    model = 'c_scan'
    plot_cic_err_tau(p, model, ns, pu)

    print(Fore.BLUE + '\CARPETS blank\n' + Fore.WHITE)
    p = '../deep_learning_mitigation/carpet_blank'
    ns = 10
    pu = ['0,05', '0,1']
    # CHIN
    model = 'c_ind'
    plot_cic_err_tau(p, model, ns, pu)
    # FREN
    model = 'c_jap'
    plot_cic_err_tau(p, model, ns, pu)
    # TUR
    model = 'c_scan'
    plot_cic_err_tau(p, model, ns, pu)

    # TEST FUNCTIONS ANALYSIS PART
    print(Fore.RED + '\n\n\nANALYSIS PART \n' + Fore.WHITE)


def get_cic_err_pu_standard(p, model, ns):
    print(Fore.WHITE + f'MODEL IS {model}')
    accs = results_utils.retrieve_accs_standard(p, model, ns)
    CICs = []
    if len(accs) > 0:
        for j in range(ns):  # for each sample
            errs = []
            for k in range(3):
                er = results_utils.calc_ERR(accs[k][j] / 100)
                errs.append(er)
            c = results_utils.calc_CIC(errs)
            CICs.append(c)
        CIC, CICstd = results_utils.retrieve_mean_dev_std(CICs)
        errs = []
        errs_std = []
        res = []
        for j in range(3):
            l = np.multiply(accs[j], 0.01)
            acc, std = results_utils.retrieve_mean_dev_std(l)
            #print(f'Acc: {acc:.1f} on Culture {j}')
            er = results_utils.calc_ERR(acc)
            errs_std.append(std)
            errs.append(er)
            print(f'Error is {er*100:.1f}%+-{std*100:.1f}% on Culture {j}')
        if len(errs) >= 3:
            #cic = calc_CIC(errs)
            #print(f'CIC for this model is {cic*100:.1f}%')
            print(f'CIC for the model is {CIC*100:.1f}%+-{CICstd*100:.1f}%\n')
        return errs, errs_std, CIC, CICstd
    


print_results = False
if print_results:
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

    print(Fore.BLUE + '\CARPETS blank\n')
    print(Fore.WHITE + 'DL')
    print('\nPU = 0.0')
    pt = '../deep_learning' + '/carpet_blank'
    model = 'c_ind'
    print_errors_CIC(pt, model, ns)
    model = 'c_jap'
    print_errors_CIC(pt, model, ns)
    model = 'c_scan'
    print_errors_CIC(pt, model, ns)
    print('\nPU = 0.1')
    pt = '../deep_learning' + '/9010/carpet_blank/percent0,1'
    model = 'c_ind'
    print_errors_CIC(pt, model, ns)
    model = 'c_jap'
    print_errors_CIC(pt, model, ns)
    model = 'c_scan'
    print_errors_CIC(pt, model, ns)
    print('\nPU = 0.05')
    pt = '../deep_learning' + '/9010/carpet_blank/percent0,05'
    model = 'c_ind'
    print_errors_CIC(pt, model, ns)
    model = 'c_jap'
    print_errors_CIC(pt, model, ns)
    model = 'c_scan'
    print_errors_CIC(pt, model, ns)

