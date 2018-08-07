import pickle
import pandas as pd
import seaborn as sns

from SALib.sample import saltelli, fast_sampler
from SALib.analyze import sobol, fast

import numpy as np

import Parameters as par
import APSim_Model as innate


def generate_samples(method):
    """
    The Saltelli sampler generated 8000 samples. The Saltelli sampler generates N*(2D+2) samples, where in this
    example N is 1000 (the argument we supplied) and D is 3 (the number of model inputs).
    """

    problem = {
        'num_vars': 22,
        'names': ['beta_CHMA', 'beta_CHNA', 'theta_ACH', 'beta_MANDA', 'gauss_min', 'lamb_ITMNDN', 'alpha_ITMNDN',\
                  'Pmax_APE', 'Pmin_APE', 'rflush', 'rinduce_peak', 'rinduce', 'r_ITM', 'r_ITMpeak', 'r_NDN', \
                  'r_AP', 'lamb_MANDN', 'lamb_MANDA', 'mu_NDA', 'Keq_CH', 'r_Nhomeo', 'Pmax_NR'],
        'bounds': [list(bound) for bound in prms.get_bounds()]
    }

    if method == 'sobol':
        param_values = saltelli.sample(problem, 10)
    elif method == 'fast':
        param_values = fast_sampler.sample(problem, 70)

    total = len(param_values)
    Y_list = []
    count = 0
    for i, X in enumerate(param_values):
        count += 1
        p, w, pred_fle = par.get_params(innate, X)
        t = par.get_t(innate)
        innate.case, innate.h = 6, 'h2'
        w0 = innate.get_init(w)
        t, wsol = innate.solve(p, w0, t)
        APE_blood = []
        for t1, w1 in zip(t, wsol):
            APE_blood.append(w1[1])
        Y_list.append(APE_blood)
        print(total, count, len(Y_list))
    return problem, Y_list


def pickle_it(samples, pickle_file):
    print("Pickling it...")
    pickle_out = open(pickle_file, "wb")
    pickle.dump(samples, pickle_out)
    pickle_out.close()


def read_pickle(pickle_file):
    pickle_in = open(pickle_file, 'rb')
    pickle_out = pickle.load(pickle_in)
    return pickle_out


def perform_analysis(problem, Y_list, method):
    S1_dic = {'beta_CHMA': [],
              'beta_CHNA': [],
              'theta_ACH': [],
              'beta_MANDA': [],
              'gauss_min': [],
              'lamb_ITMNDN': [],
              'alpha_ITMNDN': [],
              'Pmax_APE': [],
              'Pmin_APE': [],
              'rflush': [],
              'rinduce_peak': [],
              'rinduce': [],
              'r_ITM': [],
              'r_ITMpeak': [],
              'r_NDN': [],
              'r_AP': [],
              'lamb_MANDN': [],
              'lamb_MANDA': [],
              'mu_NDA': [],
              'Keq_CH': [],
              'r_Nhomeo': [],
              'Pmax_NR': []}
    Y_list = np.array(Y_list)
    Y_list_trans = Y_list.transpose()
    total = len(Y_list_trans)
    count = 0
    for Y in Y_list_trans:
        count += 1
        if method == 'fast':
            Si = fast.analyze(problem, Y, print_to_console=False)
        elif method == 'sobol':
            Si = sobol.analyze(problem, Y)
        beta_CHMA, beta_CHNA, theta_ACH, beta_MANDA, gauss_min, lamb_ITMNDN, alpha_ITMNDN, Pmax_APE, Pmin_APE, rflush, \
        rinduce_peak, rinduce, r_ITM, r_ITMpeak, r_NDN, r_AP, lamb_MANDN, lamb_MANDA, mu_NDA, Keq_CH, r_Nhomeo, \
        Pmax_NR = Si['S1']

        S1_dic['beta_CHMA'].append(beta_CHMA)
        S1_dic['beta_CHNA'].append(beta_CHNA)
        S1_dic['theta_ACH'].append(theta_ACH)
        S1_dic['beta_MANDA'].append(beta_MANDA)
        S1_dic['gauss_min'].append(gauss_min)
        S1_dic['lamb_ITMNDN'].append(lamb_ITMNDN)
        S1_dic['alpha_ITMNDN'].append(alpha_ITMNDN)
        S1_dic['Pmax_APE'].append(Pmax_APE)
        S1_dic['Pmin_APE'].append(Pmin_APE)
        S1_dic['rflush'].append(rflush)
        S1_dic['rinduce_peak'].append(rinduce_peak)
        S1_dic['rinduce'].append(rinduce)
        S1_dic['r_ITM'].append(r_ITM)
        S1_dic['r_ITMpeak'].append(r_ITMpeak)
        S1_dic['r_NDN'].append(r_NDN)
        S1_dic['r_AP'].append(r_AP)
        S1_dic['lamb_MANDN'].append(lamb_MANDN)
        S1_dic['lamb_MANDA'].append(lamb_MANDA)
        S1_dic['mu_NDA'].append(mu_NDA)
        S1_dic['Keq_CH'].append(Keq_CH)
        S1_dic['r_Nhomeo'].append(r_Nhomeo)
        S1_dic['Pmax_NR'].append(Pmax_NR)
        print(total, count)
    return pd.DataFrame(S1_dic)


def pick(result):
    pick_result = []
    for i in range(len(result)):
        if i % 200 == 0:
            pick_result.append(result[i])
    return pick_result


def plot(df, method):
    markers = ['v', '<', '>', '^', '*', 'v', '<', '>', '^', '*', 'o', '.', 'h', 'p', 'H', ".", ",",
               "o", "v", "^", "8", "s", "p", "P", "*", "h", "H",
               "+", "x", "X", "D", "d", "|", "_"]
    count = 0
    sns.set(style="white")
    for column in df:
        marker = markers[count]
        result = pick(df[column])
        time = [(i/len(result))*36. for i in range(len(result))]
        sns.plt.plot(time, result, label=column, marker=marker, ms=15, linewidth=1.5, alpha=.5)
        sns.plt.xlabel("Hours After Operation", fontsize=20)
        sns.plt.ylabel("Si(" + method + ")", fontsize=20)
        sns.plt.legend(loc='best', prop={'size': 13}, frameon=True)
        sns.plt.savefig('result/plots/sensitivity/' + method + '_SA.png', dpi=300)
        count += 1
    sns.plt.show()


def main():
    method = 'fast'
    if method == 'fast':
        pickle_file = 'result/sensitivity/FAST_Samples.pickle'
    elif method == 'sobol':
        pickle_file = 'result/sensitivity/Saltelli_Samples.pickle'
    # problem, Y_list = generate_samples(method)
    # pickle_it((problem, Y_list), pickle_file)

    problem, Y_list = read_pickle(pickle_file)
    df = perform_analysis(problem, Y_list, method)
    plot(df, method)

if __name__ == '__main__':
    main()