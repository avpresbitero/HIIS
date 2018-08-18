import _pickle as pickle
import pandas as pd
import seaborn as sns
import ast
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

from tqdm import tqdm

from SALib.sample import saltelli, fast_sampler
from SALib.analyze import sobol, fast

import numpy as np

import Parameters as par
import APSim_Model as innate


def get_label_dictionary():
    params = OrderedDict()
    params['beta_CHMA'] = r'$\beta_{MA|ITM}$'
    params['beta_CHNA'] = r'$\beta_{NA|ITM}$'
    params['theta_ACH'] = r'$\theta_{ACH}$'
    params['beta_MANDA'] = r'$\beta_{NDA|MA}$'
    params['lamb_ITMNDN'] = r'$\lambda_{ITMNDN}$'
    params['alpha_ITMNDN'] = r'$\alpha_{ITMNDN}$'
    params['Pmax_APE'] = r'$P_{APE}^{max}$'
    params['Pmin_APE'] = r'$P_{APE}^{min}$'
    params['rdistress'] = r'$r_{distress}$'
    params['w_gauss_min'] = r'$w$'
    params['rinduce_peak'] = r'$r_{inducepeak}$'
    params['rinduce'] = r'$r_{induce}$'
    params['r_AP'] = r'$r_{AP}$'
    params['r_ITM'] = r'$r_{ITM}$'
    params['r_ITMpeak'] = r'$r_{ITMpeak}$'
    params['r_NDN'] = '$r_{NDN}$'
    params['lamb_MANDN'] = r'$\lambda_{NDN|MA}$'
    params['lamb_MANDA'] = r'$\lambda_{NDA|MA}$'
    params['mu_NDA'] = r'$\mu_{NDA}$'
    params['Keq_CH'] = r'$K_{eqCH}$'
    params['r_Nhomeo'] = r'$r_{homeo}$'
    params['Pmax_NR'] = r'$P_{N_R}^{max}$'
    return params

def get_problem():
    problem = {
        'num_vars': 22,
        'names': list(par.get_boundaries().keys()),
        'bounds': list(par.get_boundaries().values())
    }
    return problem


def generate_samples(problem, params, project_dir, method):
    """
    The Saltelli sampler generated 8000 samples. The Saltelli sampler generates N*(2D+2) samples, where in this
    example N is 1000 (the argument we supplied) and D is 3 (the number of model inputs).
    """

    if method == 'Saltelli':
        param_values = saltelli.sample(problem, params['samples'])
    elif method == 'FAST':
        param_values = fast_sampler.sample(problem, params['samples'])

    count = 0

    for i, X in enumerate(param_values):
        count += 1
        p, w, pred_fle = par.get_params(innate, X)
        _numpoints = 100
        t = [par._stoptime * float(i) / (_numpoints - 1) for i in range(_numpoints)]
        w0 = innate.get_init(w, params)
        t, wsol = innate.solve(p, w0, t, params)
        APE_blood = []
        CH = []
        ACH = []
        for t1, w1 in zip(t, wsol):
            APE_blood.append(w1[1])
            CH.append(w1[10])
            ACH.append(w1[13])
        # Y_list.append(APE_blood)
        write_file(project_dir, method, APE_blood, 'AP')
        write_file(project_dir, method, CH, 'CH')
        write_file(project_dir, method, ACH, 'ACH')
        print(count, ' of ', len(param_values))


def write_file(project_dir, method, blood_parameter, param_name):
    with open(project_dir + '/' + method + '_' + param_name + '.txt', 'a') as file:
        file.write(str(blood_parameter) + '\n')
    file.close()


def read_file(project_dir, param_name, method):
    with open(project_dir + '/' + method + '_' + param_name + '.txt') as file:
        content = file.readlines()
    content = [ast.literal_eval(x.replace("\n", "")) for x in content]
    file.close()
    return content


def pickle_it(unpickled, project_dir, pickle_file):
    print("Pickling it...")
    pickle_out = open(project_dir + '/' + pickle_file, "wb")
    pickle.dump(unpickled, pickle_out)
    pickle_out.close()


def read_pickle(project_dir, pickle_file):
    pickle_in = open(project_dir + '/' + pickle_file, 'rb')
    pickle_out = pickle.load(pickle_in)
    return pickle_out


def perform_analysis(problem, Y_list, method):
    S1_dic = OrderedDict()
    S1_dic['beta_CHMA'] = []
    S1_dic['beta_CHNA'] = []
    S1_dic['theta_ACH'] = []
    S1_dic['beta_MANDA'] = []
    S1_dic['lamb_ITMNDN'] = []
    S1_dic['alpha_ITMNDN'] = []
    S1_dic['Pmax_APE'] = []
    S1_dic['Pmin_APE'] = []
    S1_dic['rdistress'] = []
    S1_dic['w_gauss_min'] = []
    S1_dic['rinduce_peak'] = []
    S1_dic['rinduce'] = []
    S1_dic['r_AP'] = []
    S1_dic['r_ITM'] = []
    S1_dic['r_ITMpeak'] = []
    S1_dic['r_NDN'] = []
    S1_dic['lamb_MANDN'] = []
    S1_dic['lamb_MANDA'] = []
    S1_dic['mu_NDA'] = []
    S1_dic['Keq_CH'] = []
    S1_dic['r_Nhomeo'] = []
    S1_dic['Pmax_NR'] = []
    Y_list = np.array(Y_list)
    print (Y_list.shape)
    Y_list_trans = Y_list.transpose()

    total = len(Y_list_trans)
    count = 0

    for Y in Y_list_trans:
        count += 1
        if method == 'FAST':
            Si = fast.analyze(problem, Y, print_to_console=True)
        elif method == 'Saltelli':
            Si = sobol.analyze(problem, Y)

        beta_CHMA, beta_CHNA, theta_ACH, beta_MANDA, lamb_ITMNDN, alpha_ITMNDN, Pmax_APE, Pmin_APE, rdistress, \
        w_gauss_min, rinduce_peak, rinduce, r_AP, r_ITM, r_ITMpeak, r_NDN, lamb_MANDN, lamb_MANDA, mu_NDA, \
        Keq_CH, r_Nhomeo, Pmax_NR = Si['S1'] #Si['ST']# = Si['S1']

        S1_dic['beta_CHMA'].append(beta_CHMA)
        S1_dic['beta_CHNA'].append(beta_CHNA)
        S1_dic['theta_ACH'].append(theta_ACH)
        S1_dic['beta_MANDA'].append(beta_MANDA)
        S1_dic['w_gauss_min'].append(w_gauss_min)
        S1_dic['lamb_ITMNDN'].append(lamb_ITMNDN)
        S1_dic['alpha_ITMNDN'].append(alpha_ITMNDN)
        S1_dic['Pmax_APE'].append(Pmax_APE)
        S1_dic['Pmin_APE'].append(Pmin_APE)
        S1_dic['rdistress'].append(rdistress)
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
        if i % 5 == 0 or i == 0:
            pick_result.append(result[i])
    print (pick_result)
    return pick_result


def plot(df, project_dir, method, param_name, label_dic, title_dic):
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 2.5})

    lw = 3
    ls = 20
    fs = 25
    ms = 20
    lfs = 25
    ts = 30
    markers = ['v', '<', '>', '^', '*', 'v', '<', '>', '^', '*', 'o', '.', 'h', 'p', 'H', ".", ",",
               "o", "v", "^", "8", "s", "p", "P", "*", "h", "H",
               "+", "x", "X", "D", "d", "|", "_"]
    line_styles = ['-', '--', '-.', ':']
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
    count = 0
    count_important = 0
    sns.set(style="white")
    df.fillna(0, inplace=True)
    increment = 0
    for column in df:
        marker = markers[count]
        # color = colors[count % len(colors)]
        result = (df[column])
        time = [(i/len(result))*36. for i in range(len(result))]
        if max(result) >= 0.05:
            line_style = line_styles[count_important % len(line_styles)]
            color = '#e41a1c'
            plt.plot(time, result, label=label_dic[column], ls=line_style, linewidth=5, color=color, alpha=1-increment)
            increment += 0.1
            count_important += 1
        elif column == 'rinduce' or column == 'rinduce_peak':
            line_style = line_styles[count_important % len(line_styles)]
            color = '#377eb8'
            plt.plot(time, result, label=label_dic[column], ls=line_style, linewidth=5, color=color, alpha=1-increment)
            increment += 0.1
            count_important += 1
        # else:
        #     color = '#999999'
        #     plt.plot(time, result, ls='-', linewidth=lw, color=color, alpha=0.4, label='')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': ls}, frameon=True)
        plt.plot(time, [0.05]*len(time), ls='-', linewidth=lw, color='black')
        plt.xlabel("Hours After Operation", fontsize=fs)
        plt.ylabel('Sensitivity Index', fontsize=fs)
        plt.title(title_dic[param_name], fontsize=fs)
        plt.xlim((0, 5))
        # plt.xlim((0, 36))
        plt.tick_params(labelsize=ls)
        # plt.hold(True)
        plt.savefig(project_dir + '/' + method + '_' + param_name + '_SA.png', dpi=500, bbox_inches='tight')
        count += 1
    # plt.show()


def do_analysis(params, project_dir, pickle_file, method):
    for param_name in params['param_names']:
        problem = read_pickle(project_dir, pickle_file)
        Y_list = read_file(project_dir, param_name, method)
        df = perform_analysis(problem, Y_list, method)
        df_file = 'df_' + param_name + '_' + method + '.pickle'
        pickle_it(df, project_dir, df_file)


def main():
    project_dir = 'C:/Users/Alva/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                  'AP Simulator Model/Frontiers in Immunology/Special Issue/Revisions/results/sensitivity/change'
    params = {'h': 'h4',
              'restrict': False,
              'case': 6,
              'method': ['FAST'],
              # 'method': ['Saltelli'],
              'samples': 5000,
              'param_names': ['AP', 'CH', 'ACH']
              # 'param_names': ['AP']
              # 'param_names': ['CH']
              }

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    label_dic = get_label_dictionary()
    title_dic = {'CH': 'Pro-inflammatory Cytokines',
                 'ACH': 'Anti-inflammatory Cytokines',
                 'AP': 'Alkaline Phosphatase'
                 }
    for method in params['method']:
        pickle_file = method + '_Problem.pickle'
        problem = get_problem()
        pickle_it(problem, project_dir, pickle_file)
        generate_samples(problem, params, project_dir, method)

        # do_analysis(params, project_dir, pickle_file, method)

        # param_name = 'AP'
        # df_file = 'df_' + param_name + '_' + method + '.pickle'
        # df = read_pickle(project_dir, df_file)
        # plot(df, project_dir, method, param_name, label_dic, title_dic)

if __name__ == '__main__':
    main()