import pandas as pd
import _pickle as pickle
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from collections import OrderedDict

import Parameters as par


def get_original_params():
    params = OrderedDict()
    params['beta_CHMA'] = 7.8e-4
    params['beta_CHNA'] = 4.8e-2
    params['theta_ACH'] = 1e-10
    params['beta_MANDA'] = 9e1
    params['lamb_ITMNDN'] = 1.e-6
    params['alpha_ITMNDN'] = 1e3
    params['Pmax_APE'] = 0.002
    params['Pmin_APE'] = 0.0002
    params['rdistress'] = 3e6
    params['w_gauss_min'] = 8e7
    params['rinduce_peak'] = 21.0
    params['rinduce'] = 0.05
    params['r_AP'] = 0.08
    params['r_ITM'] = 0.5
    params['r_ITMpeak'] = 5 * 10 ** 12
    params['r_NDN'] = 0.008
    params['lamb_MANDN'] = 5e-6
    params['lamb_MANDA'] = 3e-5
    params['mu_NDA'] = 2.5e1
    params['Keq_CH'] = 2e4
    params['r_Nhomeo'] = 1e-4
    params['Pmax_NR'] = 6e-3
    return params


def get_x_boundaries():
    boundaries = OrderedDict()
    boundaries['beta_CHMA'] = (0, 0.5)
    boundaries['beta_CHNA'] = (0, 0.5)
    boundaries['theta_ACH'] = (0, 0.005)
    boundaries['beta_MANDA'] = (0, 1500)
    boundaries['lamb_ITMNDN'] = (0.01, 0.07)
    boundaries['alpha_ITMNDN'] = (0, 5500)
    boundaries['Pmax_APE'] = (0, 1)
    boundaries['Pmin_APE'] = (0, 0.1)
    boundaries['rdistress'] = (0, 1e9)
    boundaries['w_gauss_min'] = (0, 1e9)
    boundaries['rinduce_peak'] = (0, 30)
    boundaries['rinduce'] = (0, 0.5)
    boundaries['r_AP'] = (0, 0.5)
    boundaries['r_ITM'] = (0.2, 0.5)
    boundaries['r_ITMpeak'] = (0, 10 ** 14)
    boundaries['r_NDN'] = (0, 0.02)
    boundaries['lamb_MANDN'] = (0, 0.005)
    boundaries['lamb_MANDA'] = (0, 0.5)
    boundaries['mu_NDA'] = (1100, 1400)
    boundaries['Keq_CH'] = (0, 1e6)
    boundaries['r_Nhomeo'] = (0, 0.05)
    boundaries['Pmax_NR'] = (0, 0.3)
    return boundaries


def get_y_boundaries():
    boundaries = OrderedDict()
    boundaries['beta_CHMA'] = (-3000, 5000)
    boundaries['beta_CHNA'] = (-20, 25)
    boundaries['theta_ACH'] = (0.3, 1)
    boundaries['beta_MANDA'] = (0.3, 1)
    boundaries['lamb_ITMNDN'] = (0.3, 1)
    boundaries['alpha_ITMNDN'] = (-30, 40)
    boundaries['Pmax_APE'] = (-1, 2)
    boundaries['Pmin_APE'] = (0.3, 1)
    boundaries['rdistress'] = (0.3, 1)
    boundaries['w_gauss_min'] = (0.3, 1)
    boundaries['rinduce_peak'] = (0.3, 1)
    boundaries['rinduce'] = (0, 1)
    boundaries['r_AP'] = (0.3, 1)
    boundaries['r_ITM'] = (-5, 10)
    boundaries['r_ITMpeak'] = (-0.5, 3)
    boundaries['r_NDN'] = (0.3, 1)
    boundaries['lamb_MANDN'] = (0.3, 1)
    boundaries['lamb_MANDA'] = (0.3, 1)
    boundaries['mu_NDA'] = (0.3, 1)
    boundaries['Keq_CH'] = (0.3, 1)
    boundaries['r_Nhomeo'] = (-0.5, 20)
    boundaries['Pmax_NR'] = (-1, 1)
    return boundaries

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


def read_pickle(project_dir, pickle_file):
    pickle_in = open(project_dir + '/' + pickle_file, 'rb')
    pickle_out = pickle.load(pickle_in)
    return pickle_out


def do_sweep(project_dir, param_names):
    df_dic = {}
    cin_dic = {}
    cin_list =[]
    orig_params = get_original_params()
    for param_name in param_names:
        df_param = pd.DataFrame()
        df_param['cost'] = 0.014987433951042167
        df_param['trial'] = -1
        df_param['value'] = orig_params[param_name]
        for file in os.listdir(project_dir):
            if file.endswith(".pkl"):
                param_profile = '_'.join(file.split('_')[1:-1])
                if param_name == param_profile:
                    df = read_pickle(project_dir, file)
                    df = df[(df.T != 0).any()]
                    df_segment = pd.DataFrame()
                    df_segment['trial'] = df['trial']
                    df_segment['value'] = df['value']
                    df_segment['cost'] = df['cost']
                    df_param = df_param.append(df_segment)
        LP = [np.exp(-0.5*i) for i in df_param['cost'].tolist()]
        cin = st.t.interval(0.95, len(LP) - 1, loc=np.exp(-0.5*np.median([i for i in df_param['cost'].tolist()
                                                                          if i != float("inf")])), scale=st.sem(LP))
        cin_list.append(cin[-1])
        df_param = df_param.groupby('value', as_index=False)['cost'].min()
        df_dic[param_name] = df_param
        cin_dic[param_name] = cin
    return df_dic, cin_dic, np.median(cin_list)


def plot(project_dir, df_dic, param_names, param_labels, cin_dic, cin_mean):
    # x_boundaries = get_x_boundaries()
    # y_boundaries = get_y_boundaries()
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 2.5})
    sns.set(style="white")

    n = 0
    fw = 1
    ls = 9
    fs = 9
    ms = 15

    lw = 2
    fig = plt.figure(figsize=(10, 10))
    columns = 5
    rows = 5
    count_params = 0
    for i in range(len(param_names)):
        param_name = param_names[i]
        if param_name != 'b_CHMA':
            fig.add_subplot(rows, columns, i+1)
            df_param = df_dic[param_name]
            filtered_cost = df_param['cost'].tolist()

            if count_params % columns == 0:
                plt.ylabel('$-2\log$(Profile Likelihood)', fontsize=fs)

            plt.plot(df_param['value'], filtered_cost, ls='-', linewidth=lw, color='black')
            plt.plot(df_param['value'][list(filtered_cost).index(min(filtered_cost))], min(filtered_cost),
                      markersize=ms, marker='*', linewidth=fw, color='red')
            plt.plot(df_param['value'], [-2*np.log(cin_mean)] * len(df_param['value']), ls='--', linewidth=lw,
                     color='yellow')
            plt.xlabel(param_labels[param_name], fontsize=fs)

            # plt.ylim(0.7, 1.1)
            # plt.xlim((x_boundaries[param_name]))
            plt.tick_params(labelsize=ls)
            count_params += 1
    plt.tight_layout()
    plt.savefig(project_dir + '/identifiability/PIA.png', dpi=500, bbox_inches='tight')


def get_specific(project_dir):
    df = read_pickle(project_dir, '0_Pmin_APE_optimized.pkl')
    df_small = df.nsmallest(3, 'cost')
    for i, row in df_small.iterrows():
        print(row['cost'])
        print('beta_CHMA =', row['beta_CHMA'])
        print('beta_CHNA =', row['beta_CHNA'])
        print('theta_ACH =', row['theta_ACH'])
        print('beta_MANDA =', row['beta_MANDA'])
        print('lamb_ITMNDN =', row['lamb_ITMNDN'])
        print('alpha_ITMNDN =', row['alpha_ITMNDN'])
        print('Pmax_APE =', row['Pmax_APE'])
        print('Pmin_APE =', row['Pmin_APE'])
        print('rdistress =', row['rdistress'])
        print('w_gauss_min =', row['w_gauss_min'])
        print('rinduce_peak =', row['rinduce_peak'])
        print('rinduce =' , row['rinduce'])
        print('r_AP =', row['r_AP'])
        print('r_ITM =', row['r_ITM'])
        print('r_ITMpeak =', row['r_ITMpeak'])
        print('r_NDN =', row['r_NDN'])
        print('lamb_MANDN =', row['lamb_MANDN'])
        print('lamb_MANDA =', row['lamb_MANDA'])
        print('mu_NDA =', row['mu_NDA'])
        print('Keq_CH =', row['Keq_CH'])
        print('r_Nhomeo =', row['r_Nhomeo'])
        print('Pmax_NR =', row['Pmax_NR'])
        print('_______________________')
        print('           ')


if __name__ == '__main__':
    project_dir = 'C:/Users/Alva/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                         'AP Simulator Model/Frontiers in Immunology/Special Issue/Revisions/results/Likelihood/peaks'
    param_names = par.get_boundaries().keys()
    param_names = ['beta_CHMA','Pmin_APE', 'rdistress', 'w_gauss_min', 'r_AP']
    df_dic, cin_dic, cin_mean = do_sweep(project_dir , param_names)
    param_labels = get_label_dictionary()
    plot(project_dir, df_dic, list(param_names), param_labels, cin_dic, cin_mean)
