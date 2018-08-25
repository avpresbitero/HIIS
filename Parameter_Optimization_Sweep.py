import pandas as pd
import _pickle as pickle
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from collections import OrderedDict
from scipy.signal import savgol_filter

import Parameters as par


def get_x_boundaries():
    boundaries = OrderedDict()
    boundaries['beta_CHMA'] = (0.05, 0.3)
    boundaries['beta_CHNA'] = (0, 0.5)
    boundaries['theta_ACH'] = (0, 0.3)
    boundaries['beta_MANDA'] = (2500, 7500)
    boundaries['lamb_ITMNDN'] = (0.25, 0.5)
    boundaries['alpha_ITMNDN'] = (500, 4500)
    boundaries['Pmax_APE'] = (0, 0.25)
    boundaries['Pmin_APE'] = (0, 0.025)
    boundaries['rdistress'] = (0, 0.5e9)
    boundaries['w_gauss_min'] = (0, 0.5e9)
    boundaries['rinduce_peak'] = (10, 30)
    boundaries['rinduce'] = (0.1, 0.5)
    boundaries['r_AP'] = (0, 0.25)
    boundaries['r_ITM'] = (0, 0.75)
    boundaries['r_ITMpeak'] = (0, 10 ** 14)
    boundaries['r_NDN'] = (0, 0.5)
    boundaries['lamb_MANDN'] = (0, 0.5)
    boundaries['lamb_MANDA'] = (0, 0.5)
    boundaries['mu_NDA'] = (0, 5000)
    boundaries['Keq_CH'] = (0, 0.5e7)
    boundaries['r_Nhomeo'] = (0, 0.5)
    boundaries['Pmax_NR'] = (0, 0.25)
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


def do_sweep(project_dir):
    df_final = pd.DataFrame()
    for file in os.listdir(project_dir):
        if file.endswith(".pkl"):
            trial = int(file.split('.')[0])
            df = read_pickle(project_dir, file)
            if trial == 0:
                df_final = df
            else:
                df_final = df_final.append(df)
    return df_final


def get_specific(df_dic):
    df_small = df_dic.nsmallest(5, 'cost')
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
                         'AP Simulator Model/Frontiers in Immunology/Special Issue/Revisions/results/optimization/peaks'
    df_dic = do_sweep(project_dir)
    get_specific(df_dic)