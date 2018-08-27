import numpy as np
import pandas as pd
import os

import Parameters as par
import APPIREDII_Blood_Params_Parser as dp
import APPIREDII_Cytokines_Parser as cyto
import APSim_Model as innate

from timeit import default_timer as timer
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import OrderedDict


def init_global():
    global RMSE_CH
    global RMSE_ACH
    global RMSE_AP
    global RMSE_peaks


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


def get_files():
    files = {
        'cytokine_file': par.cyto_fle,
        'sample_file': par.sample_fle,
        'treatment_file': par.trt_fle,
        'ap_german_file': par.german_hospital_file,
        'ap_dutch_file': par.dutch_hospital_file
    }
    return files


def get_header():
    header = ['N_R', 'AP_Eblood', 'AP_Etissue', 'AP_Eliver', 'AP_Sblood', 'AP_Stissue', 'ITMblood', 'ITMtissue', 'M_R',
              'M_A', 'CH', 'N_A', 'ND_A', 'ACH', 'ND_N']
    header = [name.lower() for name in header]
    return header


def get_boundaries_to_pass(boundaries, param_profile):
    boundaries_to_pass = OrderedDict()
    for key, value in boundaries.items():
        if key != param_profile:
            boundaries_to_pass[key] = value
    return boundaries_to_pass


def get_cyto_data(files):
    df_cyto, cyto_dic = cyto.get_median(ct_fle=files['cytokine_file'],
                                        smpl_fle=files['sample_file'],
                                        trt_fle=files['treatment_file'],
                                        colhd='time')
    return df_cyto


def get_ap_data(files):
    df_AP, df_AP_median = dp.get_data(german_hospital_file=files['ap_german_file'],
                                      dutch_hospital_file=files['ap_dutch_file'],
                                      treatment_file=files['treatment_file'],
                                      parameter_name='AlkFosf',
                                      restrict=False,
                                      treatment=None,
                                      patients=None,
                                      do_binning=True)
    df_AP_median['treatment'] = df_AP_median['treatment'].str.lower()
    df_AP_median = set_weights(df_AP_median, 'ap')
    return df_AP_median


def get_from_normal_dist(mu, max_boundary, counter):
    # np.random.seed(int(counter))
    mu_norm = mu / float(max_boundary)
    s_norm = np.random.normal(mu_norm, mu_norm / 3., 1000)
    s = np.random.choice(s_norm)
    if s == 0:
        get_from_normal_dist(mu, counter, max_boundary)
    return s * max_boundary


def get_value_uniform_dist(boundary, counter):
    # np.random.seed(int(counter))
    uniform_list = np.linspace(boundary[0], boundary[1], 100)
    return np.random.choice(uniform_list)


def get_initial_params(params, trial):
    boundaries = par.get_boundaries()
    original_params = get_original_params()
    initial_params = get_original_params()
    for key, boundary in boundaries.items():
        if params['distribution'] == 'normal':
            initial_params[key] = get_from_normal_dist(original_params[key], boundaries[key][-1], trial)
        elif params['distribution'] == 'uniform':
            initial_params[key] = get_value_uniform_dist(boundary, trial)
    return initial_params


def get_p0_to_pass(param_profile, p0_complete):
    params_to_pass = OrderedDict()
    for key, value in p0_complete.items():
        if key != param_profile:
            params_to_pass[key] = value
    return params_to_pass


def get_restricted_data(df_data, blood_param, params):
    """
    Restricts the data to 36 hours, since the model is only tested for 36 hours only.
    :param df_data: dataframe of data for cytokines and AP
    :param blood_param: either 'Cytokine' or 'AP'
    :return: df_restrict
            - if 'Cytokine', this dataframe contains medians for all four Cytokines corresponding to a specific
              treatment (Placebo or bIAP)
            - if 'AP', this dataframe contains median corresponding to a specific treatment (Placebo or bIAP)
              time_restrict
            - list of time (in days) that corresponds to available data
    :param treatment_type: either 'Placebo' or 'bIAP'
    """

    df_restrict = df_data.loc[(df_data['time'] >= 0) & (df_data['time'] <= 36.0)]
    df_restrict = df_restrict[df_restrict['treatment'] == params['treatment_type']]
    if blood_param == 'cytokine':
        # sample cytokine to retrieve time list
        df_restrict_cyto = df_restrict[df_restrict['cytokine'] == 'il10']
        time_restrict = df_restrict_cyto[df_restrict_cyto['treatment'] == params['treatment_type']]['time']
    elif blood_param == 'ap':
        time_restrict = df_restrict[df_restrict['treatment'] == params['treatment_type']]['time']
    time_restrict = set_to_mins(time_restrict)
    df_restrict.loc[:, 'time'] *= 60
    return df_restrict, time_restrict


def get_concentrations(df_data, p, w0, header, blood_parameter, params):
    # print("Getting concentrations for...", blood_parameter)
    df_data_rstrct, time_rstrct = get_restricted_data(df_data, blood_parameter, params)
    time, model = innate.solve(p, w0, time_rstrct, params)
    df_model = pd.DataFrame(model)
    keys = df_model.columns.values
    dictionary = dict(zip(keys, header))
    df_model = df_model.rename(columns=dictionary)
    df_model['time'] = time
    df_data_rstrct = set_weights(df_data_rstrct, blood_parameter)
    return df_model, df_data_rstrct


def get_peaks(p, w0, header, time):
    global RMSE_peaks
    params = {}
    params['h'] = 'h4'
    params['restrict'] = False
    params['case'] = 5
    time, model = innate.solve(p, w0, time, params)
    df_pred = pd.DataFrame(model)
    keys = df_pred.columns.values
    dictionary = dict(zip(keys, header))
    df_pred = df_pred.rename(columns=dictionary)
    placebo_n_a = df_pred['n_a']/max(df_pred['n_a'])
    placebo_nd_n = df_pred['nd_n']/max(df_pred['nd_n'])
    placebo_nd_a = df_pred['nd_a']/max(df_pred['nd_a'])

    params = {}
    params['h'] = 'h4'
    params['restrict'] = False
    params['case'] = 6
    time, model = innate.solve(p, w0, time, params)
    df_pred = pd.DataFrame(model)
    keys = df_pred.columns.values
    dictionary = dict(zip(keys, header))
    df_pred = df_pred.rename(columns=dictionary)
    supplemented_n_a = df_pred['n_a']/max(df_pred['n_a'])
    supplemented_nd_n = df_pred['nd_n']/max(df_pred['nd_n'])
    supplemented_nd_a = df_pred['nd_a']/max(df_pred['nd_a'])

    if max(placebo_nd_n) > max(supplemented_nd_n):
        nd_n_error = 0
    if max(placebo_nd_n) <= max(supplemented_nd_n):
        nd_n_error = abs(max(placebo_nd_n) - max(supplemented_nd_n))
    if max(placebo_nd_a) < max(supplemented_nd_a):
        nd_a_error = 0
    if max(placebo_nd_a) >= max(supplemented_nd_a):
        nd_a_error = abs(max(placebo_nd_a) - max(supplemented_nd_a))
    if max(placebo_n_a) < max(supplemented_n_a):
        n_a_error = abs(max(placebo_n_a) - max(supplemented_n_a))
    if max(placebo_n_a) >= max(supplemented_n_a):
        n_a_error = 0

    try:
        sum_error = np.sum([nd_n_error, nd_a_error, n_a_error])
    except UnboundLocalError as u_error:
        print(u_error)
        sum_error = 1*10**10

    RMSE_peaks = sum_error
    return sum_error


def get_rmse(params, df_cyto_data, df_ap_data, df_cyto_model, df_ap_model):
    # print("Computing RMSEs...")
    # note: Pro-inflammatory Cytokine = IL6, Anti-inflammatory Cytokine = IL10
    global RMSE_CH
    global RMSE_ACH
    global RMSE_AP
    global RMSE_peaks

    df_ch_data = df_cyto_data[df_cyto_data['cytokine'] == 'il6']
    df_ach_data = df_cyto_data[df_cyto_data['cytokine'] == 'il10']

    df_ch_model = df_cyto_model['ch']
    df_ach_model = df_cyto_model['ach']
    # df_ch_model = cyto.reverse_unit(df_cyto_model['ch'].tolist(), 'il6')
    # df_ach_model = cyto.reverse_unit(df_cyto_model['ach'].tolist(), 'il10')
    df_ap_model = np.add(df_ap_model['ap_eblood'], df_ap_model['ap_sblood'])
    df_ap_model_converted = innate.reverse_AP(df_ap_model, 'endo', 'blood')

    df_ch_data_median = df_ch_data['median']
    df_ach_data_median = df_ach_data['median']
    # df_ch_data_median = cyto.reverse_unit(df_ch_data['median'].tolist(), 'il6')
    # df_ach_data_median = cyto.reverse_unit(df_ach_data['median'].tolist(), 'il10')
    df_ap_data_median = df_ap_data['median']

    # normalize
    if max(df_ch_data_median) != 0: df_ch_data_median = df_ch_data_median/max(df_ch_data_median)
    if max(df_ch_model) != 0: df_ch_model = df_ch_model/max(df_ch_model.dropna())

    if max(df_ach_data_median) != 0: df_ach_data_median = df_ach_data_median/max(df_ach_data_median)
    if max(df_ach_model) != 0: df_ach_model = df_ach_model/max(df_ach_model.dropna())

    if max(df_ap_data_median) != 0: df_ap_data_median = df_ap_data_median/max(df_ap_data_median)
    if max(df_ap_model_converted) != 0: df_ap_model_converted = df_ap_model_converted/max(df_ap_model_converted.dropna())

    try:
        rmse_CH = (np.sqrt(mean_squared_error(y_true=df_ch_data_median,
                                              y_pred=df_ch_model,
                                              sample_weight=df_ch_data['weights'])))
        rmse_ACH = (np.sqrt(mean_squared_error(y_true=df_ach_data_median,
                                               y_pred=df_ach_model,
                                               sample_weight=df_ach_data['weights'])))
        rmse_AP = (np.sqrt(mean_squared_error(y_true=df_ap_data_median,
                                              y_pred=df_ap_model_converted,
                                              sample_weight=df_ap_data['weights'])))

    except ValueError as v_error:
        print(v_error)
        RMSE_CH, RMSE_ACH, RMSE_AP = 1e10, 1e10, 1e10

        rmse_CH = RMSE_CH
        rmse_ACH = RMSE_ACH
        rmse_AP = RMSE_AP

    sum_RMSE = np.sum([rmse_CH, rmse_ACH, rmse_AP])
    RMSE_CH = rmse_CH
    RMSE_ACH = rmse_ACH
    RMSE_AP = rmse_AP
    return sum_RMSE


def get_objective_function(p0, params):
    files = get_files()
    df_cyto_data = get_cyto_data(files)
    df_ap_data = get_ap_data(files)
    header = get_header()
    p, w, pred_fle = par.get_params(innate, p0)
    w0 = innate.get_init(w, params)

    df_cyto_model, df_cyto_data = get_concentrations(df_data=df_cyto_data,
                                                     p=p,
                                                     w0=w0,
                                                     header=header,
                                                     blood_parameter='cytokine',
                                                     params=params)
    df_ap_model, df_ap_data = get_concentrations(df_data=df_ap_data,
                                                 p=p,
                                                 w0=w0,
                                                 header=header,
                                                 blood_parameter='ap',
                                                 params=params)

    rmse_p = get_rmse(params, df_cyto_data, df_ap_data, df_cyto_model, df_ap_model)
    rmse_n = get_peaks(p, w0, header, time=[par._stoptime * float(i) / (par._numpoints - 1)
                                            for i in range(par._numpoints)])
    rmse = rmse_p + rmse_n
    print('RMSE is {0}.'.format(rmse))
    return rmse


def set_value(key, value, p0):
    p0[key] = value
    return p0


def set_rule(row, player):
    if 0 <= row['time'] <= 15:
        return 4
    else:
        return 1


def set_weights(df, player):
    df['weights'] = df.apply(lambda row: set_rule(row, player), axis=1)
    return df


def set_to_mins(hrs_lst):
    dys_lst = []
    for h in hrs_lst:
        dys_lst.append(h*60.0)
    return dys_lst


def optimize_params(boundaries, params, p0):
    res = minimize(get_objective_function,
                   x0=list(p0.values()),
                   method=params['method'],
                   args=(params),
                   options={'disp': True},
                   bounds=[boundary for key, boundary in boundaries.items()])
    return res.x


def scan_boundary(boundaries, params, trial):
    global RMSE_CH
    global RMSE_ACH
    global RMSE_AP
    global RMSE_peaks

    optimized = OrderedDict()
    optimized['cost'] = []
    optimized['trial'] = []
    optimized['method'] = []
    optimized['beta_CHMA'] = []
    optimized['beta_CHNA'] = []
    optimized['theta_ACH'] = []
    optimized['beta_MANDA'] = []
    optimized['lamb_ITMNDN'] = []
    optimized['alpha_ITMNDN'] = []
    optimized['Pmax_APE'] = []
    optimized['Pmin_APE'] = []
    optimized['rdistress'] = []
    optimized['w_gauss_min'] = []
    optimized['rinduce_peak'] = []
    optimized['rinduce'] = []
    optimized['r_AP'] = []
    optimized['r_ITM'] = []
    optimized['r_ITMpeak'] = []
    optimized['r_NDN'] = []
    optimized['lamb_MANDN'] = []
    optimized['lamb_MANDA'] = []
    optimized['mu_NDA'] = []
    optimized['Keq_CH'] = []
    optimized['r_Nhomeo'] = []
    optimized['Pmax_NR'] = []

    p0 = get_initial_params(params, trial)  # sets new combinations of parameters
    res = optimize_params(boundaries, params, p0)
    res_dic = dict(zip(list(p0.keys()), res))
    optimized['cost'].append(RMSE_CH + RMSE_ACH + RMSE_AP + RMSE_peaks)
    optimized['trial'].append(trial)
    optimized['method'].append(params['method'])
    for key, res_value in res_dic.items():
        optimized[key].append(res_value)
    init_global()
    return pd.DataFrame(optimized)


def pickle_it(df, destination_folder, trial):
    df.to_pickle('{0}/{1}.pkl'.format(destination_folder, trial, 'optimized'))


def run_experiment(methods, boundaries, destination_folder, params, trial):
    start = timer()
    print('Trial ', str(trial) + ' started ')
    for method in methods:
        params['method'] = method
        df_optimized = scan_boundary(boundaries, params, trial)
        pickle_it(df=df_optimized,
                  destination_folder=destination_folder,
                  trial=trial)
    end = timer()
    print('Trial ', str(trial) + ' finished, execution time: ' + str(end - start))


def run_parallel(methods, boundaries, destination_folder, params, params_to_profile):
    # PARALLEL
    Parallel(n_jobs=10)(delayed(run_experiment)(methods, boundaries, destination_folder, params, trial)
                        for trial in range(params['trials']))

    # NOT parallel
    # for param_profile in params_to_profile:
    #     print(param_profile)
    #     run_experiment(methods, boundaries, destination_folder, params, trial)


if __name__ == '__main__':
    methods = ['TNC']
    boundaries = par.get_boundaries()
    params_to_profile = [key for key, boundary in boundaries.items()]
    destination_folder = 'C:/Users/Alva/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                         'AP Simulator Model/Frontiers in Immunology/Special Issue/Revisions/results/optimization/peaks'
    data_dir = destination_folder
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    params = {'h': 'h4',
              'restrict': False,
              'case': 6,
              'distribution': 'normal',
              'treatment_type': 'biap',
              'trials': 10,
              }
    run_parallel(methods, boundaries, destination_folder, params, params_to_profile)