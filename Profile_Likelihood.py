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
    np.random.seed(int(counter))
    mu_norm = mu / float(max_boundary)
    s_norm = np.random.normal(mu_norm, mu_norm / 3., 1000)
    s = np.random.choice(s_norm)
    if s == 0:
        get_from_normal_dist(mu, counter, max_boundary)
    return s * max_boundary


def get_value_uniform_dist(boundary, counter):
    np.random.seed(int(counter))
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


def get_rmse(params, df_cyto_data, df_ap_data, df_cyto_model, df_ap_model):
    # print("Computing RMSEs...")
    # note: Pro-inflammatory Cytokine = IL6, Anti-inflammatory Cytokine = IL10
    global RMSE_CH
    global RMSE_ACH
    global RMSE_AP

    df_ch_data = df_cyto_data[df_cyto_data['cytokine'] == 'il6']
    df_ach_data = df_cyto_data[df_cyto_data['cytokine'] == 'il10']

    df_ch_model = df_cyto_model['ch']
    df_ach_model = df_cyto_model['ach']
    df_ap_model = np.add(df_ap_model['ap_eblood'], df_ap_model['ap_sblood'])
    df_ap_model_converted = innate.reverse_AP(df_ap_model, 'endo', 'blood')

    df_ch_data_median =df_ch_data['median']
    df_ach_data_median = df_ach_data['median']
    df_ap_data_median = df_ap_data['median']

    try:
        rmse_CH = ((mean_squared_error(y_true=df_ch_data_median,
                                       y_pred=df_ch_model,
                                       sample_weight=df_ch_data['weights']))
                   / np.sum((df_ch_data_median - np.median(df_ch_data_median))**2))**2
        rmse_ACH = ((mean_squared_error(y_true=df_ach_data_median,
                                        y_pred=df_ach_model,
                                        sample_weight=df_ach_data['weights']))
                    / np.sum((df_ach_data_median - np.median(df_ach_data_median))**2))**2

        rmse_AP = ((mean_squared_error(y_true=df_ap_data_median,
                                       y_pred=df_ap_model_converted,
                                       sample_weight=df_ap_data['weights']))
                   / np.sum((df_ap_data_median - np.median(df_ap_data_median))**2))**2
    except ValueError as v_error:
        print(v_error)
        rmse_CH = RMSE_CH
        rmse_ACH = RMSE_ACH
        rmse_AP = RMSE_AP

    # print('RMSE for CH : ', rmse_CH)
    # print('RMSE for ACH : ', rmse_ACH)
    # print('RMSE for AP : ', rmse_AP)

    sum_RMSE = np.sum([rmse_CH, rmse_ACH, rmse_AP])
    RMSE_CH = rmse_CH
    RMSE_ACH = rmse_ACH
    RMSE_AP = rmse_AP
    return sum_RMSE


def get_objective_function(p0, params):
    # print("Getting the objective function...")
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


    rmse = get_rmse(params, df_cyto_data, df_ap_data, df_cyto_model, df_ap_model)

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


def optimize_params(boundaries, p0, params):
    res = minimize(get_objective_function,
                   x0=list(p0.values()),
                   method=params['method'],
                   args=(params),
                   options={'disp': True},
                   bounds=[boundary for key, boundary in boundaries.items()])
    return res.x


def scan_boundary(boundaries, params, destination_folder, param_profile):
    global RMSE_CH
    global RMSE_ACH
    global RMSE_AP

    optimized = {
        'profile': [],
        'cost': [],
        'value': [],
        'trial': [],
        'method': [],
        'beta_CHMA': [],
        'beta_CHNA': [],
        'theta_ACH': [],
        'beta_MANDA': [],
        'lamb_ITMNDN': [],
        'alpha_ITMNDN': [],
        'Pmax_APE': [],
        'Pmin_APE': [],
        'rdistress': [],
        'w_gauss_min': [],
        'rinduce_peak': [],
        'rinduce': [],
        'r_AP': [],
        'r_ITM': [],
        'r_ITMpeak': [],
        'r_NDN': [],
        'lamb_MANDN': [],
        'lamb_MANDA': [],
        'mu_NDA': [],
        'Keq_CH': [],
        'r_Nhomeo': [],
        'Pmax_NR': []
    }

    # for key, boundary in tqdm(boundaries.items(), desc="boundary loop"):

    interval = np.linspace(boundaries[param_profile][0], boundaries[param_profile][1], params['interval'])
    for trial in tqdm(range(params['trials']), desc="trial loop"):
        for value in tqdm(interval, desc="interval loop"):
            p0 = get_initial_params(params, trial)  # sets new combinations of parameters
            p0 = set_value(param_profile, value, p0)

            beta_CHMA, beta_CHNA, theta_ACH, beta_MANDA, lamb_ITMNDN, alpha_ITMNDN, Pmax_APE, Pmin_APE, rdistress, \
            w_gauss_min, rinduce_peak, rinduce, r_ITM, r_ITMpeak, r_NDN, r_AP, lamb_MANDN, lamb_MANDA, mu_NDA, \
            Keq_CH, r_Nhomeo, Pmax_NR = optimize_params(boundaries, p0, params)

            optimized['profile'].append(param_profile)
            optimized['value'].append(value)
            optimized['cost'].append(RMSE_CH + RMSE_ACH + RMSE_AP)
            optimized['trial'].append(trial)
            optimized['method'].append(params['method'])
            optimized['beta_CHMA'].append(beta_CHMA)
            optimized['beta_CHNA'].append(beta_CHNA)
            optimized['theta_ACH'].append(theta_ACH)
            optimized['beta_MANDA'].append(beta_MANDA)
            optimized['lamb_ITMNDN'].append(lamb_ITMNDN)
            optimized['alpha_ITMNDN'].append(alpha_ITMNDN)
            optimized['Pmax_APE'].append(Pmax_APE)
            optimized['Pmin_APE'].append(Pmin_APE)
            optimized['rdistress'].append(rdistress)
            optimized['w_gauss_min'].append(w_gauss_min)
            optimized['rinduce_peak'].append(rinduce_peak)
            optimized['rinduce'].append(rinduce)
            optimized['r_ITM'].append(r_ITM)
            optimized['r_ITMpeak'].append(r_ITMpeak)
            optimized['r_NDN'].append(r_NDN)
            optimized['r_AP'].append(r_AP)
            optimized['lamb_MANDN'].append(lamb_MANDN)
            optimized['lamb_MANDA'].append(lamb_MANDA)
            optimized['mu_NDA'].append(mu_NDA)
            optimized['Keq_CH'].append(Keq_CH)
            optimized['r_Nhomeo'].append(r_Nhomeo)
            optimized['Pmax_NR'].append(Pmax_NR)

            init_global()
        pickle_it(pd.DataFrame(optimized), destination_folder, params, param_profile, trial)
    return pd.DataFrame(optimized), trial


def pickle_it(df, destination_folder, params, param_profile, trial):
    if trial < params['trials']:
        df.to_pickle('{0}/{1}_{2}_{3}.pkl'.format(destination_folder, trial, param_profile, 'optimized'))
    else:
        df.to_pickle('{0}/{1}_{2}.pkl'.format(destination_folder, param_profile, 'optimized'))


def run_experiment(methods, boundaries, destination_folder, params, param_profile):
    start = timer()
    print(param_profile + ' started ')
    for method in methods:
        params['method'] = method
        df_optimized, trial = scan_boundary(boundaries, params, destination_folder, param_profile)
        pickle_it(df=df_optimized,
                  destination_folder=destination_folder,
                  params=params,
                  param_profile=param_profile,
                  trial=trial)
    end = timer()
    print(param_profile + ' finished, execution time: ' + str(end - start))


def run_parallel(methods, boundaries, destination_folder, params, params_to_profile):
    Parallel(n_jobs=10)(delayed(run_experiment)(methods, boundaries, destination_folder, params, param_profile)
                        for param_profile in params_to_profile)

if __name__ == '__main__':
    # methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    methods = ['TNC']
    boundaries = par.get_boundaries()
    params_to_profile = [key for key, boundary in boundaries.items()]
    destination_folder = 'C:/Users/Alva/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                         'AP Simulator Model/Frontiers in Immunology/Special Issue/Revisions/results'
    data_dir = destination_folder
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    params = {'h': 'h4',
              'restrict': False,
              'case': 6,
              'distribution': 'normal',
              'interval': 30,
              'treatment_type': 'biap',
              'trials': 10,
              }
    run_parallel(methods, boundaries, destination_folder, params, params_to_profile)