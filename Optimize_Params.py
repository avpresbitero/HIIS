from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from math import sqrt
from memory_profiler import profile

import numpy as np
import gc

import Parameters as prms
import APSim_Model as ap
import pandas as pd
import APPIREDII_Cytokines_Parser as cyto
import APPIREDII_Blood_Params_Parser as dp

RMSE_CH = 0
RMSE_ACH = 0
RMSE_AP = 0
RMSE_MEAN = 0


def init_global():
    global RMSE_CH
    global RMSE_ACH
    global RMSE_AP
    global RMSE_MEAN


def fine_tune(p0, step, method, prms, collect, trial, norm_conv,header, df_act_cyto, df_act_ap, h,
              bounds, restrict):
    p0 = get_initial_params(trial, p0, bounds, collect)
    print(' ')
    print('------------------------------------- Initiating Optimization process -------------------------------------')
    rmse = do_step(p0, step, df_act_cyto, df_act_ap, header, prms, collect, trial, norm_conv, h, restrict)
    print(' ')
    print('FINAL RMSE FOR STEP ', step, ' ................................. ', rmse)
    collected = gc.collect()
    print('Garbage collector at fine_tune method: collected %d objects.' % collected)
    del collected
    return rmse


def get_header():
    header = ['N_R', 'AP_Eblood', 'AP_Etissue', 'AP_Eliver', 'AP_Sblood', 'AP_Stissue', 'ITMblood', 'ITMtissue', 'M_R',
              'M_A', 'CH', 'N_A', 'ND_A', 'ACH', 'ND_N']
    header = [name.lower() for name in header]
    return header


def header_to_lowercase(df):
    df.columns = map(str.lower, df.columns)
    return df


def to_days(hrs_lst):
    dys_lst = []
    for h in hrs_lst:
        dys_lst.append(h/24.0)
    return dys_lst


def to_mins(hrs_lst):
    dys_lst = []
    for h in hrs_lst:
        dys_lst.append(h*60.0)
    return dys_lst


def rule(row, player):
    if row['time'] >= 0 and row['time'] <= 15:
        return 4
    else:
        return 1


def add_weights(df, player):
    df['weights'] = df.apply(lambda row: rule(row, player), axis=1)
    return df


def get_cyto(cyto_fle, sample_fle, trt_fle, cyto):
    df_cyto, cyto_dic = cyto.get_median(cyto_fle, sample_fle, trt_fle, 'time')
    print(' ')
    print('Getting Actual Cytokine Concentrations...')

    del cyto_dic
    collected = gc.collect()
    print('Garbage collector at get_cyto method: collected %d objects.' % collected)
    del collected
    return df_cyto


def get_AP(german_hospital_file, dutch_hospital_file, treatment_file, parsed_data):
    df_AP, df_AP_median = parsed_data.get_data(german_hospital_file=german_hospital_file,
                                               dutch_hospital_file=dutch_hospital_file,
                                               treatment_file=treatment_file,
                                               parameter_name='AlkFosf',
                                               restrict=False,
                                               treatment=None,
                                               patients=None,
                                               do_binning=True)
    print('Getting Actual AP Concentrations...')
    df_AP_median = add_weights(df_AP_median, 'ap')
    return df_AP_median


def restrict_actual(df_act, player, step):
    """
    :param df_act: dataframe of data for cytokines and AP
    :param innate: innate object necessary to exract some methods
    :param player: either 'Cytokine' or 'AP'
    :return: df_restrict
            - if 'Cytokine', this dataframe contains medians for all four Cytokines corresponding to a specific
              treatment (Placebo or bIAP)
            - if 'AP', this dataframe contains median corresponding to a specific treatment (Placebo or bIAP)
              time_restrict
            - list of time (in days) that corresponds to available data
    :param step: either 'Placebo' or 'bIAP'
    """
    df_restrict = df_act.loc[(df_act['time'] >= 0) & (df_act['time'] <= 36.0)]
    df_restrict = df_restrict[df_restrict['treatment'] == step]
    if player == 'cytokine':
        # sample cytokine to retrieve time list
        df_restrict_cyto = df_restrict[df_restrict['cytokine'] == 'il10']
        time_restrict = df_restrict_cyto[df_restrict_cyto['treatment'] == step]['time']
    elif player == 'ap':
        time_restrict = df_restrict[df_restrict['treatment'] == step]['time']
    time_restrict = to_mins(time_restrict)
    df_restrict.loc[:, 'time'] *= 60
    return df_restrict, time_restrict


def process_player(df_act, innate, p, w0, pred_fle, header, player, step):
    print(' ')
    print('Solving Differential Equations for ', player, ' timeline...')
    df_act_rstrct, time_rstrct = restrict_actual(df_act, player, step)
    time, pred = innate.solve(p, w0, pred_fle, time_rstrct, False)
    df_pred = pd.DataFrame(pred)
    keys = df_pred.columns.values
    dictionary = dict(zip(keys, header))
    df_pred = df_pred.rename(columns=dictionary)
    df_pred['time'] = time
    df_act_rstrct = add_weights(df_act_rstrct, player)

    del pred
    del time
    del keys
    del dictionary
    del time_rstrct

    collected = gc.collect()
    print('Garbage collector at process_player method: collected %d objects.' % collected)
    del collected
    return df_pred, df_act_rstrct


def process_peaks(innate, p, w0, pred_fle, header, time):
    innate.case = 5
    time, pred = innate.solve(p, w0, pred_fle, time, False)
    df_pred = pd.DataFrame(pred)
    keys = df_pred.columns.values
    dictionary = dict(zip(keys, header))
    df_pred = df_pred.rename(columns=dictionary)
    placebo_n_a = df_pred['n_a']
    placebo_nd_n = df_pred['nd_n']
    placebo_nd_a = df_pred['nd_a']
    placebo_d = df_pred['d']


    innate.case = 6
    time, pred = innate.solve(p, w0, pred_fle, time, False)
    df_pred = pd.DataFrame(pred)
    keys = df_pred.columns.values
    dictionary = dict(zip(keys, header))
    df_pred = df_pred.rename(columns=dictionary)
    supplemented_n_a = df_pred['n_a']
    supplemented_nd_n = df_pred['nd_n']
    supplemented_nd_a = df_pred['nd_a']
    supplemented_d = df_pred['d']

    if max(placebo_nd_n) > max(supplemented_nd_n):
        nd_n_error = 0
    if max(placebo_nd_n) <= max(supplemented_nd_n):
        nd_n_error = 1
    if max(placebo_nd_a) < max(supplemented_nd_a):
        nd_a_error = 1
    if max(placebo_nd_a) >= max(supplemented_nd_a):
        nd_a_error = 0
    if max(placebo_d) < max(supplemented_d):
        d_error = 1
    if max(placebo_d) >= max(supplemented_d):
        d_error = 0


    rmse_n_a = sqrt(((min(supplemented_n_a) - min(placebo_n_a))-(40. * (min(placebo_n_a)+1)))**2)

    sum_error = sum([nd_n_error, nd_a_error, rmse_n_a, d_error])
    print('Error for Neutrophils : ', sum_error)
    return sum_error


def get_rmse(step, df_act_cyto, df_act_AP, df_pred_cyto, df_mod_AP, innate, prms, norm_conv):
    # note: Pro-inflammatory Cytokine = IL6, Anti-inflammatory Cytokine = IL10
    global RMSE_CH
    global RMSE_ACH
    global RMSE_AP

    df_act_CH = df_act_cyto[df_act_cyto['cytokine'] == 'il6']
    df_act_ACH = df_act_cyto[df_act_cyto['cytokine'] == 'il10']
    print(df_act_CH)
    print(df_act_ACH)

    df_pred_CH = df_pred_cyto['ch']
    df_pred_ACH = df_pred_cyto['ach']
    df_pred_AP = np.add(df_mod_AP['ap_eblood'], df_mod_AP['ap_sblood'])
    df_pred_AP_conv = innate.reverse_AP(df_pred_AP, 'endo', 'blood')

    print(df_pred_CH)
    print(df_pred_ACH)

    df_act_CH_med =df_act_CH['median']
    df_act_ACH_med = df_act_ACH['median']
    df_act_AP_med = df_act_AP['median']

    if norm_conv == 'norm_conv_':
        rmse_CH = sqrt((mean_squared_error(df_act_CH_med, df_pred_CH, sample_weight=df_act_CH['weights']))
                       * len(df_act_CH_med)/ np.sum((df_act_CH_med - np.median(df_act_CH_med))**2))
        rmse_ACH = sqrt((mean_squared_error(df_act_ACH_med, df_pred_ACH, sample_weight=df_act_ACH['weights']))
                        * len(df_act_ACH_med)/np.sum((df_act_ACH_med - np.median(df_act_ACH_med))**2))
        rmse_AP = sqrt((mean_squared_error(df_act_AP_med, df_pred_AP_conv, sample_weight=df_act_AP['weights']))
                       * len(df_act_AP_med) / np.sum((df_act_AP_med - np.median(df_act_AP_med))**2))
    else:
        rmse_CH = sqrt(mean_squared_error(df_act_CH_med, df_pred_CH, sample_weight=df_act_CH['weights']))
        rmse_ACH = sqrt(mean_squared_error(df_act_ACH_med, df_pred_ACH, sample_weight=df_act_ACH['weights']))
        rmse_AP = sqrt(mean_squared_error(df_act_AP_med, df_pred_AP_conv, sample_weight=df_act_AP['weights']))

    print('For step : ', step)
    print('RMSE for CH : ', rmse_CH)
    print('RMSE for ACH : ', rmse_ACH)
    print('RMSE for AP : ', rmse_AP)

    sum_RMSE = np.sum([rmse_CH, rmse_ACH, rmse_AP])

    RMSE_CH = rmse_CH
    RMSE_ACH = rmse_ACH
    RMSE_AP = rmse_AP

    collected = gc.collect()
    print('Garbage collector at get_rmse method: collected %d objects.' % collected)
    return sum_RMSE


def do_step(p0, step, df_act_cyto, df_act_AP, header, innate_objcts, prms, collect, trial, norm_conv, h, restrict):
    params={}
    if step == 'Placebo':
        params['case'] = 5
        params['h'] = h
    elif step == 'bIAP':
        params['case'] = 6
        params['h'] = h

    rmse_lst = []
    time = [36. * 60. * float(i) / (10000 - 1) for i in range(10000)]

    params['restrict'] = restrict
    p, w, pred_fle = prms.get_params(ap, p0)
    w0 = ap.get_init(w, params)

    df_pred_cyto_step, df_act_cyto_step = process_player(df_act_cyto, ap, p, w0, pred_fle, header, 'cytokine', step)
    df_pred_AP_step, df_act_AP_step = process_player(df_act_AP, ap, p, w0, pred_fle, header, 'ap', step)
    rmse = get_rmse(step, df_act_cyto_step, df_act_AP_step, df_pred_cyto_step, df_pred_AP_step, ap, prms, norm_conv)
    n_error = process_peaks(ap, p, w0, pred_fle, header, time)
    rmse_lst.append(rmse)
    rmse_lst.append(n_error)

    collected = gc.collect()
    print('Garbage collector at step method: collected %d objects.' % collected)
    return np.mean(rmse_lst)


def get_val_distr(mu, trial, max_bound):
    s = 0
    np.random.seed(trial)
    mu_norm = mu/float(max_bound)
    s_norm = np.random.normal(mu_norm, mu_norm/3., 1000)
    while s == 0:
        s = np.random.choice(s_norm)
    collected = gc.collect()
    print('Garbage collector at get_val_distr method: collected %d objects.' % collected)
    return s*max_bound


def get_val_range(bound, trial):
    np.random.seed(trial)
    try:
        lst = np.linspace(bound[0], bound[1], 100)
    except:
        lst = np.linspace(bound[0], 1*10**9, 100)
    return np.random.choice(lst)


def get_original_params(step):
    if step == 'bIAP':
        beta_CHMA = 5.5e-4
        beta_CHNA = 3.95e-2
        theta_ACH = 1e-10
        beta_MANDA = 9e1
        lamb_ITMNDN = 1.e-6
        alpha_ITMNDN = 1e3
        Pmax_APE = 0.002
        Pmin_APE = 0.0002
        rdistress = 3e6  # 1e6
        w_gauss_min = 8e7
        rinduce_peak = 21.0
        rinduce = 0.05
        r_AP = 0.08
        r_ITM = 0.5
        r_ITMpeak = 5 * 10 ** 12
        r_NDN = 0.008
        lamb_MANDN = 5e-6
        lamb_MANDA = 3e-5
        mu_NDA = 2.5e1
        Keq_CH = 2e4
        r_Nhomeo = 1e-4
        Pmax_NR = 6e-3

    elif step == 'Placebo':
        beta_CHMA = 1e-3
        beta_CHNA = 4.8e-2
        theta_ACH = 1e-10
        beta_MANDA = 9e1
        lamb_ITMNDN = 8e-7
        alpha_ITMNDN = 9.9e2
        Pmax_APE = 0.002
        Pmin_APE = 0.0002
        rdistress = 3e6  # 1e6
        w_gauss_min = 8e7
        rinduce_peak = 5.0
        rinduce = 0.009
        r_ITM = 0.5
        r_ITMpeak = 5 * 10 ** 12
        r_NDN = 0.008
        r_AP = 0.08
        lamb_MANDN = 6.5e-6
        lamb_MANDA = 3e-5
        mu_NDA = 2.5e1
        Keq_CH = 2e4
        r_Nhomeo = 1e-4
        Pmax_NR = 6e-3

    return [beta_CHMA, beta_CHNA, theta_ACH, beta_MANDA, lamb_ITMNDN, alpha_ITMNDN, Pmax_APE, Pmin_APE, rdistress,
            w_gauss_min, rinduce_peak, rinduce, r_ITM, r_ITMpeak, r_NDN, r_AP, lamb_MANDN, lamb_MANDA, mu_NDA, Keq_CH,
            r_Nhomeo, Pmax_NR]


def get_initial_params(trial, p0, bounds, collect):
    temp_lst = []
    if collect == 'distribution':
        for i in range(len(p0)):
            param = p0[i]
            max_bound = max(bounds[i])
            temp_lst.append(get_val_distr(param, trial, max_bound))

    elif collect == 'list':
        for bound in bounds:
            temp_lst.append(get_val_range(bound, trial))

    beta_CHMA, beta_CHNA, theta_ACH, beta_MANDA, lamb_ITMNDN, alpha_ITMNDN, Pmax_APE, Pmin_APE, rflush, \
    gauss_min, rinduce_peak, rinduce, r_ITM, r_ITMpeak, r_NDN, r_AP, lamb_MANDN, lamb_MANDA, mu_NDA, \
    Keq_CH, r_Nhomeo, Pmax_NR = temp_lst

    collected = gc.collect()
    print('Garbage collector at get_initial_params method: collected %d objects.' % collected)

    return [beta_CHMA, beta_CHNA, theta_ACH, beta_MANDA, lamb_ITMNDN, alpha_ITMNDN, Pmax_APE, Pmin_APE, rflush,
            gauss_min, rinduce_peak, rinduce, r_ITM, r_ITMpeak, r_NDN, r_AP, lamb_MANDN, lamb_MANDA, mu_NDA,
            Keq_CH, r_Nhomeo, Pmax_NR]


def get_norm_params(normalized):
    if normalized:
        norm_conv = 'norm_conv_'
        norm = 'norm_'
    else:
        norm_conv = 'conv_'
        norm = ''
    return norm_conv, norm


def get_start(restart, collect, method, norm, step, h):
    if restart:
        return 0
    fle = 'result/optimize/RMSE/' + collect + '/' + h + '/' + method + '/' + norm + method + '_' + step + '_' + 'RMSE.txt'
    df = pd.read_csv(fle, delimiter='\t')
    return max(df['trial'])+1


def do_optimization(trial, step, bounds, collect, method, norm_conv, norm, p0, header, df_act_cyto, df_act_ap, h, restrict):
    print('ENTERING TRIAL : ', trial)
    res = minimize(fine_tune, p0,
                   method=method,
                   args=(step, method, prms, collect, trial, norm_conv, header, df_act_cyto, df_act_ap, h, bounds,
                         restrict),
                   options={'disp': True},
                   bounds=bounds)

    beta_CHMA, beta_CHNA, theta_ACH, beta_MANDA, lamb_ITMNDN, alpha_ITMNDN, Pmax_APE, Pmin_APE, rflush, \
    gauss_min, rinduce_peak, rinduce, r_ITM, r_ITMpeak, r_NDN, r_AP, lamb_MANDN, lamb_MANDA, mu_NDA, \
    Keq_CH, r_Nhomeo, Pmax_NR = res.x

    print('METHOD : ', method)
    print('FINAL CALIBRATED VALUES FOR TRIAL ', trial, ' :')
    print('beta_CHMA = ', abs(beta_CHMA))
    print('beta_CHNA =', abs(beta_CHNA))
    print('theta_ACH =', abs(theta_ACH))
    print('beta_MANDA =', abs(beta_MANDA))
    print('lamb_ITMNDN = ', abs(lamb_ITMNDN))
    print('alpha_ITMNDN = ', abs(alpha_ITMNDN))
    print('Pmax_APE =', abs(Pmax_APE))
    print('Pmin_APE =', abs(Pmin_APE))
    print('rflush =', abs(rflush))
    print('gauss_min =', abs(gauss_min))
    print('rinduce_peak =', abs(rinduce_peak))
    print('rinduce = ', abs(rinduce))
    print('r_ITM = ', abs(r_ITM))
    print('r_ITMpeak = ', abs(r_ITMpeak))
    print('r_NDN = ', abs(r_NDN))
    print('r_AP = ', abs(r_AP))
    print('lamb_MANDN = ', abs(lamb_MANDN))
    print('lamb_MANDA = ', abs(lamb_MANDA))
    print('mu_NDA = ', abs(mu_NDA))
    print('Keq_CH = ', abs(Keq_CH))
    print('r_Nhomeo = ', abs(r_Nhomeo))
    print('Pmax_NR = ', abs(Pmax_NR))

    # TODO put in a separate function, implement dataframe
    # with open('result/optimize/RMSE/' + collect + '/' + h + '/' + method + '/' + norm + method + '_' + step
    #            + '_RMSE.txt', 'a') as file:
    #     file.write(str(trial) + '\t' + str(RMSE_CH) + '\t' + str(RMSE_ACH) + '\t' + str(RMSE_AP)
    #                + '\t' + str(RMSE_MEAN) + '\n')
    #
    # with open('result/optimize/calibrated/' + collect + '/' + h + '/' + method + '/' + norm + method + '_' + step + '.txt',
    #           'a') as file:
    #     file.write(
    #         str(trial) + '\t' + str(abs(beta_CHMA)) + '\t' + str(abs(beta_CHNA)) + '\t' +
    #         str(abs(theta_ACH)) + '\t' + str(abs(beta_MANDA)) + '\t' + str(abs(lamb_ITMNDN)) + '\t' +
    #         str(abs(alpha_ITMNDN)) + '\t' + str(abs(Pmax_APE)) + '\t' + str(abs(Pmin_APE)) + '\t' +
    #         str(abs(rflush)) + '\t' + str(abs(gauss_min)) + '\t' + str(abs(rinduce_peak)) + '\t' +
    #         str(abs(rinduce)) + '\t' + str(abs(r_ITM)) + '\t' +
    #         str(abs(r_ITMpeak)) + '\t' + str(abs(r_NDN)) + '\t' + str(abs(r_AP)) + '\t' +
    #         str(abs(lamb_MANDN)) + '\t' + str(abs(lamb_MANDA)) + '\t' +
    #         str(abs(mu_NDA)) + '\t' + str(abs(Keq_CH)) + '\t' + str(abs(r_Nhomeo)) + '\t' +
    #         str(abs(Pmax_NR)) + '\n')
    init_global()
    print('Collecting Garbage...')
    collected = gc.collect()
    print('Garbage collector: collected %d objects.' % collected)
    return res.x


def main(step, hypothesis, restrict, collect):
    cyto_fle = prms.cyto_fle
    sample_fle = prms.sample_fle
    treatment_file = prms.trt_fle
    AP_file = prms.AP_fle
    german_hospital_file = prms.german_hospital_file
    dutch_hospital_file = prms.dutch_hospital_file

    # TODO if you run al of them, why you did not use cycle?
    methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    methods = ['L-BFGS-B']

    # from multiprocessing import Pool
    # from functools import partial

    parsed_data = dp.Data_Parser()
    df_act_cyto = get_cyto(cyto_fle, sample_fle, treatment_file, cyto)
    df_act_ap = get_AP(german_hospital_file, dutch_hospital_file, treatment_file, parsed_data)

    for method in methods:
        trials = 500
        # collect = 'distribution'
        # collect = 'list'
        bounds = prms.get_boundaries()
        norm_conv, norm = get_norm_params(normalized=True)
        p0 = get_original_params(step=step,
                                 hypothesis=hypothesis)
        header = get_header()

        # jobs = 12
        # pool = Pool(jobs)
        #
        # make_iteration = partial(do_optimization,
        #     step=step,
        #     start=start,
        #     bounds=bounds,
        #     collect=collect,
        #     method=method,
        #     cyto_fle=cyto_fle,
        #     sample_fle=sample_fle,
        #     trt_fle=trt_fle,
        #     AP_file=AP_file,
        #     norm_conv=norm_conv,
        #     norm=norm,
        #     p0=p0,
        #     header=header,
        #     df_act_cyto=df_act_cyto,
        #     df_act_ap=df_act_ap,
        #     h=h)
        #
        # # make_iteration(1)
        #
        # data = pool.map(make_iteration, [trial for trial in range(trials)])
        # persist_data(data)

        for trial in range(trials):
            beta_CHMA, beta_CHNA, theta_ACH, beta_MANDA, lamb_ITMNDN, alpha_ITMNDN, Pmax_APE, Pmin_APE, rflush, \
            gauss_min, rinduce_peak, rinduce, r_ITM, r_ITMpeak, r_NDN, r_AP, lamb_MANDN, lamb_MANDA, mu_NDA, Keq_CH, \
            r_Nhomeo, Pmax_NR = do_optimization(trial, step, bounds, collect, method, norm_conv, norm, p0, header,
                                                df_act_cyto, df_act_ap, hypothesis, restrict)
            print('Collecting Garbage...')
            collected = gc.collect()
            print('Garbage collector in main method: collected %d objects.' % collected)

if __name__ == '__main__':
    main(step='bIAP', hypothesis='h4', restrict=False, collect='distribution')

