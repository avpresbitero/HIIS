import numpy as np
import pandas as pd
import seaborn as sns

import Parameters as params
import APSim_Model as hiis

from sklearn.metrics import mean_squared_error
from math import sqrt


def get_APE(param_bound, param_dic, param_name, innate, w0, mod_fle, time_rstrct):
    prms_lst = np.linspace(param_bound[0], param_bound[1], 1000)
    AP_area_list = []
    for param in prms_lst:
        param_dic[param_name] = param
        p = get_initial_params(param_dic)
        time, model = innate.solve(p, w0, mod_fle, time_rstrct, False)
        df_model = pd.DataFrame(model)
        keys = df_model.columns.values
        header = get_header()
        dictionary = dict(zip(keys, header))
        df_pred = df_pred.rename(columns=dictionary)
        df_pred['Time'] = time
        APE_list = df_pred['AP_Eblood'].tolist()
        area = get_area(APE_list, time)
        AP_area_list.append(area)
    return prms_lst, AP_area_list


def get_area(lst, time):
    area=0
    for i in range(len(lst)):
        if i>0:
            area+=lst[i]*(time[i]-time[i-1])
    return area


def get_header():
    header = ['N_R', 'AP_Eblood', 'AP_Etissue', 'AP_Eliver', 'AP_Sblood', 'AP_Stissue', 'ITMblood', 'ITMtissue',
              'D', 'M_R', 'M_A', 'CH', 'N_A', 'ND_A', 'G', 'ACH', 'ND_N']
    return header


def get_best_params(step, norm, bound, h):
    df_params = pd.read_csv('result/optimize/params/' + h + '_' + bound + norm + step + '_best_params.txt', sep='\t')
    beta_CHNA = df_params['beta_CHNA'].values[0]
    beta_CHMA = df_params['beta_CHMA'].values[0]
    theta_ACH = df_params['theta_ACH'].values[0]
    alpha_ACHMA = df_params['alpha_ACHMA'].values[0]
    beta_MANDA = df_params['beta_MANDA'].values[0]
    Pmin_APE = df_params['Pmin_APE'].values[0]
    Pmax_APE = df_params['Pmax_APE'].values[0]
    rflush = df_params['rflush'].values[0]
    rinduce = df_params['rinduce'].values[0]
    gauss_min = df_params['gauss_min'].values[0]
    lamb_ITMNDN = df_params['lamb_ITMNDN'].values[0]
    lamb_MANDN = df_params['lamb_MANDN'].values[0]
    alpha_ITMNDN = df_params['alpha_ITMNDN'].values[0]
    alpha_ITMD = df_params['alpha_ITMD'].values[0]
    lamb_MANDA = df_params['lamb_MANDA'].values[0]

    return [beta_CHNA, beta_CHMA, theta_ACH, alpha_ACHMA, beta_MANDA, Pmin_APE, Pmax_APE, rflush, gauss_min, rinduce,
            lamb_ITMNDN, lamb_MANDN, alpha_ITMNDN, alpha_ITMD, lamb_MANDA]


def get_initial_params(param_dic, step):
    beta_CHNA = param_dic['beta_CHNA']
    beta_CHMA = param_dic['beta_CHMA']
    theta_ACH = param_dic['theta_ACH']
    alpha_ACHMA = param_dic['alpha_ACHMA']
    beta_MANDA = param_dic['beta_MANDA']
    Pmin_APE = param_dic['Pmin_APE']
    Pmax_APE = param_dic['Pmax_APE']
    rflush = param_dic['rflush']
    gauss_min = param_dic['gauss_min']
    rinduce = param_dic['rinduce']

    if step == 'Placebo':
        return [beta_CHNA, beta_CHMA, theta_ACH, alpha_ACHMA, beta_MANDA, Pmin_APE, Pmax_APE, rflush, gauss_min]
    elif step == 'bIAP':
        return [beta_CHNA, beta_CHMA, theta_ACH, alpha_ACHMA, beta_MANDA, Pmin_APE, Pmax_APE, rflush, gauss_min,
                rinduce]


def create_dic(p0, params):
    param_dic = {}
    for i in range(len(p0)):
        param_name = params[i]
        param_dic[param_name]=p0[i]
    return param_dic


def get_param_names():
    return ['beta_CHNA', 'beta_CHMA', 'theta_ACH', 'alpha_ACHMA', 'beta_MANDA', 'Pmin_APE', 'Pmax_APE', 'rflush',
            'gauss_min', 'rinduce']


def get_df_model(model, time):
    header = get_header()
    df_model = pd.DataFrame(model)
    keys = df_model.columns.values
    dictionary = dict(zip(keys, header))
    df_model = df_model.rename(columns=dictionary)
    df_model['Time'] = time
    return df_model


def set_time(t, timein, timeout):
    tlist = []
    for i in t:
        if timeout ==  'hours':
            if timein == 'mins':
                i = i / 60
        tlist.append(i)
    return tlist


def get_model(step, prms, bounds, t, df_calib, norm, player, bound, h):
    params = get_param_names()
    dic = {}
    df_rmse = pd.DataFrame()
    for i in range(len(bounds)):
        param_name = params[i]
        param_bound = bounds[i]
        start, end = param_bound[0], param_bound[1]
        df = pd.DataFrame()
        param_vals = np.linspace(start, end, 10)
        rmse_lst = []
        for j in param_vals:
            print(' ')
            print('Computing for parameter ', param_name, ' ....')
            print('Parameter Value Tested : ', j)
            innate = hiis.Innate()
            innate = set_step(innate, step)
            innate = set_h(innate, h)
            p0 = get_best_params(step, norm, bound, h)
            p0[i] = j
            p, w, pred_fle = prms.get_params(innate, p0)
            w0 = innate.get_init(w)
            time, model = innate.solve(p, w0, pred_fle, t, False)
            df_model = get_df_model(model, time)
            df[j] = df_model[player]
            rmse_lst.append(sqrt(mean_squared_error(df[j], df_calib['calibrated'])))
            time = set_time(time, 'mins', 'hours')
            df['Time'] = time
        dic[param_name] = df
        df_rmse[param_name] = rmse_lst
    return dic, df_rmse


def get_orig_conc(innate, step, prms, t, norm, player, bound, h):
    df_calib = pd.DataFrame()
    innate = set_step(innate, step)
    innate = set_h(innate, h)
    p0 = get_best_params(step, norm, bound, h)
    p, w, pred_fle = prms.get_params(innate, p0)
    w0 = innate.get_init(w)
    time, model = innate.solve(p, w0, pred_fle, t, False)
    df_model = get_df_model(model, time)
    df_calib['calibrated'] = df_model[player]
    time = set_time(time, 'mins', 'hours')
    df_calib['Time'] = time
    return df_calib


def set_step(innate, step):
    if step == 'Placebo':
        innate.case = 5
    elif step == 'bIAP':
        innate.case = 6
    return innate


def set_h(innate, h):
    innate.h = h
    return innate


def normalize_df(df):
    df_norm = (df - df.mean()) / (df.max() - df.min())
    return df_norm


def plot_params(dic, df_calib):
    lw = 2
    ls = 20
    fs = 20
    lgs = 10

    sns.set(style="white")
    param_names = get_param_names()
    #style.use('fivethirtyeight')
    for i in range(len(param_names)):
        param_name = param_names[i]
        print('Plotting ', param_name, ' ...')
        df = pd.DataFrame(dic[param_name])
        sns.plt.figure(i)
        ax = df.plot(x='Time', linewidth=lw, color=sns.color_palette("RdBu", n_colors=11), legend=False)
        df_calib.plot(x='Time', linewidth=lw, color='black', ax=ax)
        sns.plt.xlabel("Hours After Operation", fontsize=fs)
        #sns.plt.ylabel("Endogenous Alkaline Phosphatase", fontsize=fs)
        #sns.plt.ylabel("RMSEs", fontsize=fs)
        sns.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': lgs}, frameon=True)
        #sns.plt.ylim((0, 1e9))
        sns.plt.title('Varying '+ param_name, fontsize=fs)
        sns.plt.tick_params(labelsize=ls)
        sns.plt.savefig('result/plots/sensitivity/' + param_name + '.png', format='png', dpi=500, bbox_inches='tight')
    sns.plt.show()


def plot_RMSE(df_rmse):
    lw = 2
    ls = 20
    fs = 20
    lgs = 10

    sns.set(style="white")
    param_names = get_param_names()
    # style.use('fivethirtyeight')
    df_rmse = normalize_df(df_rmse)
    df_rmse.plot(linewidth=lw, legend=False)
    sns.plt.xlabel("Normalized Parameter Bounds", fontsize=fs)
    sns.plt.ylabel("RMSEs", fontsize=fs)
    sns.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': lgs}, frameon=True)
    sns.plt.title('RMSE', fontsize=fs)
    sns.plt.tick_params(labelsize=ls)
    sns.plt.savefig('result/plots/sensitivity/RMSEs.png', format='png', dpi=500, bbox_inches='tight')
    sns.plt.show()

def main(step):
    norm = 'norm_'
    bound ='bound_'
    player = 'AP_Eblood'
    #player = 'CH'
    h = 'h2'

    prms = params.Parameters()
    innate = hiis.Innate()
    t = [prms._stoptime * float(i) / (prms._numpoints - 1) for i in range(prms._numpoints)]

    bounds = prms.get_bounds()

    df_calib = get_orig_conc(innate, step, prms, t, norm, player, bound, h)
    dic, df_rmse = get_model(step, prms, bounds, t, df_calib, norm, player, bound, h)
    #dic, df_calib = get_RMSE(step, dic, df_calib,  player)
    plot_params(dic,df_calib)
    #plot_RMSE(df_rmse)

main('bIAP')