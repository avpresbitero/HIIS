import pandas as pd

import APSim_Model as hiis
import Parameters as params
import seaborn as sns
import numpy as np

from sklearn import preprocessing
from matplotlib import style


def get_RMSE(collect, method, step, norm, h):
    """
    :param collect:
    :param method:
    :param step:
    :return: dataframe of RMSEs
    """
    rmse_fle = 'result/optimize/RMSE/' + collect + '/' + h + '/'+ method + '/' + norm + method + '_' + step + '_' + 'RMSE.txt'
    df_rmse = pd.read_csv(rmse_fle, delimiter='\t')
    return df_rmse


def get_params(collect, method, step, norm, h):
    calib_fle = 'result/optimize/calibrated/' + collect + '/' + h + '/'+ method + '/' + norm + method + '_' + step + '.txt'
    df_params = pd.read_csv(calib_fle, delimiter='\t')
    #bst_df = df.iloc[[index]]
    #param_names = list(bst_df.columns)
    #param_values = list(bst_df.values.tolist()[0])
    return df_params


def find_best_params(df_fin):
    """
    :param df_top_RMSEs:
    :param df_params:
    :return: dataframe of best calibrated values
    """
    df_fin = df_fin.ix[df_fin['mean_RMSE'] >= 0]
    params = df_fin.ix[df_fin['mean_RMSE'].idxmin()]
    values = params.values
    # values = values[:-1]
    trial, rmse_CH, rmse_ACH, rmse_AP, mean_RMSE, method, beta_CHNA, beta_CHMA, theta_ACH, beta_MANDA, Pmin_APE, \
    Pmax_APE, rflush, gauss_min, rinduce_peak, lamb_ITMNDN, alpha_ITMNDN, alpha_ITMD, r_NDN, r_ITM, r_ITMpeak, r_AP, \
    Keq_CH, rinduce, lamb_DG, lamb_MANDN, mu_NDA, lamb_MANDA, r_Nhomeo, Pmax_NR = values

    print('METHOD : ', method)
    print('FINAL CALIBRATED VALUES FOR TRIAL ', trial, ' :')
    print('beta_CHNA = ', abs(beta_CHNA))
    print('beta_CHMA =', abs(beta_CHMA))
    print('theta_ACH =', abs(theta_ACH))
    print('beta_MANDA =', abs(beta_MANDA))
    print('Pmin_APE =', abs(Pmin_APE))
    print('Pmax_APE =', abs(Pmax_APE))
    print('rflush =', abs(rflush))
    print('gauss_min =', abs(gauss_min))
    print('rinduce_peak =', abs(rinduce_peak))
    print('lamb_ITMNDN = ', abs(lamb_ITMNDN))
    print('alpha_ITMNDN = ', abs(alpha_ITMNDN))
    print('alpha_ITMD = ', abs(alpha_ITMD))
    print('r_NDN = ', abs(r_NDN))
    print('r_ITM = ', abs(r_ITM))
    print('r_ITMpeak = ', abs(r_ITMpeak))
    print('r_AP = ', abs(r_AP))
    print('Keq_CH = ', abs(Keq_CH))
    print('rinduce = ', abs(rinduce))
    print('lamb_DG = ', abs(lamb_DG))
    print('lamb_MANDN = ', abs(lamb_MANDN))
    print('mu_NDA = ', abs(mu_NDA))
    print('lamb_MANDA = ', abs(lamb_MANDA))
    print('r_Nhomeo = ', abs(r_Nhomeo))
    print('Pmax_NR = ', abs(Pmax_NR))
    return values


def join_dfs(frames):
    return pd.concat(frames)


def get_header():
    header = ['N_R', 'AP_Eblood', 'AP_Etissue', 'AP_Eliver', 'AP_Sblood', 'AP_Stissue', 'ITMblood', 'ITMtissue',
              'D', 'M_R', 'M_A', 'CH', 'N_A', 'ND_A', 'G', 'ACH', 'ND_N', 'N_B']
    return header


def get_best_params(methods, collect, step, bound, norm, h):
    frames = []
    for method in methods:
        df_rmse = get_RMSE(collect, method, step, norm, h)
        df_rmse['method'] = [method] * df_rmse.shape[0]
        df_params = get_params(collect, method, step, norm, h)
        df_join = pd.merge(df_rmse, df_params, on='Trial')
        frames.append(df_join)
    for i in range(len(frames)):
        if i == 0:
            df_fin = frames[i]
        else:
            df_fin = df_fin.append(frames[i], ignore_index=True)


    values = find_best_params(df_fin)
    trial, rmse_CH, rmse_ACH, rmse_AP, mean_RMSE, method,  beta_CHNA, beta_CHMA, theta_ACH, beta_MANDA, Pmin_APE, \
    Pmax_APE, rflush, gauss_min, rinduce_peak, lamb_ITMNDN, alpha_ITMNDN, alpha_ITMD, r_NDN, r_ITM, r_ITMpeak, r_AP, \
    Keq_CH, rinduce, lamb_DG, lamb_MANDN, mu_NDA, lamb_MANDA, r_Nhomeo, Pmax_NR = values

    with open('result/optimize/params/' + h + '_' + bound + norm + step +'_best_params.txt', 'w') as file:
        file.write(
            'beta_CHNA' + '\t' + 'beta_CHMA' + '\t' + 'theta_ACH' + '\t' + 'beta_MANDA' + '\t' + 'Pmin_APE' + '\t'
            + 'Pmax_APE' + '\t' + 'rflush' + '\t' + 'gauss_min' + '\t' + 'rinduce_peak' + '\t' + 'lamb_ITMNDN' + '\t'
            + 'alpha_ITMNDN' + '\t' + 'alpha_ITMD' + '\t' + 'r_NDN' + '\t' + 'r_ITM' + '\t' + 'r_ITMpeak' + '\t'
            + 'r_AP' + '\t' + 'Keq_CH' + '\t' + 'rinduce' + '\t' + 'lamb_DG' + '\t' + 'lamb_MANDN' + '\t'
            + 'mu_NDA' + '\t' + 'lamb_MANDA' + '\t' + 'r_Nhomeo' + '\t' + 'Pmax_NR' + '\n' +
            str(abs(beta_CHNA)) + '\t' + str(abs(beta_CHMA)) + '\t' +
            str(abs(theta_ACH)) + '\t' + str(abs(beta_MANDA)) + '\t' + str(abs(Pmin_APE)) + '\t' +
            str(abs(Pmax_APE)) + '\t' + str(abs(rflush)) + '\t' + str(abs(gauss_min)) + '\t' +
            str(abs(rinduce_peak)) + '\t' + str(abs(lamb_ITMNDN)) + '\t' + str(abs(alpha_ITMNDN)) + '\t' +
            str(abs(alpha_ITMD)) + '\t' + str(abs(r_NDN)) + '\t' + str(abs(r_ITM)) + '\t' +
            str(abs(r_ITMpeak)) + '\t' + str(abs(r_AP)) + '\t' + str(abs(Keq_CH)) + '\t' +
            str(abs(rinduce)) + '\t' + str(abs(lamb_DG)) + '\t' + str(abs(lamb_MANDN)) + '\t' +
            str(abs(mu_NDA)) + '\t' + str(abs(lamb_MANDA)) + '\t' + str(abs(r_Nhomeo)) + '\t' + str(abs(Pmax_NR)) + '\n')
    return values


def plot_RMSE(collect, method, step, trial, norm, h):
    lw = 2
    ls = 20
    fs = 20

    sns.set(style="white")
    rmse_fle = 'result/optimize/RMSE/' + collect + '/' + h + '/'+ method + '/' + norm + 'conv_' + method + '_' + step + '_' + 'RMSE.txt'
    df_rmse = pd.read_csv(rmse_fle, delimiter='\t')
    df_rmse = df_rmse[df_rmse['Trial'] == trial]
    del df_rmse['Trial']
    print(len(df_rmse))
    df_rmse.plot(linewidth=lw, use_index=False)
    sns.plt.xlabel("Time Step", fontsize=fs)
    sns.plt.ylabel("Root Mean Square Error", fontsize=fs)
    sns.plt.title("Best Parameter Optimization Using " + method, fontsize=fs)
    sns.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': fs}, frameon=True)
    sns.plt.tick_params(labelsize=ls)
    sns.plt.hold(True)
    sns.plt.savefig('result/plots/calibrated/' + method + '_RMSE.png', format='png', dpi=500, bbox_inches='tight')
    sns.plt.show()


def plot_params(collect, method, step, trial, norm, h):
    lw = 2
    ls = 20
    fs = 20

    sns.set(style="white")
    sns.set_palette("Paired", n_colors = 10)
    #style.use('fivethirtyeight')
    param_fle = 'result/optimize/calibrated/' + collect + '/' + h + '/' + method + '/' + norm +'conv_' + method + '_' + step + '.txt'
    df_param = pd.read_csv(param_fle, delimiter='\t')
    df_param = df_param[df_param['Trial'] == trial]
    del df_param['Trial']
    #print (df_param)
    del df_param['Step']

    x = df_param.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_param = pd.DataFrame(x_scaled, columns=df_param.columns)

    # manual calculation of scaling the list, for comparison
    # header = df_param.columns.values
    #for head in header:
    #    df_param[head] = abs(df_param[head]-min(df_param[head]))/(max(df_param[head])-min(df_param[head]))

    df_param.plot(linewidth=lw, use_index=False)
    sns.plt.ylim((0,1))
    sns.plt.xlabel("Time Step", fontsize=fs)
    sns.plt.ylabel("Normalized Calibrated Parameter", fontsize=fs)
    sns.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': fs}, frameon=True)
    sns.plt.title("Best Parameter Optimization Using " + method, fontsize=fs)
    sns.plt.tick_params(labelsize=ls)
    sns.plt.hold(True)
    sns.plt.savefig('result/plots/calibrated/' + method + '_params.png', format='png', dpi=500, bbox_inches='tight')
    sns.plt.show()

def main(step, hypothesis, collect):
    innate = hiis.Innate()
    prms = params.Parameters()

    # collect = 'list'
    # collect = 'distribution'


    bound = 'bound_'
    norm = 'norm_'
    # norm = ''

    methods = ['TNC','SLSQP','L-BFGS-B'] # bound methods
    methods = ['TNC', 'SLSQP', 'L-BFGS-B']# bound methods

    if step == 'Placebo':
        innate.case = 5
    elif step == 'bIAP':
        innate.case = 6

    trial, rmse_CH, rmse_ACH, rmse_AP, mean_RMSE, method, beta_CHNA, beta_CHMA, theta_ACH, beta_MANDA, Pmin_APE, \
    Pmax_APE, rflush, gauss_min, rinduce_peak, lamb_ITMNDN, alpha_ITMNDN, alpha_ITMD, r_NDN, r_ITM, r_ITMpeak, r_AP,\
    Keq_CH, rinduce, lamb_DG, lamb_MANDN, mu_NDA, lamb_MANDA, r_Nhomeo, Pmax_NR = get_best_params\
        (methods, collect, step, bound, norm, hypothesis)

    #method = 'TNC'
    #method = 'Nelder-Mead'
    #plot_RMSE(collect, method,step, trial, norm, h)
    #plot_params(collect, method, step, trial, norm, h)

# main('bIAP', hypothesis='h4', collect='list')
#main('both')
# main('Placebo')

if __name__ == '__main__':
    destination_folder = 'C:/Users/Alva/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                         'AP Simulator Model/Frontiers in Immunology/Special Issue/Revisions/results'
    df = pd.read_pickle(destination_folder + "./0_optimized")
