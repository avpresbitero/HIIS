import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats, linalg
from scipy.stats import rankdata

import APSim_Model as hiis
import Parameters as params


def get_calibrated_parameters(innate, step, bound, norm, h):
    innate.h = h

    df_params = pd.read_csv('result/optimize/params/' + h + '_' + bound + norm + step + '_best_params.txt', sep='\t')
    beta_CHNA = df_params['beta_CHNA'].values[0]
    beta_CHMA = df_params['beta_CHMA'].values[0]
    theta_ACH = df_params['theta_ACH'].values[0]
    beta_MANDA = df_params['beta_MANDA'].values[0]
    Pmin_APE = df_params['Pmin_APE'].values[0]
    Pmax_APE = df_params['Pmax_APE'].values[0]
    rflush = df_params['rflush'].values[0]
    rinduce = df_params['rinduce'].values[0]
    gauss_min = df_params['gauss_min'].values[0]
    lamb_ITMNDN = df_params['lamb_ITMNDN'].values[0]
    alpha_ITMNDN = df_params['alpha_ITMNDN'].values[0]
    alpha_ITMD = df_params['alpha_ITMD'].values[0]

    return [beta_CHNA, beta_CHMA, theta_ACH, beta_MANDA, Pmin_APE, Pmax_APE, rflush, gauss_min, rinduce,
            lamb_ITMNDN, alpha_ITMNDN, alpha_ITMD]

def get_parameter_names():
    return ['beta_CHNA', 'beta_CHMA', 'theta_ACH', 'beta_MANDA', 'Pmin_APE', 'Pmax_APE', 'rdistress', 'gauss_min',
            'rinduce', 'lamb_ITMNDN', 'alpha_ITMNDN', 'alpha_ITMD']

def do_LHC_PRCC(innate, prms, kind, p0):
    list_length = 100000
    parameter_names = get_parameter_names()
    bounds = prms.get_bounds()
    latin_hypercube, ranked_latin_hypercube = generate_latin_hyper_cube(parameter_names, list_length, p0, kind, bounds)
    output_matrix, ranked_output_matrix = generate_output_matrix(innate, prms, latin_hypercube)
    # generates the final table, each row corresponds to a different parameter
    lhc_prcc, lhc_prcc_pval = LHC_PRCC(ranked_latin_hypercube, ranked_output_matrix)
    lhc_prcc.to_csv('result/optimize/sensitivity/LHS_PRCC.csv', sep='\t', encoding='utf-8')
    lhc_prcc_pval.to_csv('result/optimize/sensitivity/LHS_PRCC_pval.csv', sep='\t', encoding='utf-8')

def change_time(lst, mult):
    time = []
    for i in lst:
        if i == 0:
            time.append(i)
        else:
            time.append(i * mult / 100.)
    return time


def plot():
    lgs = 13
    sns.set(style="white")
    lhc_prcc = pd.read_csv('result/optimize/sensitivity/LHS_PRCC.csv', sep='\t', encoding='utf-8')
    lhc_prcc_pval = pd.read_csv('result/optimize/sensitivity/LHS_PRCC_pval.csv', sep='\t', encoding='utf-8')
    nms=['beta_CHNA', 'beta_CHMA', 'theta_ACH', 'alpha_ACHMA', 'beta_MANDA', 'Pmin_APE', 'Pmax_APE', 'rflush',
         'gauss_min', 'rinduce', 'lamb_ITMNDN', 'lamb_MANDN', 'alpha_ITMNDN', 'alpha_ITMD']
    markers = ['v', '<', '>', '^', '*', '+', '2', '3', '4', 'o', '.', 'h','p', 'H']
    for i in range(len(nms) - 1):
        marker = markers[i]
        sns.plt.figure(0)
        row_prcc = lhc_prcc.iloc[i]
        vals_prcc = row_prcc.tolist()
        time_prcc = change_time(range(len(vals_prcc[1:])), 36)
        y_prcc = vals_prcc[1:]
        sns.plt.plot(time_prcc, y_prcc, marker=marker, ms=10, linewidth=1.5, alpha=.5, label=nms[i])
        sns.plt.ylabel("PRCC", fontsize=20)
        sns.plt.legend(loc='best', prop={'size': lgs}, frameon=True)
        # sns.plt.ylim((-0.1, 0.2))
        sns.plt.xlim((0, 36))
        sns.plt.tick_params(labelsize=20)
        sns.plt.xlabel("Hours After Operation", fontsize=20)
        sns.plt.savefig('result/plots/sensitivity/SA.png', dpi=300)

        sns.plt.figure(1)
        row_pval = lhc_prcc_pval.iloc[i]
        vals_pval = row_pval.tolist()
        time_pval = change_time(range(len(vals_pval[1:])), 36)
        y_pval = vals_pval[1:]
        sns.plt.plot(time_pval, y_pval, marker=marker, linewidth=1.5, alpha=.5, label=nms[i])
        sns.plt.ylabel("p-values", fontsize=20)
        sns.plt.legend(loc='best', prop={'size': lgs}, frameon=True)
        sns.plt.ylim((0, 0.05))
        sns.plt.xlim((0, 36))
        sns.plt.tick_params(labelsize=20)
        sns.plt.xlabel("Hours After Operation", fontsize=20)
        sns.plt.savefig('result/plots/sensitivity/SA_pval.png', dpi=300)
    sns.plt.show()


def generate_parameter_list(parameter_value, max_bound, list_length, kind):
    """
    :param parameter_value: parameter value
    :param max_bound: maximum boundary
    :param list_length: number of values to test sensitivity on
    :param kind: type of distribution to use
    :return: a list of varied parameter values with length list_length
    """

    if kind == 'normal':
        mu_norm = parameter_value / max_bound
        s_norm = np.random.normal(mu_norm, mu_norm / 3, list_length)
        s_list = [s*max_bound for s in s_norm]
        return s_list
    elif kind == 'uniform':
        uniform_list = np.linspace(0, max_bound, list_length)
        return uniform_list[uniform_list != 0]

def rank_data(data):
    return rankdata(data)


def use_alkaline_phosphatase_model(innate, prms, p0):
    p, w, pred_fle = prms.get_params(innate, p0)
    t = prms.get_t(innate)
    innate.case = 6
    innate.h = 'h2'
    w0 = innate.get_init(w)
    t, wsol = innate.solve(p, w0, pred_fle, t, False)
    APE_blood = []
    for t1, w1 in zip(t, wsol):
        APE_blood.append(w1[1])
    return np.array(APE_blood)


# table for LHC, first, a latin hypercube is generated, then it is ranked
def generate_latin_hyper_cube(names, list_length, parameters, kind, bounds):
    print ('Generating Latin Hypercube...')
    latin_hypercube = pd.DataFrame()
    ranked_latin_hypercube = pd.DataFrame()
    for i in range(len(parameters)):
        parameter_value = parameters[i]
        parameter_name = names[i]
        max_bound = bounds[i][1]
        generated_parameters = generate_parameter_list(parameter_value, max_bound, list_length, kind)
        np.random.shuffle(generated_parameters)
        latin_hypercube[parameter_name] = generated_parameters
        ranked_latin_hypercube[parameter_name] = rank_data(generated_parameters)
    return latin_hypercube, ranked_latin_hypercube


# table for output matrix, we focus endogenous AP concentrations
# best to print this as csv
def generate_output_matrix(innate, prms, lhc):
    print ('Generating Output Matrix...')
    output_matrix = pd.DataFrame()
    ranked_output_matrix = pd.DataFrame()
    total = len(lhc.index)
    count = 0
    for row in lhc.iterrows():
        count += 1
        index, data = row
        ap_e = use_alkaline_phosphatase_model(innate, prms, data.tolist())
        output_matrix[index] = ap_e
        ranked_output_matrix[index] = rank_data(ap_e)
        print (total, count)
    return output_matrix, ranked_output_matrix


# constructs the final table for LHC-PRCC
def LHC_PRCC(ranked_latin_hypercube, ranked_output_matrix):
    print ('Doing Latin Hypercube - Partial Rank Correlation Coefficient...')
    lhc_prcc = pd.DataFrame()
    lhc_prcc_pval = pd.DataFrame()
    for row in ranked_output_matrix.iterrows():
        df = ranked_latin_hypercube.copy(deep=True)
        index, data = row
        if index % 10 == 0:
            print(index)
            df[index] = data.tolist()
            p_corr, p_pval = partial_corr(df.values.tolist())
            pd_corr = pd.DataFrame(p_corr)
            pd_pval = pd.DataFrame(p_pval)
            # selects the last column in the dataframe, which corresponds to the partial coeff of output vs params
            lhc_prcc[index] = pd_corr.iloc[:, -1].values.tolist()
            lhc_prcc_pval[index] = pd_pval.iloc[:, -1].values.tolist()
    return lhc_prcc, lhc_prcc_pval


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.

    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable

    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    P_pval = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            # print(C[:, idx])
            # print(C[:, j])
            # print(C[:, i])

            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            # print(beta_i)
            # print(C[:, j])
            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            pval = stats.pearsonr(res_i, res_j)[1]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
            P_pval[i, j] = pval
            P_pval[j, i] = pval
    return P_corr, P_pval


def main():
    innate = hiis.Innate()
    prms = params.Parameters()

    step = 'bIAP'
    bound = 'bound_'
    norm = 'norm_'
    h = 'h2'
    kind = 'uniform'
    # kind = 'normal'

    # p0 = get_calibrated_parameters(innate, step, bound, norm, h)
    # do_LHC_PRCC(innate, prms, kind, p0)
    # plot()
    # C = [[1, 2, 3, 5],
    #      [4, 5, 6, 5],
    #      [7, 8, 9, 5]]
    # partial_corr(C)
    plot()

if __name__ == '__main__':
    main()