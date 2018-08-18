import pandas as pd
import seaborn as sns
import numpy as np

from statsmodels import robust
from scipy.interpolate import interp1d


def read_data(filename):
    df = pd.read_csv(filename, sep='\t')
    return df


def header_to_lowercase(df):
    df.columns = map(str.lower, df.columns)
    return df


def rep_sample_to_time(df, smpl_file, colhd):
    df_smpl = read_data(smpl_file)
    df_smpl = header_to_lowercase(df=df_smpl)
    dic = df_smpl.set_index('sample')['blood sampling'].to_dict()
    df[colhd] = df[colhd].map(dic)
    return df


def add_treatment(df, trt_fle):
    """
    :param df: a dataframe that need adding of treatment column
    :param trt_fle: treatment file
    :return: Adds a column referring to type of patient treatment
    """
    df_trt = read_data(trt_fle)
    df_trt = header_to_lowercase(df=df_trt)
    cols = pd.Series(df_trt.columns)
    for dup in df_trt.columns.get_duplicates():
        cols[df_trt.columns.get_loc(dup)] = [dup + str(d_idx) if d_idx != 0 else dup for d_idx in range(df_trt.columns.get_loc(dup).sum())]
    df_trt.columns = cols
    df_trt['pat.id.'].astype(str)

    dic = df_trt.set_index('pat.id.')['treatment'].to_dict()
    df['patno'] = df['patno'].astype(str)
    df['treatment'] = df['patno'].map(dic)
    return df


def get_data(ct_fle, smpl_fle, trt_fle, colhd):
    df = read_data(ct_fle)
    df = header_to_lowercase(df)
    df = change_unit(df)
    df = rep_sample_to_time(df, smpl_fle, colhd)
    df = add_treatment(df, trt_fle)
    return df


def get_header(df, strt, xcpt):
    return df.columns[strt:xcpt]


def get_med_df(cyto_nm, smpl_lst, df_cyto, trt):
    med = []
    smpls = []
    cyto_lst = []
    trt_lst = []
    total = []
    mad = []
    df_sp_cyto = pd.DataFrame()
    for smpl in smpl_lst:
        tbc = df_cyto.loc[df_cyto['sample'] == smpl, cyto_nm].values
        if len(tbc) > 0:
            med.append(np.median(tbc))
            smpls.append(smpl)
            cyto_lst.append(cyto_nm)
            trt_lst.append(trt)
            total.append(tbc)
            mad.append(robust.mad(tbc))
    df_sp_cyto['time'] = smpls
    df_sp_cyto['median'] = med
    df_sp_cyto['mad'] = mad
    df_sp_cyto['cytokine'] = cyto_lst
    df_sp_cyto['treatment'] = trt_lst
    return df_sp_cyto, total


def get_median(ct_fle, smpl_fle, trt_fle, colhd):
    df_cyto = read_data(ct_fle)
    df_cyto = header_to_lowercase(df=df_cyto)
    df_cyto = change_unit(df_cyto)
    df_cyto = add_treatment(df_cyto, trt_fle)

    df_cyto['treatment'] = df_cyto['treatment'].str.lower()
    trt_list = df_cyto['treatment'].unique()

    smpl_df = read_data(smpl_fle)
    smpl_df = header_to_lowercase(df=smpl_df)

    smpl_lst = smpl_df['sample']

    cyto_lst = get_header(df_cyto, 2, -1)
    dfs_cyto = []
    total_dic = {}
    for trt in trt_list:
        df_cyto_trt = df_cyto[df_cyto['treatment'] == trt]
        total_dic[trt] = {}
        for cyto_rename in cyto_lst:
            df_sp_cyto, total = get_med_df(cyto_rename, smpl_lst, df_cyto_trt, trt)
            df_sp_cyto = rep_sample_to_time(df_sp_cyto, smpl_fle, colhd)
            dfs_cyto.append(df_sp_cyto)
            total_dic[trt][cyto_rename] = total
        total_dic[trt]['time'] = smpl_lst
    df_concat = pd.concat(dfs_cyto, ignore_index=True)
    return df_concat, total_dic


def get_median_restrict(ct_fle, smpl_fle, trt_fle, colhd, treatment, patients):
    df_cyto = read_data(ct_fle)
    df_cyto = header_to_lowercase(df=df_cyto)
    df_cyto = change_unit(df_cyto)
    df_cyto = add_treatment(df_cyto, trt_fle)
    df_cyto = df_cyto[df_cyto['treatment'] == treatment]
    df_cyto = df_cyto[df_cyto['patno'].astype(str).isin(patients)]
    df_cyto['treatment'] = df_cyto['treatment'].str.lower()

    smpl_df = read_data(smpl_fle)
    smpl_df = header_to_lowercase(df=smpl_df)

    smpl_lst = smpl_df['sample']

    cyto_lst = get_header(df_cyto, 2, -1)
    dfs_cyto = []
    total_dic = {}
    for cyto_rename in cyto_lst:
        df_sp_cyto, total = get_med_df(cyto_rename, smpl_lst, df_cyto, treatment)
        df_sp_cyto = rep_sample_to_time(df_sp_cyto, smpl_fle, colhd)
        dfs_cyto.append(df_sp_cyto)
        total_dic[cyto_rename] = total
        total_dic['time'] = smpl_lst
    df_concat = pd.concat(dfs_cyto, ignore_index=True)
    return df_concat, total_dic


def loop_fig(fignum):
    return fignum + 1


def change_unit(df):
    cyto_lst = get_header(df, 2, -1)
    for cyto in cyto_lst:
        if cyto == 'il6':
            mw = 21*10**3
        elif cyto == 'il10':
            mw = 21 * 10 ** 3
        elif cyto == 'il8':
            mw = 11 * 10 ** 3
        mult = ((1*10**-12)*(6.02*10**23)/((1*10**-3)*mw*10**6))
        df.loc[:, cyto] *= mult
    return df


def reverse_unit(lst, cyto):
    if cyto == 'il6':
        mw = 21 * 10 ** 3
    elif cyto == 'il10':
        mw = 21 * 10 ** 3
    elif cyto == 'il8':
        mw = 11 * 10 ** 3
    mult = ((1 * 10 ** -12) * (6.02 * 10 ** 23) / ((1 * 10 ** -3) * mw * 10 ** 6))
    return [i/mult for i in lst]


def plot(df, cyto_lst, trt_fle, smpl_fle):
    sns.set(style="white")
    n = 0
    df = add_treatment(df, trt_fle)
    df = rep_sample_to_time(df, smpl_fle, 'sample')
    df = df[df['treatment'] == 'Placebo']
    for cyto in cyto_lst:
        n = loop_fig(n)
        sns.plt.figure(n)
        g = sns.pointplot(x="sample", y=cyto, hue = "patno", data=df, linestyles="-",palette="muted")
        # g.set_yscale('log',basex=2)
        sns.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, frameon=True)
        sns.plt.hold(True)
        sns.plt.tight_layout()
        sns.plt.ylabel(cyto, fontsize=20)
        sns.plt.xlabel('Hours After Surgery', fontsize=20)
        # self.labelLines(sns.plt.gca().get_lines(), zorder=2.5)
        sns.plt.savefig('data/plots/' + cyto + '.png', format='png', dpi=500, bbox_inches='tight')
    sns.plt.show()


def get_special_patients(ct_fle, trt_fle, smpl_fle, peak):
    sns.set(style="white")
    # cytos = ['tnfa', 'il6', 'il8']
    cytos = ['tnfa']
    fs = 20

    df_cyto = read_data(ct_fle)
    df_cyto = header_to_lowercase(df=df_cyto)
    df_cyto = change_unit(df_cyto)
    df_cyto = add_treatment(df_cyto, trt_fle)
    df_placebo = df_cyto[df_cyto['treatment'] == 'Placebo']
    df_sp_cyto = rep_sample_to_time(df_placebo, smpl_fle, 'sample')
    ts_group = df_sp_cyto.groupby('patno')

    score = {}
    special_patients = {}
    for i in range(50):
        for cyto in cytos:
            for patno in ts_group.groups.keys():
                conc = list(ts_group.get_group(patno)[cyto])
                time = list(ts_group.get_group(patno)['sample'])
                if peak:
                    conc = [conc[i] for i in range(len(conc)) if time[i] <= 24]
                    special_patients[patno] = max(conc)
                else:
                    f = interp1d(time, conc)
                    xnew = np.linspace(1.3, 12, num=50, endpoint=True)
                    special_patients[patno] = sum(f(xnew))
            sorted_patients = sorted(special_patients, key=special_patients.get, reverse=True)
            for i in range(len(sorted_patients)):
                if i <= 5:
                    if sorted_patients[i] not in score.keys():
                        score[sorted_patients[i]] = 0
                    score[sorted_patients[i]] += 1
    # print(sorted(score, key=score.get, reverse=True))
    df_sp = pd.DataFrame(score, index=[0])
    df_sp.plot.bar()
    sns.plt.show()


if __name__ == "__main__":
    cyto_fle = 'data/APPIREDII/cytokines.txt'
    sample_fle = 'data/APPIREDII/sample.txt'
    trt_fle = 'data/APPIREDII/segregation.txt'
    colhd = 'time'
    treatment = 'Placebo'
    df = read_data(cyto_fle)
    df_median, cyto_dic = get_median(cyto_fle, sample_fle, trt_fle, 'time')
    df_IL10 = df_median[df_median['cytokine'] == 'il10']
    df_IL6 = df_median[df_median['cytokine'] == 'il6']

