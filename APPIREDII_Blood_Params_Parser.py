import pandas as pd
import numpy as np

from statsmodels import robust as r


def read_data(filename):
    df = pd.read_csv(filename, sep='\t')
    return df


def concatenate_dataframes(df_list):
    df_total = pd.concat(df_list, ignore_index=True)
    return df_total


def header_to_lowercase(df):
    df.columns = map(str.lower, df.columns)
    return df


def get_concentration_dataframe(df, parameter_name):
    parameter_name_lc = parameter_name.lower()
    df_concentration = df[['patno', 'datum', parameter_name_lc, 'treatment']]
    return df_concentration


def add_treatment(df, df_treatment):
    """
    :param df: a dataframe that need adding of treatment column
    :param trt_fle: treatment file
    :return: Adds a column referring to type of patient treatment
    """

    cols = pd.Series(df_treatment.columns)
    for dup in df_treatment.columns.get_duplicates():
        cols[df_treatment.columns.get_loc(dup)] = \
            [dup + str(d_idx) if d_idx != 0 else dup for d_idx in range(df_treatment.columns.get_loc(dup).sum())]
    df_treatment.columns = cols
    df_treatment['pat.id.'].astype(str)

    dic = df_treatment.set_index('pat.id.')['treatment'].to_dict()
    df['patno'] = df['patno'].astype(str)
    df['treatment'] = df['patno'].map(dic)
    return df


def get_unique_patient_ids(df):
    patient_ids = df['patno'].unique()
    return patient_ids


def replace_datestring_to_datetime(df, hospital):
    if hospital == 'german':
        df['datum'] = pd.to_datetime(df['datum'], format='%d.%b.%Y %I:%M:%S %p')
    elif hospital == 'dutch':
        df['datum'] = pd.to_datetime(df['datum'], format='%d.%m.%Y %H:%M')
    return df


def capture_operation_date(df):
    count = 0
    for index, row in df.iterrows():
        if count == 1:
            capture_date = np.datetime64(row['datum'])
            return capture_date
        count += 1


def set_numerical_time_dataframe(df, parameter_name):
    patients_dic = {'patno': [], 'treatment': [], parameter_name: [], 'time':[]}
    ts_group = df.groupby('patno')
    for patient_id in ts_group.groups.keys():
        patient_ts = list(ts_group.get_group(patient_id)['datum'])
        operation_date = patient_ts[1]
        patients_dic['time'] += [time.total_seconds()/3600. for time in pd.to_datetime(patient_ts)
                                - pd.to_datetime(operation_date)]
        patients_dic['patno'] += list(ts_group.get_group(patient_id)['patno'])
        patients_dic[parameter_name] += list(ts_group.get_group(patient_id)[parameter_name])
        patients_dic['treatment'] += list(ts_group.get_group(patient_id)['treatment'])
    return pd.DataFrame(patients_dic)


def get_median(df, parameter_name, do_binning=True):
    original_df = df.copy(deep=True)
    if do_binning:
        df = get_bin(df=df,
                        parameter_name=parameter_name,
                        do_mean=True)
    treatment_unique = df['treatment'].unique()
    df_median = pd.DataFrame()
    treatment_list = []
    median_list = []
    time_list = []
    mad_list = []
    for treatment in treatment_unique:
        df_treatment = df[df['treatment'] == treatment]
        ts_group = df_treatment.groupby('time')
        for time in ts_group.groups.keys():
            patient_parameter_ts = list(ts_group.get_group(time)[parameter_name])
            patient_parameter_ts = [float(i) for i in patient_parameter_ts if i != '.']
            if len(patient_parameter_ts) != 0:
                median_list.append(np.median(patient_parameter_ts))
                mad_list.append(r.mad(patient_parameter_ts))
                time_list.append(time)
                treatment_list.append(treatment)
            # print(time, treatment, np.median(patient_parameter_ts), patient_parameter_ts)
    df_median['treatment'] = treatment_list
    df_median['median'] = median_list
    df_median['time'] = time_list
    df_median['mad'] = mad_list
    df_median = df_median.sort_values(['treatment', 'time'])
    return original_df, df_median


def get_median_restrict(df, parameter_name, treatment, patients, do_binning=True):
    df_treatment = df[df['treatment'] == treatment]
    df_treatment = df_treatment[df_treatment['patno'].astype(str).isin(patients)]
    original_df = df.copy(deep=True)
    if do_binning:
        df = get_bin(df=df,
                        parameter_name=parameter_name,
                        do_mean=True)
    df_median = pd.DataFrame()
    treatment_list = []
    median_list = []
    time_list = []
    mad_list = []
    ts_group = df_treatment.groupby('time')
    for time in ts_group.groups.keys():
        patient_parameter_ts = list(ts_group.get_group(time)[parameter_name])
        patient_parameter_ts = [float(i) for i in patient_parameter_ts if i != '.']
        if len(patient_parameter_ts) != 0:
            median_list.append(np.median(patient_parameter_ts))
            mad_list.append(r.mad(patient_parameter_ts))
            time_list.append(time)
            treatment_list.append(treatment)
        # print(time, treatment, np.median(patient_parameter_ts), patient_parameter_ts)
    df_median['treatment'] = treatment_list
    df_median['median'] = median_list
    df_median['time'] = time_list
    df_median['mad'] = mad_list
    df_median = df_median.sort_values(['treatment', 'time'])
    return original_df, df_median


def get_bin(df, parameter_name, do_mean=True):
    df_binned = pd.DataFrame()
    treatment_unique = df['treatment'].unique()
    treatment_list = []
    param_list = []
    time_list = []
    for treatment in treatment_unique:
        df_treatment = df[df['treatment'] == treatment]
        time_list_treatment = df_treatment['time'].tolist()
        rounded_time_list = df_treatment['time'].round()
        min_round, max_round = min(rounded_time_list), max(rounded_time_list)
        for time_bin in np.linspace(min_round, max_round, 2*(max_round-min_round)+1):
            bin_time_list = [time for time in time_list_treatment if time >= time_bin - 0.25 and time < time_bin + 0.25]
            bin_concentration_list = df_treatment[df_treatment['time'].isin(bin_time_list)][parameter_name].dropna().tolist()
            for binned in bin_concentration_list:
                treatment_list.append(treatment)
                param_list.append(binned)
                time_list.append(time_bin)
    df_binned['treatment'] = treatment_list
    df_binned[parameter_name] = param_list
    df_binned['time'] = time_list
    return df_binned


def get_data(german_hospital_file, dutch_hospital_file, treatment_file, parameter_name, do_binning, restrict,
             treatment, patients):
    df_german = read_data(filename=german_hospital_file)
    df_dutch = read_data(filename=dutch_hospital_file)
    df_treatment = read_data(filename=treatment_file)

    df_german_lc = header_to_lowercase(df=df_german)
    df_dutch_lc = header_to_lowercase(df=df_dutch)
    df_treatment_lc = header_to_lowercase(df=df_treatment)

    df_german_lc = replace_datestring_to_datetime(df=df_german_lc,
                                                  hospital='german')
    df_dutch_lc = replace_datestring_to_datetime(df=df_dutch_lc,
                                                 hospital='dutch')

    df_total = concatenate_dataframes(df_list=[df_german_lc, df_dutch_lc])

    df_total_treatment = add_treatment(df=df_total,
                                       df_treatment=df_treatment_lc)

    df_concentration = get_concentration_dataframe(df=df_total_treatment,
                                                   parameter_name=parameter_name)

    df_timeseries = set_numerical_time_dataframe(df=df_concentration,
                                                 parameter_name=parameter_name.lower())

    if restrict:
        df_timeseries, df_median = get_median_restrict(df=df_timeseries, parameter_name=parameter_name.lower(),
                                               treatment=treatment, patients=patients, do_binning=do_binning)
    else:
        df_timeseries, df_median = get_median(df=df_timeseries,
                                              parameter_name=parameter_name.lower(),
                                              do_binning=do_binning)
    df_timeseries = df_timeseries[df_timeseries[parameter_name.lower()] != '.']
    df_timeseries[parameter_name.lower()] = [float(i) for i in df_timeseries[parameter_name.lower()]]
    return df_timeseries, df_median


if __name__ == "__main__":
    german_hospital_file = 'data/APPIREDII/Hospital2.txt'
    dutch_hospital_file = 'data/APPIREDII/Hospital1.txt'
    treatment_file = 'data/APPIREDII/segregation.txt'

    parameter_name = 'AlkFosf'
    restrict = True
    treatment = 'Placebo'
    patients = ['16', '51', '36', '26', '59']
    get_data(german_hospital_file=german_hospital_file,
             dutch_hospital_file=dutch_hospital_file,
             treatment_file=treatment_file,
             parameter_name=parameter_name,
             do_binning=True,
             restrict=restrict,
             treatment=treatment,
             patients=patients)