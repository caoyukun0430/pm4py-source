import os
import functools
import operator
import time
from scipy.spatial.distance import pdist
import pandas as pd
import numpy as np
import act_dist_calc
import suc_dist_calc
from IPython.display import display
from collections import Counter
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.log import EventLog, Trace, EventStream
from pm4py.util.constants import PARAMETER_CONSTANT_ATTRIBUTE_KEY, PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util import constants
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.log import case_statistics
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.adapters.pandas import csv_import_adapter
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory


def apply_trace_attributes(log, list_of_values, parameters=None):
    """
    Filter log by keeping only traces that has/has not certain case attribute value that belongs to the provided
    values list

    Parameters
    -----------
    log
        Trace log
    values
        Allowed attribute values(if it's numerical value, [] is needed to make it a list)
    parameters
        Parameters of the algorithm, including:
            activity_key -> Attribute identifying the case in the log
            positive -> Indicate if events should be kept/removed

    Returns
    -----------
    filtered_log
        Filtered log
    """
    if parameters is None:
        parameters = {}

    attribute_key = parameters[
        PARAMETER_CONSTANT_ATTRIBUTE_KEY] if PARAMETER_CONSTANT_ATTRIBUTE_KEY in parameters else DEFAULT_NAME_KEY
    positive = parameters["positive"] if "positive" in parameters else True

    filtered_log = EventLog()
    for trace in log:
        new_trace = Trace()

        found = False
        if attribute_key in trace.attributes:
            attribute_value = trace.attributes[attribute_key]
            if attribute_value in list_of_values:
                found = True

        if (found and positive) or (not found and not positive):
            new_trace = trace
        else:
            for attr in trace.attributes:
                new_trace.attributes[attr] = trace.attributes[attr]

        if len(new_trace) > 0:
            filtered_log.append(new_trace)
    return filtered_log

def sublog2varlist(log, freq_thres, num):
    '''
    extract lists of variants from selected sublogs together with frequency threshold to filter out infrequent variants
    :param log: sublog containing the selected case attribute value
    :param freq_thres: (int) frequency threshold to filter out infrequent variants
    :return: lists of variant strings
    '''
    variants_count = case_statistics.get_variant_statistics(log)
    variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
    filtered_var_list = []
    filtered_var_list_1 = []
    filtered_var_list_2 = []
    for i in range(len(variants_count)):
        if variants_count[i]['count'] >= freq_thres:
            filtered_var_list_1.append(variants_count[i]['variant'])  # variant string
        elif i< num:
            filtered_var_list_2.append(variants_count[i]['variant'])

    # union set ensure the ordered union will be satisfied
    filtered_var_list = filtered_var_list_1+filtered_var_list_2
    #print(filtered_var_list)
    str_var_list = []
    for str in filtered_var_list:
        str_var_list.extend([str.split(',')])
    return str_var_list

def sublog_percent(log, percent):
    '''
    change variant dictionary got from sublog into dataframe, so that we can extract the frequency of each variant
    :param log: same as sublog2varlist()
    :param freq_thres: same as sublog2varlist()
    :return: dataframe of variants with their counts together with the correspond var_list(until the percent )
    '''
    variants_count = case_statistics.get_variant_statistics(log)
    variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
    df = pd.DataFrame.from_dict(variants_count)
    #calculate the cumunative sum
    csum = np.array(df['count']).cumsum()
    csum = csum/csum[-1]
    #print(csum)
    num_list = csum[csum<percent]
    #print(num_list)
    # stop until the percent is satisfied
    df_w_count = df.iloc[0:len(num_list), :]
    # get correspond var_list
    filtered_var_list = df_w_count['variant'].values.tolist()
    str_var_list = []
    for str in filtered_var_list:
        str_var_list.extend([str.split(',')])
    return df_w_count, str_var_list


def sublog2df_num(log, num):
    '''
    change variant dictionary got from sublog into dataframe, so that we can extract the frequency of each variant
    :param log: same as sublog2varlist()
    :param freq_thres: same as sublog2varlist()
    :return: dataframe of variants with their counts
    '''
    variants_count = case_statistics.get_variant_statistics(log)
    variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
    df = pd.DataFrame.from_dict(variants_count)
    df_w_count = df.iloc[0:num,:]
    return df_w_count


def sublog2df(log, freq_thres, num):
    '''
    change variant dictionary got from sublog into dataframe, so that we can extract the frequency of each variant
    :param log: same as sublog2varlist()
    :param freq_thres: same as sublog2varlist()
    :return: dataframe of variants with their counts
    '''
    variants_count = case_statistics.get_variant_statistics(log)
    variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
    df = pd.DataFrame.from_dict(variants_count)
    df_w_count_1 = df[df['count'] >= freq_thres]
    df_w_count_2 = df.iloc[0:num,:]
    # take union of two dataframes
    df_w_count = pd.merge(df_w_count_1, df_w_count_2, how='outer', on=['variant','count'])
    #display(df_w_count['variant'])
    return df_w_count


def act_dist(var_list_1, var_list_2, log1, log2, freq_thres):
    '''

    this function compare the activity similarity between two sublogs via the two lists of variants.
    :param var_list_1: lists of variants in sublog 1
    :param var_list_2: lists of variants in sublog 2
    :param freq_thres: same as sublog2df()
    :param log1: input sublog1 of sublog2df(), which must correspond to var_list_1
    :param log2: input sublog2 of sublog2df(), which must correspond to var_list_2
    :return: the distance matrix between 2 sublogs in which each element is the distance between two variants.
    '''

    if len(var_list_1) >= len(var_list_2):
        max_len = len(var_list_1)
        min_len = len(var_list_2)
        max_var = var_list_1
        min_var = var_list_2
        var_count_max = sublog2df(log1, freq_thres)['count']
        var_count_min = sublog2df(log2, freq_thres)['count']
    else:
        max_len = len(var_list_2)
        min_len = len(var_list_1)
        max_var = var_list_2
        min_var = var_list_1
        var_count_max = sublog2df(log2, freq_thres)['count']
        var_count_min = sublog2df(log1, freq_thres)['count']

    dist_matrix = np.zeros((max_len, min_len))

    for i in range(max_len):
        if i < min_len:
            for j in range(0, i + 1):
                result = Counter(max_var[i])  # count number of occurrence of each element
                df_1 = pd.DataFrame.from_dict(dict(result), orient='index',
                                              columns=['freq_1'])  # convert dict to dataframe
                df_1 = df_1.reset_index().rename(columns={'index': 'var'})
                result = Counter(min_var[j])  # count number of occurrence of each element
                df_2 = pd.DataFrame.from_dict(dict(result), orient='index', columns=['freq_2'])
                df_2 = df_2.reset_index().rename(columns={'index': 'var'})
                df = pd.merge(df_1, df_2, how='outer', on='var').fillna(
                    0)  # merge two variants and replace empty value by zero
                df['prod'] = df.apply(lambda x: x['freq_1'] * x['freq_2'], axis=1)
                df['sq_1'] = df.apply(lambda x: x['freq_1'] ** 2, axis=1)
                df['sq_2'] = df.apply(lambda x: x['freq_2'] ** 2, axis=1)
                innerprod = df['prod'].sum()
                sqrt_1 = np.sqrt(df['sq_1'].sum())
                sqrt_2 = np.sqrt(df['sq_2'].sum())
                # dist_matrix[i][j] = innerprod / (sqrt_1 * sqrt_2)
                dist_matrix[i][j] = (innerprod / (sqrt_1 * sqrt_2)) * var_count_max.iloc[i] * var_count_min.iloc[
                    j]  # weighted with trace frequency
                dist_matrix[j][i] = dist_matrix[i][j]
        if i >= min_len:
            for j in range(min_len):
                result = Counter(max_var[i])  # count number of occurrence of each element
                df_1 = pd.DataFrame.from_dict(dict(result), orient='index', columns=['freq_1'])
                df_1 = df_1.reset_index().rename(columns={'index': 'var'})
                result = Counter(min_var[j])  # count number of occurrence of each element
                df_2 = pd.DataFrame.from_dict(dict(result), orient='index', columns=['freq_2'])
                df_2 = df_2.reset_index().rename(columns={'index': 'var'})
                df = pd.merge(df_1, df_2, how='outer', on='var').fillna(0)
                df['prod'] = df.apply(lambda x: x['freq_1'] * x['freq_2'], axis=1)
                df['sq_1'] = df.apply(lambda x: x['freq_1'] ** 2, axis=1)
                df['sq_2'] = df.apply(lambda x: x['freq_2'] ** 2, axis=1)
                innerprod = df['prod'].sum()
                sqrt_1 = np.sqrt(df['sq_1'].sum())
                sqrt_2 = np.sqrt(df['sq_2'].sum())
                # dist_matrix[i][j] = innerprod / (sqrt_1 * sqrt_2)
                dist_matrix[i][j] = (innerprod / (sqrt_1 * sqrt_2)) * var_count_max.iloc[i] * var_count_min.iloc[
                    j]  # weighted with trace frequency
    if len(var_list_1) >= len(var_list_2):
        dist_matrix = dist_matrix
    else:
        dist_matrix = np.transpose(dist_matrix)

    return dist_matrix


'''
def get_trace_attribute(log):
    log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPI_Challenge_2012.xes")
    #log = xes_importer.apply("D:\\Sisc\\19SS\\APM\\APM_A1\\APM_Assignment_1.xes")


    eve=attributes_filter.get_all_event_attributes_from_log(log)
    print(eve)
    activities = attributes_filter.get_attribute_values(log, attribute_key="concept:name")#dictionary
    print(activities)
    '''

if __name__ == "__main__":
    log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPI_Challenge_2012.xes")
    # print(log)
    # extract all case attributes
    #att = attributes_filter.get_all_trace_attributes_from_log(log)
    #print(att)
    # extract all attribute values of the chosen case attribute
    #att_val = attributes_filter.get_trace_attribute_values(log, attribute_key="AMOUNT_REQ")
    # print(att_val)  # return dictionary value

    # give the attribute values needed to be filtered out

    list_of_vals = ['15000','25000','8000']

    tracefilter_log = apply_trace_attributes(log, list_of_vals,
                                             parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                         "positive": True})

    tracefilter_log_15000 = apply_trace_attributes(tracefilter_log, ['15000'],
                                                   parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                               "positive": True})
    tracefilter_log_8000 = apply_trace_attributes(tracefilter_log, ['8000'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})

    tracefilter_log_25000 = apply_trace_attributes(tracefilter_log, ['25000'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})


    '''
    tracefilter_log_30000 = apply_trace_attributes(tracefilter_log, ['30000'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})
    tracefilter_log_10000 = apply_trace_attributes(tracefilter_log, ['10000'],
                                                   parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                               "positive": True})
    
    tracefilter_log_7500 = apply_trace_attributes(tracefilter_log, ['7500'],
                                                   parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                               "positive": True})
    tracefilter_log_15000 = apply_trace_attributes(tracefilter_log, ['15000'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})

    tracefilter_log_22000 = apply_trace_attributes(tracefilter_log, ['22000'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})
    tracefilter_log_45000 = apply_trace_attributes(tracefilter_log, ['45000'],
                                                   parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                               "positive": True})'''

    #print(tracefilter_log_7000)
    #print(tracefilter_log_30000)


    number=4
    freq=1
    str_var_list_15000 = sublog2varlist(tracefilter_log_15000, freq,number)
    str_var_list_8000 = sublog2varlist(tracefilter_log_8000, freq, number)
    #print(str_var_list_15000)
    print(len(str_var_list_15000))
    print(len(str_var_list_8000))
    '''
    str_var_list_7500 = sublog2varlist(tracefilter_log_7500, freq,number)
    print(str_var_list_7500)
    #str_var_list_50000 = sublog2varlist(tracefilter_log_50000, freq,4)
    #print(len(str_var_list_50000))'''

    #str_var_list_30000 = sublog2varlist(tracefilter_log_30000, freq, number)
    #print(len(str_var_list_30000))
    '''
    str_var_list_10000 = sublog2varlist(tracefilter_log_10000, freq, number)
    str_var_list_7500 = sublog2varlist(tracefilter_log_7500, freq, number)
    str_var_list_15000 = sublog2varlist(tracefilter_log_15000, freq, number)
    '''
    #print(str_var_list_7000)
    #print(str_var_list_30000)
    #print(str_var_list_10000)

    # print(len(str_var_list_5000))
    # print(len(str_var_list_7000))
    #print(len(str_var_list_5000))
    #print(len(str_var_list_50002))
    #df = act_dist_calc.occu_var_act(str_var_list_5000[0])
    # print(np.array([df['freq'], df['freq']]))
    # print((1- pdist(np.array([df['freq'], df['freq']]), 'cosine'))[0])

    # print(case_statistics.get_cases_description(tracefilter_log_7000))

    #var_count_max = sublog2df(tracefilter_log_5000, 5)
    '''
    (a,b) = sublog_percent(tracefilter_log_10000, 0.4)
    print(len(a))
    print(b)
    (a, b) = sublog_percent(tracefilter_log_15000, 0.33)
    print(len(a))
    '''
    # print(len(str_var_list_7000))
    # print(act_dist_calc.occu_var_ele(str_var_list_5000[0]))

    # combi= functools.reduce(operator.concat, combi)# fastest way to flatten lists
    # print(combi)

    # display(str_var_list)

    # test spped and correctness
    start = time.time()
    # act_dis = act_dist_calc.act_dist(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000, 5)
    dist_mat = act_dist_calc.act_sim(str_var_list_15000, str_var_list_8000, tracefilter_log_15000, tracefilter_log_8000,
                                      freq, number, parameters={"single": False})
    print("1:",dist_mat)
    '''                                
    #dist_mat7 = act_dist_calc.act_sim_percent(tracefilter_log_15000, tracefilter_log_25000, 0.58, 0.58)
    # print("7:", dist_mat7)
    #dist_mat8 = act_dist_calc.act_sim_percent(tracefilter_log_15000, tracefilter_log_25000, 1, 1)
    #print("8:", dist_mat8)
    dist_mat9 = act_dist_calc.act_sim_percent(tracefilter_log_15000, tracefilter_log_8000, 0.58, 0.58)
    dist_mat10 = act_dist_calc.act_sim_percent(tracefilter_log_15000, tracefilter_log_8000, 1, 1)
    print("9:", dist_mat9)
    print("10:", dist_mat10)
    dist_mat11 = act_dist_calc.act_sim_percent(tracefilter_log_25000, tracefilter_log_8000, 0.58, 0.58)
    print("11:", dist_mat11)
    dist_mat12 = act_dist_calc.act_sim_percent(tracefilter_log_25000, tracefilter_log_8000, 1, 1)
    # dist_mat2 = suc_dist_calc.suc_sim(str_var_list_7500, str_var_list_15000, tracefilter_log_7500,
    #                                  tracefilter_log_15000,
    #                                  freq, number, parameters={"single": True})
    # print(dist_mat==np.transpose(dist_mat2))



    print("12:", dist_mat12)
    end = time.time()
    print(end - start)

    start = time.time()
     #act_dis = act_dist_calc.act_dist(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000, 5)
    #dist_mat = act_dist_calc.act_sim_dual(str_var_list_30000, str_var_list_50000, tracefilter_log_30000, tracefilter_log_50000,
    #                                  freq, number, parameters={"single": True})
    #print("1:",dist_mat)
    #dist_mat1 = suc_dist_calc.suc_sim_percent(tracefilter_log_15000,tracefilter_log_25000, 0.58, 0.58)
    #dist_mat2 = suc_dist_calc.suc_sim_percent(tracefilter_log_15000, tracefilter_log_25000, 1, 1)
    #print("1:", dist_mat1)
    #print("2:", dist_mat2)
    dist_mat3 = suc_dist_calc.suc_sim_percent(tracefilter_log_15000, tracefilter_log_8000, 0.58, 0.58)
    print("3:", dist_mat3)
    dist_mat4 = suc_dist_calc.suc_sim_percent(tracefilter_log_15000, tracefilter_log_8000, 1, 1)

    print("4:", dist_mat4)
    dist_mat5 = suc_dist_calc.suc_sim_percent(tracefilter_log_25000, tracefilter_log_8000, 0.58, 0.58)
    print("5:", dist_mat5)
    dist_mat6 = suc_dist_calc.suc_sim_percent(tracefilter_log_25000, tracefilter_log_8000, 1, 1)
    #dist_mat2 = suc_dist_calc.suc_sim(str_var_list_7500, str_var_list_15000, tracefilter_log_7500,
    #                                  tracefilter_log_15000,
    #                                  freq, number, parameters={"single": True})
    # print(dist_mat==np.transpose(dist_mat2))



    print("6:", dist_mat6)
    end = time.time()
    print(end - start)'''


    '''
    csv_exporter.export(tracefilter_log_1, "tracefilter_log_1.csv")
    variants = variants_filter.get_variants(tracefilter_log_1)
    df2 = (df['variant'].str.split(',', expand=True))
    print(df2[(df2.notnull())])
    print((df.loc[0, ['variant']]))'''
