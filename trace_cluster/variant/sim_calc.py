import pandas as pd
import numpy as np
import time
from trace_cluster.variant import act_dist_calc, suc_dist_calc
import filter_subsets
from pm4py.util import constants
from scipy.spatial.distance import pdist
from pm4py.objects.log.importer.xes import factory as xes_importer


def inner_prod_calc(df):
    innerprod = ((df.loc[:, 'freq_x']) * (df.loc[:, 'freq_y'])).sum()
    sqrt_1 = np.sqrt(((df.loc[:, 'freq_x']) ** 2).sum())
    sqrt_2 = np.sqrt(((df.loc[:, 'freq_y']) ** 2).sum())
    return innerprod, sqrt_1, sqrt_2


def dist_calc(var_list_1, var_list_2, log1, log2, freq_thres, num, alpha, parameters=None):
    '''
    this function compare the activity similarity between two sublogs via the two lists of variants.
    :param var_list_1: lists of variants in sublog 1
    :param var_list_2: lists of variants in sublog 2
    :param freq_thres: same as sublog2df()
    :param log1: input sublog1 of sublog2df(), which must correspond to var_list_1
    :param log2: input sublog2 of sublog2df(), which must correspond to var_list_2
    :param alpha: the weight parameter between activity similarity and succession similarity, which belongs to (0,1)
    :param parameters: state which linkage method to use
    :return: the similarity value between two sublogs
    '''

    if parameters is None:
        parameters = {}

    single = parameters["single"] if "single" in parameters else False

    if len(var_list_1) >= len(var_list_2):
        max_len = len(var_list_1)
        min_len = len(var_list_2)
        max_var = var_list_1
        min_var = var_list_2
        var_count_max = filter_subsets.sublog2df(log1, freq_thres, num)['count']
        var_count_min = filter_subsets.sublog2df(log2, freq_thres, num)['count']
    else:
        max_len = len(var_list_2)
        min_len = len(var_list_1)
        max_var = var_list_2
        min_var = var_list_1
        var_count_max = filter_subsets.sublog2df(log2, freq_thres, num)['count']
        var_count_min = filter_subsets.sublog2df(log1, freq_thres, num)['count']

    print("list1:", len(max_var))
    print("list2:", len(min_var))
    # act
    max_per_var_act = np.zeros(max_len)
    max_freq_act = np.zeros(max_len)
    col_sum_act = np.zeros(max_len)

    # suc
    max_per_var_suc = np.zeros(max_len)
    col_sum_suc = np.zeros(max_len)
    max_freq_suc = np.zeros(max_len)

    if var_list_1 == var_list_2:
        print("Please give different variant lists!")
    else:
        for i in range(max_len):
            dist_vec_act = np.zeros(min_len)
            dist_vec_suc = np.zeros(min_len)
            df_1_act = act_dist_calc.occu_var_act(max_var[i])
            df_1_suc = suc_dist_calc.occu_var_suc(max_var[i], parameters={"binarize": True})
            for j in range(min_len):
                df_2_act = act_dist_calc.occu_var_act(min_var[j])
                df_2_suc = suc_dist_calc.occu_var_suc(min_var[j], parameters={"binarize": True})

                df_act = pd.merge(df_1_act, df_2_act, how='outer', on='var').fillna(0)
                df_suc = pd.merge(df_1_suc, df_2_suc, how='outer', on='direct_suc').fillna(0)
                # cosine similarity is used to calculate trace similarity
                # pist defintion is distance, so it is subtracted by 1 already
                dist_vec_act[j] = (pdist(np.array([df_act['freq_x'].values, df_act['freq_y'].values]), 'cosine')[0])
                dist_vec_suc[j] = (pdist(np.array([df_suc['freq_x'].values, df_suc['freq_y'].values]), 'cosine')[0])

                if (single):
                    #dist_vec_act[j] = innerprod_act / (sqrt_1_act * sqrt_2_act)
                    #dist_vec_suc[j] = innerprod_suc / (sqrt_1_suc * sqrt_2_suc)
                    if (abs(dist_vec_act[j]) <= 1e-8) and (abs(dist_vec_suc[j]) <= 1e-6):  # ensure both are 1
                        max_freq_act[i] = var_count_max.iloc[i] * var_count_min.iloc[j]
                        max_freq_suc[i] = max_freq_act[i]
                        max_per_var_act[i] = dist_vec_act[j] * max_freq_act[i]
                        max_per_var_suc[i] = dist_vec_suc[j] * max_freq_suc[i]

                        break
                    elif j == (min_len - 1):
                        max_loc_col_act = np.argmin(dist_vec_act)  # location of max value
                        max_loc_col_suc = np.argmin(dist_vec_suc)  # location of max value
                        max_freq_act[i] = var_count_max.iloc[i] * var_count_min.iloc[max_loc_col_act]
                        max_freq_suc[i] = var_count_max.iloc[i] * var_count_min.iloc[max_loc_col_suc]
                        max_per_var_act[i] = dist_vec_act[max_loc_col_act] * max_freq_act[i]
                        max_per_var_suc[i] = dist_vec_suc[max_loc_col_suc] * max_freq_suc[i]

                else:
                    col_sum_act[i] += dist_vec_act[j] * var_count_max.iloc[i] * var_count_min.iloc[j]
                    col_sum_suc[i] += dist_vec_suc[j] * var_count_max.iloc[i] * var_count_min.iloc[j]
    if (single):
        # single linkage
        dist_act = np.sum(max_per_var_act) / np.sum(max_freq_act)
        dist_suc = np.sum(max_per_var_suc) / np.sum(max_freq_suc)
        dist = dist_act * alpha + dist_suc * (1 - alpha)
        # sim = (np.sum(max_per_var_act) * alpha) / np.sum(max_freq_act) + (
        #           np.sum(max_per_var_suc) * (1 - alpha)) / np.sum(max_freq_suc)
    else:
        vmax_vec = (var_count_max.values).reshape(-1, 1)
        vmin_vec = (var_count_min.values).reshape(1, -1)
        vec_sum = np.sum(np.dot(vmax_vec, vmin_vec))
        # dist = np.sum(dist_matrix) / vec_sum
        #dist_act = np.sum(col_sum_act) / vec_sum
        #dist_suc = np.sum(col_sum_suc) / vec_sum
        #sim = dist_act * alpha + dist_suc * (1 - alpha)
        dist = (np.sum(col_sum_act) * alpha + np.sum(col_sum_suc) * (1 - alpha)) / vec_sum

    # print(dist_matrix)
    # print(max_per_var)
    # print(max_freq)

    return  dist


if __name__ == "__main__":
    log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPI_Challenge_2012.xes")
    list_of_vals = ['5000', '7000']
    tracefilter_log = filter_subsets.apply_trace_attributes(log, list_of_vals,
                                                            parameters={
                                                                constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                "positive": True})
    tracefilter_log_7000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['7000'],
                                                                 parameters={
                                                                     constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                     "positive": True})
    tracefilter_log_5000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['5000'],
                                                                 parameters={
                                                                     constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                     "positive": True})
    threshold = 5

    str_var_list_5000 = filter_subsets.sublog2varlist(tracefilter_log_5000, threshold)
    str_var_list_7000 = filter_subsets.sublog2varlist(tracefilter_log_7000, threshold)
    # print(len(str_var_list_5000))
    start = time.time()
    # act_dis = act_dist_calc.act_dist(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000, 5)
    dist_mat1 = dist_calc(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000,
                         threshold, 0.5, parameters={"single": True})
    # dist_mat2 = suc_dist_calc.suc_sim(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000,
    #                    5, parameters={"single": True})
    print(dist_mat1)

    # print(dist_mat==np.transpose(dist_mat2))
    end = time.time()
    print(end - start)

    start = time.time()
    # act_dis = act_dist_calc.act_dist(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000, 5)
    dist_act = act_dist_calc.act_sim(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000,
                                     threshold, parameters={"single": True})
    dist_suc = suc_dist_calc.suc_sim(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000,
                                     threshold, parameters={"single": True})
    # dist_mat2 = suc_dist_calc.suc_sim(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000,
    #                    5, parameters={"single": True})

    dist = dist_act * 0.5 + dist_suc * (1 - 0.5)
    print(dist)

    # print(dist_mat==np.transpose(dist_mat2))
    end = time.time()
    print(end - start)