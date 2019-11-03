import pandas as pd
import numpy as np
import time
import filter_subsets
from scipy.spatial.distance import pdist
from collections import Counter
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.util import constants


def occu_suc(dfg, filter_percent):
    '''

    :param dfg: a counter containing all the direct succession relationship with frequency
    :param filter_percent: clarify the percentage of direct succession one wants to preserve
    :return: dataframe of direct succession relationship with frequency
    '''

    df = pd.DataFrame.from_dict(dict(dfg), orient='index', columns=['freq'])
    df = df.sort_values(axis=0, by=['freq'], ascending=False)
    df = df.reset_index().rename(columns={'index': 'suc'})
    # delete duplicated successions
    df = df.drop_duplicates('suc', keep='first')
    # delete self succession
    # filter out direct succession by percentage
    filter = list(range(0, round(filter_percent * len(df))))
    df = df[(df.index).isin(filter)].reset_index(drop=True)
    return df


def occu_var_suc(var_list, parameters=None):
    '''
    return dataframe that shows the frequency of each element(direct succession) in each variant list
    :param var_list:
    :param parameters: binarize states if user wants to binarize the frequency, default is binarized
    :return:
    '''
    if parameters is None:
        parameters = {}

    binarize = parameters["binarize"] if "binarize" in parameters else True

    comb_list = [var_list[i] + ',' + var_list[i + 1] for i in range(len(var_list) - 1)]
    result = Counter(comb_list)  # count number of occurrence of each element
    df = pd.DataFrame.from_dict(dict(result), orient='index', columns=['freq'])
    df = df.reset_index().rename(columns={'index': 'direct_suc'})
    if (binarize):
        # Binarize succession frequency (optional)
        df.loc[df.freq > 1, 'freq'] = 1
        return df
    else:
        return df


def suc_sim(var_list_1, var_list_2, log1, log2, freq_thres, num, parameters=None):
    '''

    this function compare the activity similarity between two sublogs via the two lists of variants.
    :param var_list_1: lists of variants in sublog 1
    :param var_list_2: lists of variants in sublog 2
    :param freq_thres: same as sublog2df()
    :param log1: input sublog1 of sublog2df(), which must correspond to var_list_1
    :param log2: input sublog2 of sublog2df(), which must correspond to var_list_2
    :return: the distance matrix between 2 sublogs in which each element is the distance between two variants.
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

    #print(filter_subsets.sublog2df(log1, freq_thres))
    print(var_count_max)
    print(var_count_min)
    dist_matrix = np.zeros((max_len, min_len))
    max_per_var = np.zeros(max_len)
    max_freq = np.zeros(max_len)
    col_sum = np.zeros(max_len)

    if var_list_1 == var_list_2:
        print("Please give different variant lists!")
    else:
        for i in range(max_len):
            dist_vec = np.zeros(min_len)
            df_1 = occu_var_suc(max_var[i], parameters={"binarize": False})
            for j in range(min_len):
                df_2 = occu_var_suc(min_var[j], parameters={"binarize": False})
                df = pd.merge(df_1, df_2, how='outer', on='direct_suc').fillna(0)
                print(df)
                # cosine similarity is used to calculate trace similarity
                dist_vec[j] = (pdist(np.array([df['freq_x'].values, df['freq_y'].values]), 'cosine')[0])
                '''

                # use tuple is slow
                innerprod = sim_calc.inner_prod_calc(df)[0]
                sqrt_1 = sim_calc.inner_prod_calc(df)[1]
                sqrt_2 = sim_calc.inner_prod_calc(df)[2]
                
                innerprod = ((df.loc[:, 'freq_x']) * (df.loc[:, 'freq_y'])).sum()
                sqrt_1 = np.sqrt(((df.loc[:, 'freq_x']) ** 2).sum())
                sqrt_2 = np.sqrt(((df.loc[:, 'freq_y']) ** 2).sum())
                '''

                dist_matrix[i][j] = dist_vec[j]
                if (single):
                    # dist_matrix[i][j] = innerprod / (sqrt_1 * sqrt_2)
                    # dist_vec[j] = innerprod / (sqrt_1 * sqrt_2)
                    if abs(dist_vec[j]) <= 1e-8:
                        print([i,j])
                        # max_per_var[i] = dist_vec[j]
                        # max_per_var[i] = dist_matrix[i][j] * var_count_max.iloc[i] * var_count_min.iloc[j]
                        max_freq[i] = var_count_max.iloc[i] * var_count_min.iloc[j]
                        max_per_var[i] = dist_vec[j] * max_freq[i]

                        break
                    elif j == (min_len - 1):
                        # max_loc_col = np.argmax(dist_matrix[i, :])  # location of max value
                        max_loc_col = np.argmin(dist_vec)  # location of max value
                        # max_per_var[i] = dist_vec[max_loc_col]
                        # max_per_var[i] = dist_matrix[i][max_loc_col] * var_count_max.iloc[i] * var_count_min.iloc[
                        #    max_loc_col]
                        max_freq[i] = var_count_max.iloc[i] * var_count_min.iloc[max_loc_col]
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i]
                        # print([i,max_loc_col])
                else:
                    # dist_matrix[i][j] = (innerprod / (sqrt_1 * sqrt_2)) * var_count_max.iloc[i] * var_count_min.iloc[
                    # j]  # weighted with trace frequency
                    col_sum[i] += dist_vec[j] * var_count_max.iloc[i] * var_count_min.iloc[j]
                    # col_sum[i] += dist_vec[j]

    if (single):
        # single linkage
        dist = np.sum(max_per_var) / np.sum(max_freq)
    else:
        vmax_vec = (var_count_max.values).reshape(-1, 1)
        vmin_vec = (var_count_min.values).reshape(1, -1)
        vec_sum = np.sum(np.dot(vmax_vec, vmin_vec))
        # dist = np.sum(dist_matrix) / vec_sum
        dist = np.sum(col_sum) / vec_sum

    print(dist_matrix)
    # print(max_per_var)
    # print(max_freq)

    return dist


def suc_sim_dual(var_list_1, var_list_2, log1, log2, freq_thres, num, parameters=None):
    '''

    this function compare the activity similarity between two sublogs via the two lists of variants.
    :param var_list_1: lists of variants in sublog 1
    :param var_list_2: lists of variants in sublog 2
    :param freq_thres: same as sublog2df()
    :param log1: input sublog1 of sublog2df(), which must correspond to var_list_1
    :param log2: input sublog2 of sublog2df(), which must correspond to var_list_2
    :return: the distance matrix between 2 sublogs in which each element is the distance between two variants.
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

    # print(filter_subsets.sublog2df(log1, freq_thres))
    # dist_matrix = np.zeros((max_len, min_len))
    dist_matrix = np.zeros((max_len, min_len))
    max_per_var = np.zeros(max_len)
    max_freq = np.zeros(max_len)
    min_freq = np.zeros(min_len)
    min_per_var = np.zeros(min_len)
    col_sum = np.zeros(max_len)
    index_rec = set(list(range(min_len)))

    if var_list_1 == var_list_2:
        print("Please give different variant lists!")
    else:
        for i in range(max_len):
            dist_vec = np.zeros(min_len)
            df_1 = occu_var_suc(max_var[i], parameters={"binarize": False})
            for j in range(min_len):
                df_2 = occu_var_suc(min_var[j], parameters={"binarize": False})
                df = pd.merge(df_1, df_2, how='outer', on='direct_suc').fillna(0)
                # cosine similarity is used to calculate trace similarity
                dist_vec[j] = (pdist(np.array([df['freq_x'].values, df['freq_y'].values]), 'cosine')[0])
                dist_matrix[i][j] = dist_vec[j]
                '''

                # use tuple is slow
                innerprod = sim_calc.inner_prod_calc(df)[0]
                sqrt_1 = sim_calc.inner_prod_calc(df)[1]
                sqrt_2 = sim_calc.inner_prod_calc(df)[2]

                innerprod = ((df.loc[:, 'freq_x']) * (df.loc[:, 'freq_y'])).sum()
                sqrt_1 = np.sqrt(((df.loc[:, 'freq_x']) ** 2).sum())
                sqrt_2 = np.sqrt(((df.loc[:, 'freq_y']) ** 2).sum())
                '''

                # dist_vec[j] = innerprod / (sqrt_1 * sqrt_2)
                if j == (min_len - 1):
                    # max_loc_col = np.argmax(dist_matrix[i, :])  # location of max value
                    max_loc_col = np.argmin(dist_vec)
                    if abs(dist_vec[max_loc_col]) <= 1e-8:
                        index_rec.discard(max_loc_col)
                        # print("skip:",[i,max_loc_col])
                        max_freq[i] = var_count_max.iloc[i] * var_count_min.iloc[max_loc_col] * 2
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i] * 2
                    else:
                        max_freq[i] = var_count_max.iloc[i] * var_count_min.iloc[max_loc_col]
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i]

        if (len(index_rec) != 0):
            for i in list(index_rec):
                min_loc_row = np.argmin(dist_matrix[:, i])
                min_freq[i] = var_count_max.iloc[min_loc_row] * var_count_min.iloc[i]
                min_per_var[i] = dist_matrix[min_loc_row, i] * min_freq[i]

    if (single):
        # single linkage
        dist = np.sum(max_per_var) / np.sum(max_freq)
    else:
        vmax_vec = (var_count_max.values).reshape(-1, 1)
        vmin_vec = (var_count_min.values).reshape(1, -1)
        vec_sum = np.sum(np.dot(vmax_vec, vmin_vec))
        # dist = np.sum(dist_matrix) / vec_sum
        dist = np.sum(col_sum) / vec_sum

    # print(dist_matrix)
    # print(max_per_var)
    # print(max_freq)

    return dist


def suc_sim_percent(log1, log2, percent_1, percent_2):
    '''

    this function compare the activity similarity between two sublogs via the two lists of variants.
    :param var_list_1: lists of variants in sublog 1
    :param var_list_2: lists of variants in sublog 2
    :param freq_thres: same as sublog2df()
    :param log1: input sublog1 of sublog2df(), which must correspond to var_list_1
    :param log2: input sublog2 of sublog2df(), which must correspond to var_list_2
    :return: the distance matrix between 2 sublogs in which each element is the distance between two variants.
    '''

    (dataframe_1, var_list_1) = filter_subsets.sublog_percent(log1, percent_1)
    (dataframe_2, var_list_2) = filter_subsets.sublog_percent(log2, percent_2)

    if len(var_list_1) >= len(var_list_2):
        max_len = len(var_list_1)
        min_len = len(var_list_2)
        max_var = var_list_1
        min_var = var_list_2
        var_count_max = dataframe_1['count']
        var_count_min = dataframe_2['count']
    else:
        max_len = len(var_list_2)
        min_len = len(var_list_1)
        max_var = var_list_2
        min_var = var_list_1
        var_count_max = dataframe_2['count']
        var_count_min = dataframe_1['count']
    #print("list1:",len(max_var))
    #print("list2:",len(min_var))

    # print(len(var_count_max))
    # print(len(var_count_min))
    dist_matrix = np.zeros((max_len, min_len))
    max_per_var = np.zeros(max_len)
    max_freq = np.zeros(max_len)
    min_freq = np.zeros(min_len)
    min_per_var = np.zeros(min_len)
    col_sum = np.zeros(max_len)
    index_rec = set(list(range(min_len)))

    if var_list_1 == var_list_2:
        print("Please give different variant lists!")
        dist = 0
    else:
        for i in range(max_len):
            dist_vec = np.zeros(min_len)
            df_1 = occu_var_suc(max_var[i], parameters={"binarize": False})
            for j in range(min_len):
                df_2 = occu_var_suc(min_var[j], parameters={"binarize": False})
                df = pd.merge(df_1, df_2, how='outer', on='direct_suc').fillna(0)
                # cosine similarity is used to calculate trace similarity
                dist_vec[j] = (pdist(np.array([df['freq_x'].values, df['freq_y'].values]), 'cosine')[0])
                dist_matrix[i][j] = dist_vec[j]
                # dist_matrix[i][j] = innerprod / (sqrt_1 * sqrt_2)
                if j == (min_len - 1):
                    # max_loc_col = np.argmax(dist_matrix[i, :])  # location of max value
                    max_loc_col = np.argmin(dist_vec)
                    if abs(dist_vec[max_loc_col]) <= 1e-8:
                        index_rec.discard(max_loc_col)
                        # print("skip:",[i,max_loc_col])
                        max_freq[i] = var_count_max.iloc[i] * var_count_min.iloc[max_loc_col] * 2
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i] * 2
                    else:
                        max_freq[i] = var_count_max.iloc[i] * var_count_min.iloc[max_loc_col]
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i]
                        # print(type(dist_vec)) # location of max value
                        # print([i,max_loc_col])
                        # max_per_var[i] = dist_vec[max_loc_col]
                        # max_per_var[i] = dist_matrix[i][max_loc_col] * var_count_max.iloc[i] * var_count_min.iloc[
                        #    max_loc_col]

        if (len(index_rec) != 0):
            for i in list(index_rec):
                min_loc_row = np.argmin(dist_matrix[:, i])
                min_freq[i] = var_count_max.iloc[min_loc_row] * var_count_min.iloc[i]
                min_per_var[i] = dist_matrix[min_loc_row, i] * min_freq[i]
            # print("dual",[i,min_loc_row])
            # print("dual",dist_matrix[min_loc_row, i])

        # print(max_freq)
        # single linkage
        dist = (np.sum(max_per_var) + np.sum(min_per_var)) / (np.sum(max_freq) + np.sum(min_freq))

    # print(index_rec)
    #print(dist_matrix)
    # print(max_per_var)
    # print(max_freq)

    return dist


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
    '''
    dfg_5000 = dfg_factory.apply(tracefilter_log_5000, variant='frequency')
    dfg_7000 = dfg_factory.apply(tracefilter_log_7000, variant='frequency')
    df_5000 = occu_suc(dfg_5000, 0.5)
    df_7000 = occu_suc(dfg_7000, 0.5)
    '''
    #df = pd.merge(df_5000, df_7000, how='outer', on='suc').fillna(0)
    # print(df.drop_duplicates('suc', keep='first'))
    str_var_list_5000 = filter_subsets.sublog2varlist(tracefilter_log_5000, 5)
    str_var_list_7000 = filter_subsets.sublog2varlist(tracefilter_log_7000, 5)
    #display(str_var_list_5000)
    df1 = occu_var_suc(str_var_list_5000[2], parameters={"binarize": False})
    #print(df1)
    '''
    print(len(str_var_list_5000))
    print(len(str_var_list_7000))
    var_count_min = filter_subsets.sublog2df(tracefilter_log_7000, 5)
    # display(act_dist_calc.occu_var_ele(str_var_list_5000[0])

    # print(var_count_min)

    list1 = str_var_list_7000[3]
    df1 = occu_var_suc(list1, parameters={"binarize": False})
    print(df1)
    list2 = str_var_list_5000[3]
    df2 = occu_var_suc(list2, parameters={"binarize": False})
    print(df2)
    df = pd.merge(df1, df2, how='outer', on='direct_suc').fillna(0)
    start = time.time()
    print(((df.loc[:, 'freq_x']) ** 2))
    end = time.time()
    print(end - start)

    start = time.time()
    print(df.apply(lambda x: x['freq_x'] ** 2, axis=1))
    end = time.time()
    print(end - start)
    '''



    # test spped and correctness
    start = time.time()
    # act_dis = act_dist_calc.act_dist(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000, 5)
    dist_mat2 = suc_sim(str_var_list_5000, str_var_list_7000, tracefilter_log_5000, tracefilter_log_7000,
                                       5, parameters={"single": True})
    print(1-dist_mat2)
    # print(dist_mat==np.transpose(dist_mat2))
    end = time.time()
    print(end - start)


