import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.util import constants
from pm4py.algo.discovery.dfg.versions import native
import pandas as pd
import filter_subsets
from trace_cluster.variant import act_dist_calc
import cluster
import time

'''
def slice_dist_act(log_1,log_2,unit):


    (log1_list, freq1_list) = filter_subsets.logslice_percent_act(log_1, unit)
    (log2_list, freq2_list) = filter_subsets.logslice_percent_act(log_2, unit)

    if len(freq1_list) >= len(freq2_list):
        max_len = len(freq1_list)
        min_len = len(freq2_list)
        max_log = log1_list
        min_log = log2_list
        var_count_max = freq1_list
        var_count_min = freq2_list

    else:
        max_len = len(freq2_list)
        min_len = len(freq1_list)
        max_log = log2_list
        min_log = log1_list
        var_count_max = freq2_list
        var_count_min = freq1_list

    print(max_log)
    print(min_log)
    print((var_count_max))
    print((var_count_min))
    dist_matrix = np.zeros((max_len, min_len))
    max_per_var = np.zeros(max_len)
    max_freq = np.zeros(max_len)
    min_freq = np.zeros(min_len)
    min_per_var = np.zeros(min_len)
    col_sum = np.zeros(max_len)
    index_rec = set(list(range(min_len)))

    if log1_list == log2_list:
        print("Please give different variant lists!")
        dist = 0
    else:
        for i in range(max_len):
            dist_vec = np.zeros(min_len)
            df1_act = act_dist_calc.occu_var_act(max_log[i])
            for j in range(min_len):
                df2_act = act_dist_calc.occu_var_act(min_log[j])
                df_act = pd.merge(df1_act, df2_act, how='outer', on='var').fillna(0)
                print(df_act)
                dist_vec[j] = pdist(np.array([df_act['freq_x'].values, df_act['freq_y'].values]), 'cosine')[0]
                dist_matrix[i][j] = dist_vec[j]
                # print(dist_vec[j])
                if j == (min_len - 1):
                    max_loc_col = np.argmin(dist_vec)
                    # print([i,max_loc_col])
                    if abs(dist_vec[max_loc_col]) <= 1e-8:
                        # print("skip:", [i, max_loc_col])
                        index_rec.discard(max_loc_col)
                        max_freq[i] = var_count_max[i] * var_count_min[max_loc_col] * 2
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i] * 2
                    else:
                        max_freq[i] = var_count_max[i] * var_count_min[max_loc_col]
                        #print("max", [i, max_loc_col])
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i]
                        #print(dist_vec[max_loc_col])

        if (len(index_rec) != 0):
            # print(index_rec)
            for i in list(index_rec):
                min_loc_row = np.argmin(dist_matrix[:, i])
                min_freq[i] = var_count_max[min_loc_row] * var_count_min[i]
                min_per_var[i] = dist_matrix[min_loc_row, i] * min_freq[i]
        # print(max_freq, max_per_var)
        # print(min_freq,min_per_var)

        dist = (np.sum(max_per_var) + np.sum(min_per_var)) / (np.sum(max_freq) + np.sum(min_freq))

    #print(dist_matrix)


    return dist
'''

def slice_dist_suc(log_1,log_2,unit):


    (log1_list, freq1_list) = filter_subsets.logslice_percent(log_1, unit)
    (log2_list, freq2_list) = filter_subsets.logslice_percent(log_2, unit)

    if len(freq1_list) >= len(freq2_list):
        max_len = len(freq1_list)
        min_len = len(freq2_list)
        max_log = log1_list
        min_log = log2_list
        var_count_max = freq1_list
        var_count_min = freq2_list

    else:
        max_len = len(freq2_list)
        min_len = len(freq1_list)
        max_log = log2_list
        min_log = log1_list
        var_count_max = freq2_list
        var_count_min = freq1_list

    #print((var_count_max))
    #print((var_count_min))
    dist_matrix = np.zeros((max_len, min_len))
    max_per_var = np.zeros(max_len)
    max_freq = np.zeros(max_len)
    min_freq = np.zeros(min_len)
    min_per_var = np.zeros(min_len)
    col_sum = np.zeros(max_len)
    index_rec = set(list(range(min_len)))

    if log1_list == log2_list:
        print("Please give different variant lists!")
        dist = 0
    else:
        for i in range(max_len):
            dist_vec = np.zeros(min_len)
            dfg1 = native.apply(max_log[i])
            df1_dfg = act_dist_calc.occu_var_act(dfg1)
            for j in range(min_len):
                dfg2 = native.apply(min_log[j])
                df2_dfg = act_dist_calc.occu_var_act(dfg2)
                df_dfg = pd.merge(df1_dfg, df2_dfg, how='outer', on='var').fillna(0)
                dist_vec[j] = pdist(np.array([df_dfg['freq_x'].values, df_dfg['freq_y'].values]), 'cosine')[0]
                dist_matrix[i][j] = dist_vec[j]
                # print(dist_vec[j])
                if j == (min_len - 1):
                    max_loc_col = np.argmin(dist_vec)
                    # print([i,max_loc_col])
                    if abs(dist_vec[max_loc_col]) <= 1e-8:
                        # print("skip:", [i, max_loc_col])
                        index_rec.discard(max_loc_col)
                        max_freq[i] = var_count_max[i] * var_count_min[max_loc_col] * 2
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i] * 2
                    else:
                        max_freq[i] = var_count_max[i] * var_count_min[max_loc_col]
                        #print("max", [i, max_loc_col])
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i]
                        #print(dist_vec[max_loc_col])

        if (len(index_rec) != 0):
            # print(index_rec)
            for i in list(index_rec):
                min_loc_row = np.argmin(dist_matrix[:, i])
                min_freq[i] = var_count_max[min_loc_row] * var_count_min[i]
                min_per_var[i] = dist_matrix[min_loc_row, i] * min_freq[i]
        # print(max_freq, max_per_var)
        # print(min_freq,min_per_var)

        dist = (np.sum(max_per_var) + np.sum(min_per_var)) / (np.sum(max_freq) + np.sum(min_freq))

    #print(dist_matrix)
    return dist


def slice_dist_act(log_1,log_2,unit, parameters=None):


    (log1_list, freq1_list) = filter_subsets.logslice_percent(log_1, unit)
    (log2_list, freq2_list) = filter_subsets.logslice_percent(log_2, unit)

    if len(freq1_list) >= len(freq2_list):
        max_len = len(freq1_list)
        min_len = len(freq2_list)
        max_log = log1_list
        min_log = log2_list
        var_count_max = freq1_list
        var_count_min = freq2_list

    else:
        max_len = len(freq2_list)
        min_len = len(freq1_list)
        max_log = log2_list
        min_log = log1_list
        var_count_max = freq2_list
        var_count_min = freq1_list

    #print((var_count_max))
    #print((var_count_min))
    dist_matrix = np.zeros((max_len, min_len))
    max_per_var = np.zeros(max_len)
    max_freq = np.zeros(max_len)
    min_freq = np.zeros(min_len)
    min_per_var = np.zeros(min_len)
    col_sum = np.zeros(max_len)
    index_rec = set(list(range(min_len)))

    if log1_list == log2_list:
        print("Please give different variant lists!")
        dist = 0
    else:
        for i in range(max_len):
            dist_vec = np.zeros(min_len)
            act1 = attributes_filter.get_attribute_values(max_log[i], "concept:name")
            df1_act = act_dist_calc.occu_var_act(act1)
            for j in range(min_len):
                act2 = attributes_filter.get_attribute_values(min_log[j], "concept:name")
                df2_act = act_dist_calc.occu_var_act(act2)
                df_act = pd.merge(df1_act, df2_act, how='outer', on='var').fillna(0)
                dist_vec[j] = pdist(np.array([df_act['freq_x'].values, df_act['freq_y'].values]), 'cosine')[0]
                dist_matrix[i][j] = dist_vec[j]
                # print(dist_vec[j])
                if j == (min_len - 1):
                    max_loc_col = np.argmin(dist_vec)
                    # print([i,max_loc_col])
                    if abs(dist_vec[max_loc_col]) <= 1e-8:
                        # print("skip:", [i, max_loc_col])
                        index_rec.discard(max_loc_col)
                        max_freq[i] = var_count_max[i] * var_count_min[max_loc_col] * 2
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i] * 2
                    else:
                        max_freq[i] = var_count_max[i] * var_count_min[max_loc_col]
                        #print("max", [i, max_loc_col])
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i]
                        #print(dist_vec[max_loc_col])

        if (len(index_rec) != 0):
            # print(index_rec)
            for i in list(index_rec):
                min_loc_row = np.argmin(dist_matrix[:, i])
                min_freq[i] = var_count_max[min_loc_row] * var_count_min[i]
                min_per_var[i] = dist_matrix[min_loc_row, i] * min_freq[i]
        # print(max_freq, max_per_var)
        # print(min_freq,min_per_var)

        dist = (np.sum(max_per_var) + np.sum(min_per_var)) / (np.sum(max_freq) + np.sum(min_freq))

    #print(dist_matrix)
    return dist


if __name__ == "__main__":
    percent = 1
    alpha = 0.5
    unit = 1

    #loglist = pt_gen.openAllXes("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 2)
    '''
    dist=slice_dist(loglist[1],loglist[3],0.1,parameters={"choose_act":True})
    print(dist)
    #(dist_act, dist_dfg) = dfg_dist.dfg_dist_calc(loglist[1],loglist[3])
    #print(dist_act)
    sim_act = act_dist_calc.act_sim_percent(loglist[1],loglist[3], percent, percent)
    print(sim_act)'''
    

    log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPI_Challenge_2012.xes")
    list_of_vals = ['25000', '15000', '7000', '10000', '12000']

    tracefilter_log = filter_subsets.apply_trace_attributes(log, list_of_vals,
                                                            parameters={
                                                                constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                "positive": True})
    loglist = []

    for i in range(len(list_of_vals)):
        logsample = cluster.log2sublog(tracefilter_log, list_of_vals[i])
        loglist.append(logsample)


    lists = loglist
    size = len(lists)
    print(size)
    dist_mat = np.zeros((size, size))

    start = time.time()

    for i in range(0, size - 1):
        for j in range(i + 1, size):
            sim_act = slice_dist_act(loglist[i],loglist[j],unit)
            # print([i, len(lists[i]), j, len(lists[j]), sim_act])

            sim_suc = slice_dist_suc(loglist[i],loglist[j],unit)
            # print([i, len(lists[i]), j, len(lists[j]), sim_suc])
            print([i, j, sim_act, sim_suc])

            # sim_suc = suc_dist_calc.suc_sim(lists[i], lists[j], lists[i+size],
            #                                lists[j+size],
            #                                freq, num, parameters={"single": True})
            # dist_mat[i][j] = sim_act
            dist_mat[i][j] = (sim_act * alpha + sim_suc * (1 - alpha))
            dist_mat[j][i] = dist_mat[i][j]

    end = time.time()
    print(end - start)
    print(dist_mat)

    y = squareform(dist_mat)
    print(y)
    Z = linkage(y, method='average')
    print(Z)
    print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    fig = plt.figure(figsize=(10, 8))
    # dn = fancy_dendrogram(Z, max_d=0.35)
    dn = dendrogram(Z)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.savefig('cluster1.svg')
    plt.show()
