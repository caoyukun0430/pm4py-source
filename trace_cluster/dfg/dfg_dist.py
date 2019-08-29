import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.util import constants
import pandas as pd
import filter_subsets
from trace_cluster.variant import act_dist_calc
import cluster


def dfg_dist_calc_act(log1, log2):
    act1 = attributes_filter.get_attribute_values(log1, "concept:name")
    act2 = attributes_filter.get_attribute_values(log2, "concept:name")
    df1_act = act_dist_calc.occu_var_act(act1)
    df2_act = act_dist_calc.occu_var_act(act2)
    df_act = pd.merge(df1_act, df2_act, how='outer', on='var').fillna(0)
    #print(df_act)
    dist_act = pdist(np.array([df_act['freq_x'].values, df_act['freq_y'].values]), 'cosine')[0]
    #print([dist_act, dist_dfg])
    #dist = dist_act * alpha + dist_dfg * (1 - alpha)
    return dist_act


def dfg_dist_calc_suc(log1, log2):
    dfg1 = dfg_factory.apply(log1)
    dfg2 = dfg_factory.apply(log2)
    df1_dfg = act_dist_calc.occu_var_act(dfg1)
    df2_dfg = act_dist_calc.occu_var_act(dfg2)
    df_dfg = pd.merge(df1_dfg, df2_dfg, how='outer', on='var').fillna(0)
    #print(df_dfg)
    dist_dfg = pdist(np.array([df_dfg['freq_x'].values, df_dfg['freq_y'].values]), 'cosine')[0]
    return dist_dfg


def dfg_dist_calc(log1, log2):
    act1 = attributes_filter.get_attribute_values(log1, "concept:name")
    print(act1)
    act2 = attributes_filter.get_attribute_values(log2, "concept:name")
    dfg1 = dfg_factory.apply(log1)
    dfg2 = dfg_factory.apply(log2)
    df1_act = act_dist_calc.occu_var_act(act1)
    print(act1)
    df2_act = act_dist_calc.occu_var_act(act2)
    df1_dfg = act_dist_calc.occu_var_act(dfg1)
    df2_dfg = act_dist_calc.occu_var_act(dfg2)
    df_act = pd.merge(df1_act, df2_act, how='outer', on='var').fillna(0)
    print(df_act)
    #print(df_act)
    df_dfg = pd.merge(df1_dfg, df2_dfg, how='outer', on='var').fillna(0)
    #print(df_act)
    #print(df_dfg)
    dist_act = pdist(np.array([df_act['freq_x'].values, df_act['freq_y'].values]), 'cosine')[0]
    dist_dfg = pdist(np.array([df_dfg['freq_x'].values, df_dfg['freq_y'].values]), 'cosine')[0]
    print([dist_act, dist_dfg])
    #dist = dist_act * alpha + dist_dfg * (1 - alpha)
    return dist_act, dist_dfg


def dfg_dist_calc_minkowski(log1, log2, alpha):
    act1 = attributes_filter.get_attribute_values(log1, "concept:name")
    act2 = attributes_filter.get_attribute_values(log2, "concept:name")
    dfg1 = dfg_factory.apply(log1)
    dfg2 = dfg_factory.apply(log2)
    df1_act = act_dist_calc.occu_var_act(act1)
    df2_act = act_dist_calc.occu_var_act(act2)
    df1_dfg = act_dist_calc.occu_var_act(dfg1)
    df2_dfg = act_dist_calc.occu_var_act(dfg2)
    df_act = pd.merge(df1_act, df2_act, how='outer', on='var').fillna(0)
    #print(df_act)
    df_dfg = pd.merge(df1_dfg, df2_dfg, how='outer', on='var').fillna(0)
    #print(df_dfg)
    dist_act = pdist(np.array([df_act['freq_x'].values / np.sum(df_act['freq_x'].values),
                               df_act['freq_y'].values / np.sum(df_act['freq_y'].values)]), 'minkowski',p=2.)[0]
    dist_dfg = pdist(np.array([df_dfg['freq_x'].values / np.sum(df_dfg['freq_x'].values),
                               df_dfg['freq_y'].values / np.sum(df_dfg['freq_y'].values)]), 'minkowski',p=2.)[0]
    print([dist_act, dist_dfg])
    dist = dist_act * alpha + dist_dfg * (1 - alpha)
    return dist


if __name__ == "__main__":
    '''
    log_1 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_1_1.xes")
    log_2 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_2_1.xes")
    log_3 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_3_1.xes")
    log_4 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_4_1.xes")
    '''


    percent = 1
    alpha = 0.5
    #loglist = pt_gen.openAllXes("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 2)




    #real data
    log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPI_Challenge_2012.xes")
    list_of_vals = ['25000', '15000', '7000', '10000','12000']

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
    #dist = dfg_dist_calc_minkowski(lists[4], lists[5], alpha)
    #print(dist)



    for i in range(0, size - 1):
        for j in range(i + 1, size):
            (dist_act, dist_dfg) = dfg_dist_calc(lists[i], lists[j])
            dist_mat[i][j] = dist_act * alpha + dist_dfg * (1 - alpha)
            dist_mat[j][i] = dist_mat[i][j]

    print(dist_mat)

    y = squareform(dist_mat)
    print(y)
    Z = linkage(y, method='average')
    print(Z)
    print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    fig = plt.figure(figsize=(10, 8))
    # dn = fancy_dendrogram(Z, max_d=0.35)
    dn = dendrogram(Z)
    #dn = dendrogram(Z,labels=np.array(list_of_vals))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Loan Amount')
    plt.ylabel('Distance')
    plt.savefig('cluster.svg')
    plt.show()

    '''
    activities = attributes_filter.get_attribute_values(log_3, "concept:name")
    activities2 = attributes_filter.get_attribute_values(log_4, "concept:name")
    activities3 = attributes_filter.get_attribute_values(log_2, "concept:name")
    #dfg = dfg_factory.apply(log_3)
    #dfg2 = dfg_factory.apply(log_4)
    df1=(act_dist_calc.occu_var_act(activities))
    df2= act_dist_calc.occu_var_act(activities2)
    df3 = act_dist_calc.occu_var_act(activities3)
    #df3 = (act_dist_calc.occu_var_act(dfg))
    #df4 = act_dist_calc.occu_var_act(dfg2)
    df_1 = pd.merge(df1, df2, how='outer', on='var').fillna(0)
    df_2 = pd.merge(df1, df3, how='outer', on='var').fillna(0)
    print(df_1)
    print(df_2)
    dist1 = (pdist(np.array([df_1['freq_x'].values, df_1['freq_y'].values]), 'cosine')[0])
    dist2 = (pdist(np.array([df_2['freq_x'].values, df_2['freq_y'].values]), 'cosine')[0])
    print(dist1)
    print(dist2)'''
