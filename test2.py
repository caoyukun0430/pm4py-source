import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import time
import act_dist_calc
import filter_subsets
import suc_dist_calc
import sim_calc
from pm4py.util import constants
from scipy.spatial.distance import pdist
from IPython.display import display
from collections import Counter
from pm4py.objects.log.importer.xes import factory as xes_importer


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def log2sublog(log, str):
    tracefilter_log = filter_subsets.apply_trace_attributes(log, [str],
                                                            parameters={
                                                                constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                "positive": True})

    return tracefilter_log


if __name__ == "__main__":
    # inputs
    log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPI_Challenge_2012.xes")
    list_of_vals = ['25000', '10000', '15000', '30000', '7000','8000']
    percent = 1
    alpha = 0.5

    tracefilter_log = filter_subsets.apply_trace_attributes(log, list_of_vals,
                                                            parameters={
                                                                constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                "positive": True})
    loglist = []
    varlist = []

    for i in range(len(list_of_vals)):
        logsample = log2sublog(tracefilter_log, list_of_vals[i])
        loglist.append(logsample)
    # print((loglist[0]))
    # print((varlist[0]))
    # print(len(varlist))
    lists = loglist
    print(lists)
    size = len(lists)
    print(size)
    dist_mat = np.zeros((size, size))

    for i in range(0, size - 1):
        for j in range(i + 1, size):
            sim_act = act_dist_calc.act_sim_percent(lists[i], lists[j], percent,percent)
            print([i,len(lists[i]),j,len(lists[j]),sim_act])
            sim_suc = suc_dist_calc.suc_sim_percent(lists[i], lists[j], percent,percent)
            print([i,len(lists[i]),j,len(lists[j]),sim_suc])

            # sim_suc = suc_dist_calc.suc_sim(lists[i], lists[j], lists[i+size],
            #                                lists[j+size],
            #                                freq, num, parameters={"single": True})

            '''
            sim_act = sim_calc.dist_calc(lists[i], lists[j], lists[i + size],
                                         lists[j + size],
                                         freq, num, 0.5, parameters={"single": True})'''
            #dist_mat[i][j] = sim_act
            dist_mat[i][j] = (sim_act * alpha + sim_suc * (1 - alpha))
            dist_mat[j][i] = dist_mat[i][j]

    print(dist_mat)

    y = squareform(dist_mat)
    print(y)
    Z = linkage(y, method='average')
    print(Z)
    print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    fig = plt.figure(figsize=(10, 8))
    # dn = fancy_dendrogram(Z, max_d=0.35)
    dn = fancy_dendrogram(Z)
    plt.savefig('cluster.svg')
    plt.show()
