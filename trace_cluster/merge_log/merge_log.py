import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import scipy.spatial
import scipy.cluster
import numpy as np
import json
import time
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, to_tree, fcluster
from scipy.spatial.distance import squareform
from trace_cluster import filter_subsets
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.evaluation.replay_fitness import factory as replay_factory
from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.evaluation.precision import factory as precision_factory
from pm4py.objects.log.log import EventLog
from pm4py.util import constants
from pm4py.algo.filtering.log.attributes import attributes_filter
from trace_cluster.evaluation import fake_log_eval
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from trace_cluster.variant import act_dist_calc
from trace_cluster.variant import suc_dist_calc
from trace_cluster.linkage_method import linkage_avg


def merge_log(path, cate, iter):
    '''
    modify the trace attribute for each log and then merge all fake sublogs into one log in order to use in WS
    :param path:
    :param cate:
    :param iter:
    :return:
    '''
    loglist = []
    mergedlog = EventLog()

    for i in range(1, cate + 1):
        for j in range(1, iter + 1):
            log = xes_importer.apply(path + '\\log_1_' + str(i) + '_' + str(j) + ".xes")
            for trace in log:
                trace.attributes["concept:name"] = str(iter * (i - 1) + j)
                trace.attributes["index"] = str(iter * (i - 1) + j)
            print(path + '\\log_1_' + str(i) + '_' + str(j) + ".xes")
            print(filter_subsets.sublog_percent(log, 1))
            loglist.append(log)

    for i in range(len(loglist)):
        for trace in loglist[i]:
            # print(trace)
            mergedlog.append(trace)

    return loglist, mergedlog


def update_merge(loglist):
    mergedlog = EventLog()

    for i in range(len(loglist)):
        for trace in loglist[i]:
            # print(trace)
            mergedlog.append(trace)
    return mergedlog


# this is for single string
def log2sublog(log, string, KEY):
    tracefilter_log = filter_subsets.apply_trace_attributes(log, [string],
                                                            parameters={
                                                                constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: KEY,
                                                                "positive": True})

    return tracefilter_log


# this is for string list
def logslice(log, str_list, KEY):
    tracefilter_log = filter_subsets.apply_trace_attributes(log, str_list,
                                                            parameters={
                                                                constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: KEY,
                                                                "positive": True})

    return tracefilter_log


# Create a nested dictionary from the ClusterNode's returned by SciPy
def add_node(node, parent):
    # First create the new node and append it to its parent's children
    newNode = dict(node_id=node.id, children=[])
    parent["children"].append(newNode)

    # Recursively add the current node's children
    if node.left: add_node(node.left, newNode)
    if node.right: add_node(node.right, newNode)


# Label each node with the names of each leaf in its subtree
def label_tree(n, id2name):
    # flatten_tree=[]
    # If the node is a leaf, then we have its name
    if len(n["children"]) == 0:
        leafNames = [id2name[n["node_id"]]]

    # If not, flatten all the leaves in the node's subtree
    else:
        leafNames = reduce(lambda ls, c: ls + label_tree(c, id2name), n["children"], [])
        print("flatten", leafNames)

    # Delete the node id since we don't need it anymore and
    # it makes for cleaner JSON
    del n["node_id"]

    # Labeling convention: "-"-separated leaf names
    n["name"] = name = "-".join(sorted(map(str, leafNames)))
    # flatten_tree.append(n)
    # print("flatree",flatten_tree)

    return leafNames


def get_dendrogram_svg(log, parameters=None):
    if parameters is None:
        parameters = {}

    cluster_avg = parameters["cluster_avg"] if "cluster_avg" in parameters else True

    percent = 1
    alpha = 0.5
    TYPE = 'index'

    list_of_vals = []
    list_log = []
    list_of_vals_dict = attributes_filter.get_trace_attribute_values(log, 'index')
    # ta = (attributes_filter.get_all_trace_attributes_from_log(log))
    # ea = (attributes_filter.get_all_event_attributes_from_log(log))
    # print(ta|ea)
    # print(list_of_vals_dict.keys())
    list_of_vals_keys = list(list_of_vals_dict.keys())
    for i in range(len(list_of_vals_keys)):
        list_of_vals.append(list_of_vals_keys[i])

    for i in range(len(list_of_vals)):
        logsample = log2sublog(log, list_of_vals[i], TYPE)
        list_log.append(logsample)

    if (cluster_avg):
        y = fake_log_eval.eval_avg_leven(list_log, percent, alpha)
    else:
        y = fake_log_eval.eval_DMM_leven(list_log, percent, alpha)

    Z = linkage(y, method='average')

    # Create dictionary for labeling nodes by their IDs

    id2name = dict(zip(range(len(list_of_vals)), list_of_vals))

    T = to_tree(Z, rd=False)
    d3Dendro = dict(children=[], name="Root1")
    add_node(T, d3Dendro)

    label_tree(d3Dendro["children"][0], id2name)
    d3Dendro = d3Dendro["children"][0]
    d3Dendro["name"] = 'root'
    ret = d3Dendro
    print(ret)

    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # dn = dendrogram(Z, labels=list_of_vals)
    # # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # plt.title('Hierarchical Clustering Dendrogram')
    # # plt.xlabel('Loan Amount')
    # plt.ylabel('Distance')
    # plt.savefig('cluster.svg')
    # plt.show()


def clusteredlog(Z, maxclust, list_of_vals, log, TYPE):
    clu_index = fcluster(Z, maxclust, criterion='maxclust')
    clu_index = dict(zip(list_of_vals, clu_index))
    clu_list_log = []
    clu_list = []
    for i in range(maxclust):
        temp = [key for key, value in clu_index.items() if value == i + 1]
        print([i, temp])
        clu_list.append(temp)
        logtemp = logslice(log, temp, TYPE)
        clu_list_log.append(logtemp)
        filename = 'log' + '_' + str(maxclust) + '_' + str(i) + '_' + TYPE + '.xes'
        xes_exporter.export_log(logtemp, filename)
    return clu_list_log, clu_list


if __name__ == "__main__":

    log = xes_importer.apply(
        "D:\\Sisc\\19SS\\thesis\\Dataset\\document_logs\\Control summary.xes")

    # sublog = xes_importer.apply(
    #     "D:\\Sisc\\19SS\\thesis\\Dataset\\BPIC2017\\sublog_598.xes")
    # log1 = xes_importer.apply(
    #     "C:\\Users\\yukun\\PycharmProjects\\pm4py-source\\trace_cluster\\merge_log\\log_3_0_dfg.xes")

    percent = 1
    alpha = 0.5
    attr_name = 'amount_applied0'
    TYPE = 'DMM' + attr_name

    list_of_vals = []
    list_log = []
    list_of_vals_dict = attributes_filter.get_trace_attribute_values(log, attr_name)

    list_of_vals_keys = list(list_of_vals_dict.keys())
    for i in range(len(list_of_vals_keys)):
        list_of_vals.append(list_of_vals_keys[i])

    print(list_of_vals)
    for i in range(len(list_of_vals)):
        logsample = log2sublog(log, list_of_vals[i], attr_name)
        list_log.append(logsample)
    # print(list_log)

    # DFG test
    start = time.time()
    y = fake_log_eval.dfg_dis(list_log, percent, alpha, list_of_vals)
    # y = fake_log_eval.eval_DMM_variant(list_log, percent, alpha)
    print(y)
    Z = linkage(y, method='average')
    print(Z)

    dn = dendrogram(Z, labels=np.array(list_of_vals))
    # plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel(attr_name)
    plt.ylabel('Distance')
    plt.savefig('cluster_wupdate' + '_' + TYPE + '.svg')
    plt.show()

    # clu_list_log2, clu_list2 = clusteredlog(Z, 2, list_of_vals, log, TYPE)
    # clu_list_log3, clu_list3 = clusteredlog(Z,3,list_of_vals,log,TYPE)
    #
    # clu_list_log4, clu_list4 = clusteredlog(Z, 4, list_of_vals, log,TYPE)
    # clu_list_log5, clu_list5 = clusteredlog(Z, 5, list_of_vals, log,TYPE)
    # clu_list_log6, clu_list6 = clusteredlog(Z, 6, list_of_vals, log,TYPE)
    # clu_list_log7, clu_list7 = clusteredlog(Z, 7, list_of_vals, log,TYPE)

    plot_clu = 7
    plot_fit = dict()
    plot_prec = dict()
    plot_F1 = dict()
    plot_box = dict()
    clu_list_dict = dict()
    for i in range(1, plot_clu + 1):
        if i == 1:
            inductive_petri, inductive_initial_marking, inductive_final_marking = inductive_miner.apply(log)
            fitness = replay_factory.apply(log, inductive_petri, inductive_initial_marking,
                                           inductive_final_marking, variant="alignments")['averageFitness']

            precision = precision_factory.apply(log, inductive_petri, inductive_initial_marking,
                                                inductive_final_marking)
            F1 = 2 * fitness * precision / (fitness + precision)
            plot_fit[str(i)] = fitness
            plot_prec[str(i)] = precision
            plot_F1[str(i)] = F1
            plot_box[str(i)] = pd.Series(F1)
        else:
            clu_list_log, clu_list = clusteredlog(Z, i, list_of_vals, log, attr_name)
            clu_list_dict[str(i)] = clu_list
            length_li = []
            fit_li = []
            prec_li = []
            F1_li = []
            for j in range(0, i):
                length = len(clu_list_log[j])
                inductive_petri, inductive_initial_marking, inductive_final_marking = inductive_miner.apply(
                    clu_list_log[j])
                fitness = replay_factory.apply(log, inductive_petri, inductive_initial_marking,
                                               inductive_final_marking, variant="alignments")['averageFitness']
                precision = precision_factory.apply(log, inductive_petri, inductive_initial_marking,
                                                    inductive_final_marking)
                F1 = 2 * fitness * precision / (fitness + precision)
                # individual info for each sublog
                length_li.append(length)
                fit_li.append(fitness)
                prec_li.append(precision)
                F1_li.append(F1)
            print(length_li)
            print("fit", fit_li)
            print("prec", prec_li)
            plot_fit[str(i)] = np.average(fit_li, weights=length_li)
            plot_prec[str(i)] = np.average(prec_li, weights=length_li)
            plot_F1[str(i)] = np.average(F1_li, weights=length_li)
            plot_box[str(i)] = pd.Series(F1_li)

    # print(plot_fit)
    # print(plot_prec)
    # print(plot_F1)
    # print(plot_box)
    # print(clu_list_dict)

    x_axis = range(1, plot_clu + 1)

    # plot fit&prec
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(x_axis, list(plot_fit.values()), color="r", linestyle="-", marker="s", linewidth=1, label='Fitness')  # 画图
    # ax1.set_ylim(0,1.02)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Num. of Cluster')
    ax1.set_xticks(x_axis)
    ax1.yaxis.label.set_color('r')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')
    ax2 = ax1.twinx()

    ax2.plot(x_axis, list(plot_prec.values()), color="b", linestyle="-", marker="s", linewidth=1,
             label='Precision')  # 画图
    # ax2.set_ylim(0,1.02)
    ax2.set_ylim(np.min(list(plot_prec.values())) - 0.01, 1)
    ax2.set_ylabel('Precision')
    ax2.yaxis.label.set_color('b')
    for tl in ax2.get_yticklabels():
        tl.set_color('b')
    # plt.grid(axis='x')
    fig.savefig('fitprec' + '_' + TYPE + '.svg')
    fig.show()

    # plot F1
    fig2 = plt.figure()
    plt.plot(x_axis, list(plot_F1.values()), color="b", linestyle="-", marker="s", linewidth=1)
    plt.ylim(np.min(list(plot_F1.values())) - 0.01, 1)
    # plt.ylim(0,1)
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    # plt.grid(axis='x')
    plt.savefig('f1' + '_' + TYPE + '.svg')
    plt.show()

    # rescale to 0-1
    # plot fit&prec
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(x_axis, list(plot_fit.values()), color="r", linestyle="-", marker="s", linewidth=1, label='Fitness')  # 画图
    ax1.set_ylim(0, 1.04)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Num. of Cluster')
    ax1.set_xticks(x_axis)
    ax1.yaxis.label.set_color('r')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')
    ax2 = ax1.twinx()

    ax2.plot(x_axis, list(plot_prec.values()), color="b", linestyle="-", marker="s", linewidth=1,
             label='Precision')  # 画图
    ax2.set_ylim(0, 1.04)
    # ax2.set_ylim(np.min(list(plot_prec.values()))-0.01,1)
    ax2.set_ylabel('Precision')
    ax2.yaxis.label.set_color('b')
    for tl in ax2.get_yticklabels():
        tl.set_color('b')
    # plt.grid(axis='x')
    fig.savefig('fitprec_sca' + '_' + TYPE + '.svg')
    fig.show()

    # plot F1
    fig2 = plt.figure()
    plt.plot(x_axis, list(plot_F1.values()), color="b", linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis)
    # plt.ylim(np.min(list(plot_F1.values()))-0.01,1)
    plt.ylim(0, 1)
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    # plt.grid(axis='x')
    plt.savefig('f1_sca' + '_' + TYPE + '.svg')
    plt.show()

    # plot boxplot
    fig3 = plt.figure()
    plot_box["2"] = plot_box["1"]

    data = pd.DataFrame(plot_box)
    print(data)
    plt.plot(x_axis, list(plot_F1.values()), color="b", linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis)
    data.boxplot(sym='o')

    plt.ylim(np.min(plot_box[str(plot_clu)]) - 0.01, 1)
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='x')
    plt.savefig('f1_boxplot' + '_' + TYPE + '.svg')
    plt.show()

    end = time.time()
    print("woupdate", end - start)
    # print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # # dn = dendrogram(Z)
    # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Credit Score')
    # plt.ylabel('Distance')
    # plt.savefig('cluster_dfg_woupdate.svg')
    # plt.show()
    #
    # dist_mat = squareform(y)
    # Z = linkage_avg.linkage_dfg_update(list_log, dist_mat,alpha,percent)
    # print("Zupdate2", Z)
    # end = time.time()
    # print("wupdate", end - start)
    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # # dn = dendrogram(Z)
    # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Credit Score')
    # plt.ylabel('Distance')
    # plt.savefig('cluster_dfg_wupdate.svg')
    # plt.show()

    # list1 = fcluster(Z, 2, criterion='maxclust')
    # list2 = fcluster(Z, 3, criterion='maxclust')
    # list3 = fcluster(Z, 6, criterion='maxclust')
    # list1=dict(zip(list_of_vals, list1))
    # list2 = dict(zip(list_of_vals, list2))
    # list3 = dict(zip(list_of_vals, list3))
    # print([list1,list2,list3])
    # clu_list_log2, clu_list2 = clusteredlog(Z, 2, list_of_vals, log, 'dfg')
    # clu_list_log3, clu_list3 = clusteredlog(Z,3,list_of_vals,log,'dfg')
    #
    # clu_list_log4, clu_list4 = clusteredlog(Z, 4, list_of_vals, log,'dfg')
    # clu_list_log5, clu_list5 = clusteredlog(Z, 5, list_of_vals, log,'dfg')
    # clu_list_log6, clu_list6 = clusteredlog(Z, 6, list_of_vals, log,'dfg')
    # clu_list_log7, clu_list7 = clusteredlog(Z, 7, list_of_vals, log,'dfg')

    # for i in range(k):
    #     filename = 'log'+'_'+str(k)+'_'+str(i)+'.xes'
    #     xes_exporter.export_log(clu_list_log[i], filename)

    # log1 = logslice(log, [0])
    # print(len(log1))
    # inductive_petri, inductive_initial_marking, inductive_final_marking = inductive_miner.apply(log1)
    # fitness_inductive = replay_factory.apply(log, inductive_petri, inductive_initial_marking, inductive_final_marking)
    # print("fitness_inductive=", fitness_inductive)
    #
    # gviz = pn_vis_factory.apply(inductive_petri, inductive_initial_marking, inductive_final_marking)
    # pn_vis_factory.save(gviz,"ind.png")
    #
    # precision_inductive = precision_factory.apply(log, inductive_petri, inductive_initial_marking,
    #                                               inductive_final_marking)
    # print("precision_inductive=", precision_inductive)
    # precision_inductive = precision_factory.apply(log, inductive_petri, inductive_initial_marking,
    #                                               inductive_final_marking)
    # print("precision_inductive=", precision_inductive)

    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # # dn = dendrogram(Z)
    # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Credit Score')
    # plt.ylabel('Distance')
    # plt.savefig('cluster_dfg.svg')
    # plt.show()

    # DMM test
    # y = fake_log_eval.eval_DMM_variant(list_log, percent, alpha)
    # Z = linkage(y, method='average')
    # print(Z)
    # end = time.time()
    # print("DMMwoupdate",end - start)
    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # # dn = dendrogram(Z)
    # dn = dendrogram(Z, labels=np.array(list_of_vals))
    # # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Credit Score')
    # plt.ylabel('Distance')
    # plt.savefig('cluster_DMM_woupdate.svg')
    # plt.show()

    # dist_mat = squareform(y)
    # Z = linkage_avg.linkage_DMM_update(list_log, dist_mat, alpha, percent)
    # print("Zupdate2", Z)
    # end = time.time()
    # print("DMMwupdate",end - start)
    # # clu_list_log2, clu_list2 = clusteredlog(Z, 2, list_of_vals, log, 'DMM')
    # # clu_list_log3, clu_list3 = clusteredlog(Z,3,list_of_vals,log,'DMM')
    # #
    # # clu_list_log4, clu_list4 = clusteredlog(Z, 4, list_of_vals, log,'DMM')
    # # clu_list_log5, clu_list5 = clusteredlog(Z, 5, list_of_vals, log,'DMM')
    # # clu_list_log6, clu_list6 = clusteredlog(Z, 6, list_of_vals, log,'DMM')
    # # clu_list_log7, clu_list7 = clusteredlog(Z, 7, list_of_vals, log,'DMM')
    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # # dn = dendrogram(Z)
    # dn = dendrogram(Z, labels=np.array(list_of_vals))
    # # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Credit Score')
    # plt.ylabel('Distance')
    # plt.savefig('cluster_DMM_wupdate.svg')
    # plt.show()

    # avg test
    # start = time.time()
    # y = fake_log_eval.eval_avg_variant(list_log, percent, alpha)
    # Z = linkage(y, method='average')
    # # print(Z)
    # end = time.time()
    # print("avgwupdate", end - start)
    # # print("avgwoupdate",end - start)
    # # plt.figure(figsize=(10, 8))
    # # # dn = fancy_dendrogram(Z, max_d=0.35)
    # # # dn = dendrogram(Z)
    # # dn = dendrogram(Z, labels=np.array(list_of_vals))
    # # # plt.title('Hierarchical Clustering Dendrogram')
    # # plt.xlabel('Credit Score')
    # # plt.ylabel('Distance')
    # # plt.savefig('cluster_avg_woupdate.svg')
    # # plt.show()
    #
    # start2 = time.time()
    # y = fake_log_eval.eval_avg_variant(list_log, percent, alpha)
    # dist_mat = squareform(y)
    # Z = linkage_avg.linkage_avg(list_log, dist_mat)
    # print("Zupdate2", Z)
    # end2 = time.time()
    # print("avgwpdate", end2 - start2)
    #
    # clu_list_log2, clu_list2 = clusteredlog(Z, 2, list_of_vals, log, 'avg')
    # clu_list_log3, clu_list3 = clusteredlog(Z,3,list_of_vals,log,'avg')
    #
    # clu_list_log4, clu_list4 = clusteredlog(Z, 4, list_of_vals, log,'avg')
    # clu_list_log5, clu_list5 = clusteredlog(Z, 5, list_of_vals, log,'avg')
    # clu_list_log6, clu_list6 = clusteredlog(Z, 6, list_of_vals, log,'avg')
    # clu_list_log7, clu_list7 = clusteredlog(Z, 7, list_of_vals, log,'avg')

    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # # dn = dendrogram(Z)
    # dn = dendrogram(Z, labels=np.array(list_of_vals))
    # # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Credit Score')
    # plt.ylabel('Distance')
    # plt.savefig('cluster_avg_wupdate.svg')
    # plt.show()

    # dist_act = act_dist_calc.act_sim_percent_avg(list_log[4], list_log[8], percent, percent)
    # dist_suc = suc_dist_calc.suc_sim_percent_avg(list_log[4], list_log[8], percent, percent)
    # print([dist_act, dist_suc])

    # percent = 1
    # alpha = 0.5
    # (loglist, mergedlog) = merge_log("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 5)
    #
    # #get_dendrogram_svg(mergedlog,parameters={"cluster_avg":True})
    # #tr=(attributes_filter.get_trace_attribute_values(mergedlog,'concept:name'))
    # #print(len(tr))
    #
    # xes_exporter.export_log(mergedlog, "mergedlog_all.xes")

    # ta=(attributes_filter.get_all_trace_attributes_from_log(mergedlog))
    # ea=(attributes_filter.get_all_event_attributes_from_log(mergedlog))
    # print((ta))
    # print((ea))
    '''
    list_of_vals = []
    list_log = []
    list_of_vals_dict = attributes_filter.get_trace_attribute_values(mergedlog, 'index')
    print(list_of_vals_dict)
    #print(list_of_vals_dict.keys())
    list_of_vals_keys = list(list_of_vals_dict.keys())
    #list_of_vals_keys = ['1', '2', '4', '5', '7', '8']
    for i in range(len(list_of_vals_keys)):
        list_of_vals.append(list_of_vals_keys[i])

    print(list_of_vals)
    

    for i in range(len(list_of_vals)):
        logsample = log2sublog(mergedlog, list_of_vals[i])
        list_log.append(logsample)
        # print(filter_subsets.sublog_percent(logsample, 1))

    id2name = dict(zip(range(len(list_of_vals)), list_of_vals))
    print(id2name)

    y = fake_log_eval.eval_avg_leven(loglist, percent, alpha)
    Z = linkage(y, method='average')
    #print(Z)
    #print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    T = to_tree(Z, rd=False)
    d3Dendro = dict(children=[], name="Root1")
    add_node(T, d3Dendro)
    print(d3Dendro["children"][0])

    tr = label_tree(d3Dendro["children"][0],id2name)
    #print(d3Dendro["name"])
    print(d3Dendro["children"][0])
    d3Dendro=d3Dendro["children"][0]
    d3Dendro["name"] = 'root'
    
    json.dump(d3Dendro, open("d3-dendrogram.json", "w"), sort_keys=True)

    plt.figure(figsize=(10, 8))
    # dn = fancy_dendrogram(Z, max_d=0.35)
    dn = dendrogram(Z,labels=list_of_vals)
    # dn = dendrogram(Z,labels=np.array(list_of_vals))
    plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Loan Amount')
    plt.ylabel('Distance')
    plt.savefig('cluster.svg')
    plt.show()
    #fake_log_eval.eval_avg_variant(list_log, percent, alpha)
    # eval_DMM_variant(loglist, percent, alpha)'''



