import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import scipy.spatial
import scipy.cluster
import numpy as np
import json
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, to_tree
from scipy.spatial.distance import squareform
from trace_cluster import filter_subsets
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.log import EventLog
from pm4py.util import constants
from pm4py.algo.filtering.log.attributes import attributes_filter
from trace_cluster.evaluation import fake_log_eval
from pm4py.algo.discovery.alpha import factory as alpha_miner
from pm4py.visualization.petrinet import factory as pn_vis_factory
from trace_cluster.merge_log import merge_log
from trace_cluster.evaluation import fake_log_eval

VARIANT_DMM_LEVEN = "variant_DMM_leven"
VARIANT_AVG_LEVEN = "variant_avg_leven"
VARIANT_DMM_VEC = "variant_DMM_vec"
VARIANT_AVG_VEC = "variant_avg_vec"

VERSION_METHODS = {VARIANT_DMM_LEVEN: fake_log_eval.eval_DMM_leven, VARIANT_AVG_LEVEN: fake_log_eval.eval_avg_leven,
                   VARIANT_DMM_VEC: fake_log_eval.eval_DMM_variant, VARIANT_AVG_VEC: fake_log_eval.eval_avg_variant}


def apply(log, variant=VARIANT_DMM_LEVEN, parameters=None):
    if parameters is None:
        parameters = {}

    percent = 1
    alpha = 0.5

    list_of_vals = []
    list_log = []
    list_of_vals_dict = attributes_filter.get_trace_attribute_values(log, 'index')

    list_of_vals_keys = list(list_of_vals_dict.keys())
    for i in range(len(list_of_vals_keys)):
        list_of_vals.append(list_of_vals_keys[i])

    for i in range(len(list_of_vals)):
        logsample = merge_log.log2sublog(log, list_of_vals[i])
        list_log.append(logsample)

    if variant in VERSION_METHODS:
        y = VERSION_METHODS[variant](list_log, percent, alpha)

    Z = linkage(y, method='average')

    # Create dictionary for labeling nodes by their IDs

    id2name = dict(zip(range(len(list_of_vals)), list_of_vals))
    print("id",id2name)

    T = to_tree(Z, rd=False)
    d3Dendro = dict(children=[], name="Root1")
    merge_log.add_node(T, d3Dendro)
    print("d3", d3Dendro)

    leafname = merge_log.label_tree(d3Dendro["children"][0], id2name)
    print("leafname",leafname)
    d3Dendro = d3Dendro["children"][0]
    d3Dendro["name"] = 'root'
    ret = d3Dendro
    print("ret",ret)
    results = []


    # # rec returns a generator
    # def rec(current_object):
    #     if isinstance(current_object, dict):
    #         yield current_object["name"]
    #         for item in rec(current_object["children"]):
    #             yield item
    #     elif isinstance(current_object, list):
    #         for items in current_object:
    #             for item in rec(items):
    #                 yield item
    #
    # print("res",list(rec(ret)))


    # get the triple group of nodes
    def bfs(tree):
        queue =[]
        output = []
        queue.append(tree)
        while queue:
            #element in queue is waiting to become root and splited into child
            # root is the first ele of queue
            root = queue.pop(0)
            if len(root['children'])>0:
                name = [root['name']]
                for child in root['children']:
                    queue.append(child)
                    name.append(child['name'])
                output.append(name)

        return output



    trilist = bfs(ret)
    trilist[0][0] = trilist[0][1] + '-' + trilist[0][2]
    print("res",trilist )

    rootlist=[]
    for ele in trilist:
        rootlist.append(ele[0])
    # we should not pop it, because the selectindex will be wrong
    # rootlist.pop(0)
    # rootlist[0]=rootlist[1]+'-'+rootlist[2]
    print("rootlist",rootlist)



    # selectedNode='1-2-3-4'
    #
    # select_index = rootlist.index(selectedNode)
    # show_triple = trilist[select_index]
    # print(show_triple)
    # slice_val = []
    # for ele in show_triple:
    #     slice_val.append(ele.split('-'))
    # print(slice_val)


    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # dn = dendrogram(Z, labels=list_of_vals)
    # # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # plt.title('Hierarchical Clustering Dendrogram')
    # # plt.xlabel('Loan Amount')
    # plt.ylabel('Distance')
    # plt.savefig('cluster.svg')
    # plt.show()

    return ret,leafname


if __name__ == "__main__":
    percent = 1
    alpha = 0.5
    (loglist, mergedlog) = merge_log.merge_log("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 2)

    # mergedlog = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\pm4py-ws\\logs\\mergedlog_2_J235.xes")
    (ret,leafname)=apply(mergedlog, variant=VARIANT_DMM_LEVEN)

    slice_num = 4
    index_list=[]
    length = int(len(leafname) / slice_num)
    # for i in range(0,slice_num):
    #     index_list.append(int(i*length))
    # print(index_list)

    # slice_val=[]
    # #slice list_of_vals
    # for i in range(slice_num):
    #     slice_val.append(leafname[i*length:(i+1)*length])
    # print(slice_val)
    slice_val = [['1', '2', '3', '4'], ['1', '2'], ['3', '4']]

    gviz_list=[]
    for i in range(len(slice_val)):
        logsample = merge_log.logslice(mergedlog, slice_val[i])
        net, initial_marking, final_marking = alpha_miner.apply(logsample)
        parameters = {"format": "svg"}
        gviz = pn_vis_factory.apply(net, initial_marking, final_marking,parameters={"format":"svg"})
        filenname = "C:\\Users\yukun\\PycharmProjects\\pm4py-source\\trace_cluster\\evaluation\\"+"pn"+str(i+1)+".svg"
        print(filenname)
        pn_vis_factory.save(gviz, filenname)
        gviz_list.append({"name":filenname})

    print("list",gviz_list)
    for gviz in gviz_list:
        print(gviz)

