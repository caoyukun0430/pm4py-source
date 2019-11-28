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

from trace_cluster.merge_log import merge_log
from trace_cluster.evaluation import fake_log_eval

VARIANT_DMM_LEVEN = "variant_DMM_leven"
VARIANT_AVG_LEVEN = "variant_avg_leven"
VARIANT_DMM_VEC = "variant_DMM_vec"
VARIANT_AVG_VEC = "variant_avg_vec"
DFG = 'dfg'

VERSION_METHODS = {VARIANT_DMM_LEVEN: fake_log_eval.eval_DMM_leven, VARIANT_AVG_LEVEN: fake_log_eval.eval_avg_leven,
                   VARIANT_DMM_VEC: fake_log_eval.eval_DMM_variant, VARIANT_AVG_VEC: fake_log_eval.eval_avg_variant,
                   DFG:fake_log_eval.dfg_dis}


def apply(log, variant=VARIANT_DMM_LEVEN, parameters=None):
    if parameters is None:
        parameters = {}

    percent = 1
    alpha = 0.5
    ATTR_NAME = 'responsible'

    list_of_vals = []
    list_log = []
    list_of_vals_dict = attributes_filter.get_trace_attribute_values(log, ATTR_NAME)

    list_of_vals_keys = list(list_of_vals_dict.keys())
    for i in range(len(list_of_vals_keys)):
        list_of_vals.append(list_of_vals_keys[i])

    for i in range(len(list_of_vals)):
        logsample = merge_log.log2sublog(log, list_of_vals[i],ATTR_NAME)
        list_log.append(logsample)

    if variant in VERSION_METHODS:
        y = VERSION_METHODS[variant](list_log, percent, alpha)

    Z = linkage(y, method='average')

    # Create dictionary for labeling nodes by their IDs

    id2name = dict(zip(range(len(list_of_vals)), list_of_vals))

    T = to_tree(Z, rd=False)
    d3Dendro = dict(children=[], name="Root1")
    merge_log.add_node(T, d3Dendro)

    leafname = merge_log.label_tree(d3Dendro["children"][0], id2name)
    #print("leafname",leafname)
    d3Dendro = d3Dendro["children"][0]
    d3Dendro["name"] = 'root'
    ret = d3Dendro
    print(ret)


    return ret,leafname


if __name__ == "__main__":
    percent = 1
    alpha = 0.5
    (loglist, mergedlog) = merge_log.merge_log("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 2)

    apply(mergedlog, variant=VARIANT_DMM_LEVEN)
