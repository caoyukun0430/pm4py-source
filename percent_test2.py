import time
from scipy.spatial.distance import pdist
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import act_dist_calc
import suc_dist_calc
import filter_subsets
from IPython.display import display
from collections import Counter
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.log import EventLog, Trace, EventStream
from pm4py.util.constants import PARAMETER_CONSTANT_ATTRIBUTE_KEY, PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util import constants
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.log import case_statistics

log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPI_Challenge_2012.xes")
list_of_vals = ['8000','12000','9000','7000','30000']

tracefilter_log = filter_subsets.apply_trace_attributes(log, list_of_vals,
                                                        parameters={
                                                            constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                            "positive": True})
'''

tracefilter_log_5500 = filter_subsets.apply_trace_attributes(tracefilter_log, ['5500'],
                                                             parameters={
                                                                 constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                 "positive": True})
tracefilter_log_17500 = filter_subsets.apply_trace_attributes(tracefilter_log, ['17500'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})
tracefilter_log_22000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['22000'],
                                                   parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                               "positive": True})
tracefilter_log_11000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['11000'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})'''


tracefilter_log_9000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['9000'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})
tracefilter_log_8000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['8000'],
                                                   parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                               "positive": True})
tracefilter_log_12000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['12000'],
                                                   parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                               "positive": True})
tracefilter_log_7000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['7000'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})
tracefilter_log_30000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['30000'],
                                                             parameters={
                                                                 constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                 "positive": True})
dist_value_1 = []
dist_value_2 = []
dist_value_3 = []
dist_value_4 = []
dist_value_5 = []
dist_value_6 = []
dist_value_7 = []

percent_value = np.array(range(40, 51,2)) * 0.01

for i in range(len(percent_value)):


    dist_mat_3 = act_dist_calc.act_sim_percent(tracefilter_log_30000, tracefilter_log_9000, percent_value[i],
                                               1)
    dist_value_3.append(dist_mat_3)
    print("3:", dist_value_3)

    dist_mat_4 = act_dist_calc.act_sim_percent(tracefilter_log_9000, tracefilter_log_30000, percent_value[i],
                                               1)
    dist_value_4.append(dist_mat_4)
    print("4:", dist_value_4)

    dist_mat_2 = act_dist_calc.act_sim_percent(tracefilter_log_9000, tracefilter_log_30000, percent_value[i],
                                               percent_value[i])
    dist_value_2.append(dist_mat_2)
    print("2:", dist_value_2)

for i in range(len(percent_value)):
    dist_mat_5 = suc_dist_calc.suc_sim_percent(tracefilter_log_9000, tracefilter_log_30000, percent_value[i],
                                               1)
    dist_value_5.append(dist_mat_5)
    print("5:", dist_value_5)

    dist_mat_6 = suc_dist_calc.suc_sim_percent(tracefilter_log_30000, tracefilter_log_9000, percent_value[i],
                                              1)
    dist_value_6.append(dist_mat_6)
    print("6:", dist_value_6)

    dist_mat_7 = suc_dist_calc.suc_sim_percent(tracefilter_log_9000, tracefilter_log_30000, percent_value[i],
                                               percent_value[i])
    dist_value_7.append(dist_mat_7)
    print("7:", dist_value_7)
