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
list_of_vals = ['10000','15000','8000','7000','30000']

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


tracefilter_log_15000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['15000'],
                                                  parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                              "positive": True})
tracefilter_log_8000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['8000'],
                                                   parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                               "positive": True})
tracefilter_log_10000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['10000'],
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
dist_value_8 = []
dist_value_9 = []
dist_value_10 = []

percent_value = np.array(range(4, 11)) * 0.1
percent_value[0]= 0.44
for i in range(len(percent_value)):
    dist_mat_2 = act_dist_calc.act_sim_percent(tracefilter_log_10000, tracefilter_log_7000, percent_value[i],
                                               percent_value[i])
    dist_value_2.append(dist_mat_2)
    print("2:", dist_value_2)

    dist_mat_3 = act_dist_calc.act_sim_percent(tracefilter_log_7000, tracefilter_log_10000, percent_value[i],
                                               1)
    dist_value_3.append(dist_mat_3)
    print("3:", dist_value_3)

    dist_mat_4 = act_dist_calc.act_sim_percent(tracefilter_log_10000, tracefilter_log_7000, percent_value[i],
                                               1)
    dist_value_4.append(dist_mat_4)
    print("4:", dist_value_4)


percent_value = np.array(range(4, 11)) * 0.1
for i in range(len(percent_value)):
    dist_mat_5 = act_dist_calc.act_sim_percent(tracefilter_log_8000, tracefilter_log_15000, percent_value[i],
                                               1)
    dist_value_5.append(dist_mat_5)
    print("5:", dist_value_5)

    dist_mat_6 = act_dist_calc.act_sim_percent(tracefilter_log_15000, tracefilter_log_8000, percent_value[i],
                                              1)
    dist_value_6.append(dist_mat_6)
    print("6:", dist_value_6)

    dist_mat_7 = act_dist_calc.act_sim_percent(tracefilter_log_8000, tracefilter_log_15000, percent_value[i],
                                               percent_value[i])
    dist_value_7.append(dist_mat_7)
    print("7:", dist_value_7)

for i in range(len(percent_value)):
    dist_mat_8 = act_dist_calc.act_sim_percent(tracefilter_log_10000, tracefilter_log_15000, percent_value[i],
                                               1)
    dist_value_8.append(dist_mat_8)
    print("8:", dist_value_8)

    dist_mat_9 = act_dist_calc.act_sim_percent(tracefilter_log_15000, tracefilter_log_10000, percent_value[i],
                                              1)
    dist_value_9.append(dist_mat_9)
    print("9:", dist_value_9)

    dist_mat_10 = act_dist_calc.act_sim_percent(tracefilter_log_10000, tracefilter_log_15000, percent_value[i],
                                               percent_value[i])
    dist_value_10.append(dist_mat_10)
    print("10:", dist_value_10)





'''

percent_value = np.array(range(58, 60,2)) * 0.01

for i in range(len(percent_value)):

    dist_mat_2 = act_dist_calc.act_sim_percent(tracefilter_log_45000, tracefilter_log_22000, percent_value[i],
                                               percent_value[i])
    dist_value_2.append(dist_mat_2)
    print("2:", dist_value_2)

    dist_mat_3 = act_dist_calc.act_sim_percent(tracefilter_log_12000, tracefilter_log_9000, percent_value[i],
                                               percent_value[i])
    dist_value_3.append(dist_mat_3)
    print("3:", dist_value_3)

    dist_mat_4 = act_dist_calc.act_sim_percent(tracefilter_log_8000, tracefilter_log_9000, percent_value[i],
                                               percent_value[i])
    dist_value_4.append(dist_mat_4)
    print("4:", dist_value_4)

    dist_mat_5 = act_dist_calc.act_sim_percent(tracefilter_log_12000, tracefilter_log_8000, percent_value[i],
                                               percent_value[i])
    dist_value_5.append(dist_mat_5)
    print("5:", dist_value_5)

    dist_mat_6 = suc_dist_calc.suc_sim_percent(tracefilter_log_8000, tracefilter_log_9000, percent_value[i],
                                               percent_value[i])
    dist_value_6.append(dist_mat_6)
    print("6:", dist_value_6)

for i in range(len(percent_value)):
    dist_mat_7 = suc_dist_calc.suc_sim_percent(tracefilter_log_9000, tracefilter_log_12000, percent_value[i],
                                               percent_value[i])
    dist_value_7.append(dist_mat_7)
    print("7:", dist_value_7)

    dist_mat_5 = suc_dist_calc.suc_sim_percent(tracefilter_log_12000, tracefilter_log_9000, percent_value[i],1)
    dist_value_5.append(dist_mat_5)
    print("5:", dist_value_5)

    dist_mat_6 = suc_dist_calc.suc_sim_percent(tracefilter_log_9000, tracefilter_log_12000, percent_value[i], 1)
    dist_value_6.append(dist_mat_6)
    print("6:", dist_value_6)





#plt.yscale('log')
plt.plot(percent_value,np.array(dist_value_5))
plt.show()
plt.plot(percent_value,np.array(dist_value_6))
plt.show()
plt.plot(percent_value,np.array(dist_value_7))
plt.show()
'''


'''
print("1:",dist_value_1)
print("2:",dist_value_2)
print("3:",dist_value_3)
#plt.yscale('log')
plt.plot(percent_value,np.array(dist_value_1))
plt.show()
plt.plot(percent_value,np.array(dist_value_2))
plt.show()
plt.plot(percent_value,np.array(dist_value_3))
plt.show()
'''
