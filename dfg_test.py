import pandas as pd
import numpy as np
import time
import act_dist_calc
import filter_subsets
import sim_calc
from IPython.display import display
from scipy.spatial.distance import pdist
from collections import Counter
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.log import EventLog, Trace, EventStream
from pm4py.util.constants import PARAMETER_CONSTANT_ATTRIBUTE_KEY, PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util import constants
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.log import case_statistics
from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory
from pm4py.algo.discovery.dfg.versions import native

log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPI_Challenge_2012.xes")
list_of_vals = ['5000', '7000']
tracefilter_log = filter_subsets.apply_trace_attributes(log, list_of_vals,
                                                            parameters={
                                                            constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                "positive": True})
tracefilter_log_7000 = filter_subsets.apply_trace_attributes(tracefilter_log, ['5000'],
                                                                 parameters={
                                                                constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                                     "positive": True})
suc = native.apply(tracefilter_log_7000)
print(suc)
dfg = dfg_factory.apply(tracefilter_log_7000)
print(len(dfg))
print(suc==dfg)
parameters = {"format":"svg"}
gviz = dfg_vis_factory.apply(dfg, log=tracefilter_log_7000, variant="frequency",parameters=parameters)
dfg_vis_factory.view(gviz)
dfg_vis_factory.save(gviz, "dfg.svg")
