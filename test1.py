from pm4pydistr.remote_wrapper import factory as remote_wrapper_factory
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.importer.parquet import factory as parquet_importer
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.objects.petri.importer import factory as petri_importer
import time

log = xes_importer.apply("/home/yukun/dataset/BPIC2017.xes")
#log = parquet_importer.import_minimal_log("bpic2017_application.parquet")
print("imported")
#net, im, fm = inductive_miner.apply(log)
net, im, fm = petri_importer.apply("calculatedonprom.pnml")
#gviz = pn_vis_factory.apply(net, im, fm)
#pn_vis_factory.view(gviz)
print("calculated model")
wrapper = remote_wrapper_factory.apply("137.226.117.71", "5001", "hello", "DUMMYDUMMY")
aa = time.time()
fitness = wrapper.calculate_fitness_with_tbr(net, im, fm, log)
bb = time.time()
print(fitness)

precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
print(precision)

"""
wrapper = remote_wrapper_factory.apply("137.226.117.71", "5001", "hello", "DUMMYDUMMY")
aa = time.time()
fitness = wrapper.calculate_fitness_with_alignments(net, im, fm, log)
bb = time.time()
print(fitness, (bb-aa))
aa = time.time()
precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
bb = time.time()
print(precision, (bb-aa))
"""
