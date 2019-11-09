import pm4pycvxopt
from pm4pydistr.remote_wrapper import factory as remote_wrapper_factory
from pm4py.evaluation.precision import factory as precision_factory
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.importer.parquet import factory as parquet_importer
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.objects.petri.importer import factory as petri_importer
from pm4py.evaluation.replay_fitness import factory as replay_factory
import time

def get_fit_prec_hpc(log):
    net, im, fm = inductive_miner.apply(log)
    wrapper = remote_wrapper_factory.apply("137.226.117.71", "5001", "hello", "DUMMYDUMMY")

    fitness = wrapper.calculate_fitness_with_tbr(net, im, fm, log)

    precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)

    return fitness,precision


log = xes_importer.apply("/home/yukun/dataset/filteredbpic2017.xes")
sublog2 = xes_importer.apply("/home/yukun/pm4py-source/log_3_1_DMMCreditScore.xes")
#log = parquet_importer.import_minimal_log("bpic2017_application.parquet")
print("imported")
net, im, fm = inductive_miner.apply(sublog2)
# net, im, fm = petri_importer.apply("/home/yukun/dataset/sublog2.pnml")
#gviz = pn_vis_factory.apply(net, im, fm)
#pn_vis_factory.view(gviz)
print("calculated model")
wrapper = remote_wrapper_factory.apply("137.226.117.71", "5001", "hello", "DUMMYDUMMY")
aa = time.time()
# fitness = wrapper.calculate_fitness_with_tbr(net, im, fm, log)
fitness = wrapper.calculate_fitness_with_alignments(net, im, fm, log, parameters={"align_variant": "state_equation_a_star"})['averageFitness']
bb = time.time()
print(fitness)
fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
print("local",fitness)

precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
print("cluster",precision)

precision = precision_factory.apply(log, net, im, fm)
print("local",precision)
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