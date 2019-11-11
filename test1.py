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


# log = xes_importer.apply("/home/yukun/dataset/document_logs/Filtered_Geo.xes")
# # sublog2 = xes_importer.apply("/home/yukun/resultlog/Payment_application/amount_applied0/log_4_0_dfgamount_applied0.xes")
# #log = parquet_importer.import_minimal_log("bpic2017_application.parquet")
# print("imported")
# # net, im, fm = inductive_miner.apply(log)
# for i in range(0,7):
#     petripath = '/home/yukun/dataset/document_logs/'+str(i)+'.pnml'
#     net, im, fm = petri_importer.apply(petripath)
#     # gviz = pn_vis_factory.apply(net, im, fm)
#     # pn_vis_factory.view(gviz)
#     print("calculated model")
#     wrapper = remote_wrapper_factory.apply("137.226.117.71", "5001", "hello", "DUMMYDUMMY")
#     aa = time.time()
#     # fitness = wrapper.calculate_fitness_with_tbr(net, im, fm, log)
#     fitness = wrapper.calculate_fitness_with_alignments(net, im, fm, log, parameters={"align_variant": "state_equation_a_star"})[
#         'averageFitness']
#     bb = time.time()
#     print("fit",fitness)
#     # fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
#     # print("local",fitness)
#
#     precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
#     print("prec", precision)
#     F1 = 2 * fitness * precision / (fitness + precision)
#     print("F1",F1)


# precision = precision_factory.apply(log, net, im, fm)
# print("local",precision)
log = xes_importer.apply("/home/yukun/dataset/Receipt.xes")
sublog = xes_importer.apply("/home/yukun/resultlog/Receipt/responsible/log_4_0_dfgresponsible.xes")
# sublog = xes_importer.apply("/home/yukun/dataset/sublog_receipt.xes")
print("imported")
net, im, fm = inductive_miner.apply(sublog)
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
# fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
# print("local",fitness)
precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
print("prec",precision)

fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
print("local",fitness)

precision = precision_factory.apply(log, net, im, fm)
print("prec",precision)


sublog = xes_importer.apply("/home/yukun/resultlog/Receipt/responsible/log_4_1_dfgresponsible.xes")
# sublog = xes_importer.apply("/home/yukun/dataset/sublog_receipt.xes")
print("imported")
net, im, fm = inductive_miner.apply(sublog)
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
# fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
# print("local",fitness)
precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
print("prec",precision)

fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
print("local",fitness)

precision = precision_factory.apply(log, net, im, fm)
print("prec",precision)


sublog = xes_importer.apply("/home/yukun/resultlog/Receipt/responsible/log_4_2_dfgresponsible.xes")
# sublog = xes_importer.apply("/home/yukun/dataset/sublog_receipt.xes")
print("imported")
net, im, fm = inductive_miner.apply(sublog)
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
# fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
# print("local",fitness)
precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
print("prec",precision)

fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
print("local",fitness)

precision = precision_factory.apply(log, net, im, fm)
print("prec",precision)

sublog = xes_importer.apply("/home/yukun/resultlog/Receipt/responsible/log_4_3_dfgresponsible.xes")
# sublog = xes_importer.apply("/home/yukun/dataset/sublog_receipt.xes")
print("imported")
net, im, fm = inductive_miner.apply(sublog)
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
# fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
# print("local",fitness)
precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
print("prec",precision)

fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
print("local",fitness)

precision = precision_factory.apply(log, net, im, fm)
print("prec",precision)
#
#
# log = xes_importer.apply("/home/yukun/dataset/document_logs/Filtered_Geo.xes")
# print("imported")
# net, im, fm = inductive_miner.apply(log)
# # net, im, fm = petri_importer.apply("/home/yukun/dataset/sublog2.pnml")
# #gviz = pn_vis_factory.apply(net, im, fm)
# #pn_vis_factory.view(gviz)
# print("calculated model")
# wrapper = remote_wrapper_factory.apply("137.226.117.71", "5001", "hello", "DUMMYDUMMY")
# aa = time.time()
# # fitness = wrapper.calculate_fitness_with_tbr(net, im, fm, log)
# fitness = wrapper.calculate_fitness_with_alignments(net, im, fm, log, parameters={"align_variant": "state_equation_a_star"})['averageFitness']
# bb = time.time()
# print(fitness)
# # fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
# # print("local",fitness)
#
# precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
# print("cluster",precision)
#
# log = xes_importer.apply("/home/yukun/dataset/document_logs/Filtered_Inspection.xes")
# print("imported")
# net, im, fm = inductive_miner.apply(log)
# # net, im, fm = petri_importer.apply("/home/yukun/dataset/sublog2.pnml")
# #gviz = pn_vis_factory.apply(net, im, fm)
# #pn_vis_factory.view(gviz)
# print("calculated model")
# wrapper = remote_wrapper_factory.apply("137.226.117.71", "5001", "hello", "DUMMYDUMMY")
# aa = time.time()
# # fitness = wrapper.calculate_fitness_with_tbr(net, im, fm, log)
# fitness = wrapper.calculate_fitness_with_alignments(net, im, fm, log, parameters={"align_variant": "state_equation_a_star"})['averageFitness']
# bb = time.time()
# print(fitness)
# # fitness = replay_factory.apply(log, net, im, fm, variant="alignments")['averageFitness']
# # print("local",fitness)
#
# precision = wrapper.calculate_precision_with_tbr(net, im, fm, log)
# print("cluster",precision)

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
