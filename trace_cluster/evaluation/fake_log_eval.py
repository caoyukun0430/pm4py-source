import pm4pycvxopt
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
import numpy as np
from trace_cluster.pt_gene import pt_gen
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.evaluation.replay_fitness import factory as replay_factory
from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.evaluation.precision import factory as precision_factory
from trace_cluster.variant import act_dist_calc
from trace_cluster.variant import suc_dist_calc
from trace_cluster.leven_dist import leven_dist_calc
from trace_cluster.merge_log import merge_log
from pm4py.statistics.traces.log import case_statistics
from trace_cluster.dfg import dfg_dist

def dfg_dis(loglist, percent, alpha,list_of_vals):
    size = len(loglist)
    # print(size)
    dist_mat = np.zeros((size, size))

    for i in range(0, size - 1):
        for j in range(i + 1, size):
            (dist_act, dist_dfg) = dfg_dist.dfg_dist_calc(loglist[i], loglist[j])
            print([i, j, dist_act, dist_dfg])
            dist_mat[i][j] = dist_act * alpha + dist_dfg * (1 - alpha)
            dist_mat[j][i] = dist_mat[i][j]

    # print(dist_mat)

    y = squareform(dist_mat)
    # print(y)
    # Z = linkage(y, method='average')
    # print(Z)
    # print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    # fig = plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # # dn = dendrogram(Z)
    # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Credit Score')
    # plt.ylabel('Distance')
    # plt.savefig('cluster.svg')
    # plt.show()
    return y


def eval_avg_variant(loglist, percent, alpha):

    size = len(loglist)
    #print(size)
    dist_mat = np.zeros((size, size))

    for i in range(0, size - 1):
        for j in range(i + 1, size):
            dist_act = act_dist_calc.act_sim_percent_avg(loglist[i], loglist[j],percent,percent)
            dist_suc = suc_dist_calc.suc_sim_percent_avg(loglist[i], loglist[j], percent, percent)
            print([i,j,dist_act,dist_suc])
            dist_mat[i][j] = dist_act * alpha + dist_suc * (1 - alpha)
            dist_mat[j][i] = dist_mat[i][j]

    # print(dist_mat)

    y = squareform(dist_mat)
    # print(y)
    # Z = linkage(y, method='average')
    # print(Z)
    # print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # dn = dendrogram(Z)
    # # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Case Index')
    # plt.ylabel('Distance')
    # plt.savefig('cluster.svg')
    # plt.show()

    return y


def eval_DMM_variant(loglist, percent, alpha):

    size = len(loglist)
    #print(size)
    dist_mat = np.zeros((size, size))
    print("stop1")

    for i in range(0, size - 1):
        for j in range(i + 1, size):
            print("stop2")
            dist_act = act_dist_calc.act_sim_percent(loglist[i], loglist[j],percent,percent)
            print("stop3")
            dist_suc = suc_dist_calc.suc_sim_percent(loglist[i], loglist[j], percent, percent)
            print([i, j, dist_act, dist_suc])
            dist_mat[i][j] = dist_act * alpha + dist_suc * (1 - alpha)
            dist_mat[j][i] = dist_mat[i][j]

    # print(dist_mat)

    y = squareform(dist_mat)
    # print(y)
    # Z = linkage(y, method='average')
    # print(Z)
    # print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    # plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # dn = dendrogram(Z)
    # # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Case Index')
    # plt.ylabel('Distance')
    # plt.savefig('cluster.svg')
    # plt.show()
    return y

def eval_avg_leven(loglist, percent, alpha):

    size = len(loglist)
    #print(size)
    dist_mat = np.zeros((size, size))

    for i in range(0, size - 1):
        for j in range(i + 1, size):
            dist_mat[i][j] = leven_dist_calc.leven_dist_avg(loglist[i], loglist[j], percent, percent)
            dist_mat[j][i] = dist_mat[i][j]

    print(dist_mat)

    y = squareform(dist_mat)
    print(y)
    Z = linkage(y, method='average')
    print(Z)
    print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    plt.figure(figsize=(10, 8))
    # dn = fancy_dendrogram(Z, max_d=0.35)
    dn = dendrogram(Z)
    # dn = dendrogram(Z,labels=np.array(list_of_vals))
    plt.title('Hierarchical Clustering Dendrogram')
    #plt.xlabel('Loan Amount')
    plt.ylabel('Distance')
    plt.savefig('cluster.svg')
    plt.show()
    return y

def eval_DMM_leven(loglist, percent, alpha):

    size = len(loglist)
    #print(size)
    dist_mat = np.zeros((size, size))

    for i in range(0, size - 1):
        for j in range(i + 1, size):
            dist_mat[i][j] = leven_dist_calc.leven_dist(loglist[i], loglist[j], percent, percent)
            dist_mat[j][i] = dist_mat[i][j]

    print(dist_mat)

    y = squareform(dist_mat)
    print(y)
    # print((y[1]+y[3])/2)
    # print((y[2] + y[4]) / 2)
    Z = linkage(y, method='average')
    print(Z)
    print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    plt.figure(figsize=(10, 8))
    # dn = fancy_dendrogram(Z, max_d=0.35)
    dn = dendrogram(Z)
    # dn = dendrogram(Z,labels=np.array(list_of_vals))
    plt.title('Hierarchical Clustering Dendrogram')
    #plt.xlabel('Loan Amount')
    plt.ylabel('Distance')
    plt.savefig('cluster.svg')
    plt.show()
    return y

if __name__ == "__main__":

    # LOG_PATH = "D:\\Sisc\\19SS\\thesis\\Dataset\\BPIC2017\\bpic2017.xes"
    log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPIC2017\\filteredbpic2017.xes")
    METHOD = 'dfg'
    ATTR_NAME = 'RequestedAmount'
    # sublog = xes_importer.apply(
    #         "D:\\Sisc\\19SS\\thesis\\Dataset\\resultlog6\\log_6_2_dfgRequestedAmount.xes")
    filename = 'D:/Sisc/19SS/thesis/Dataset/BPIC2017/' + ATTR_NAME + '/' + 'log' + '_'+ METHOD + ATTR_NAME + '.png'
    # xes_exporter.export_log(log, filename)
    inductive_petri, inductive_initial_marking, inductive_final_marking = inductive_miner.apply(log)
    fitness_inductive = replay_factory.apply(log, inductive_petri, inductive_initial_marking, inductive_final_marking)
    print("fitness_inductive=", fitness_inductive)

    gviz = pn_vis_factory.apply(inductive_petri, inductive_initial_marking, inductive_final_marking)
    pn_vis_factory.save(gviz,filename)

    # precision_inductive = precision_factory.apply(log, inductive_petri, inductive_initial_marking,
    #                                               inductive_final_marking)
    # print("precision_inductive=", precision_inductive)


    # percent = 1
    # alpha = 0.5
    # # loglist = pt_gen.openAllXes("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 1)
    # (loglist, mergedlog) = merge_log.merge_log("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 1)
    #
    # #dist = suc_dist_calc.suc_sim_percent_avg(loglist[2], loglist[3], 1, 1)
    # # print(dist)
    #
    # eval_DMM_leven(loglist,percent,alpha)
    # #eval_DMM_variant(loglist, percent, alpha)
    #
    # update_loglist = [loglist[0],loglist[1]]
    #
    # # print(case_statistics.get_variant_statistics(loglist[0]))
    # # print(case_statistics.get_variant_statistics(loglist[1]))
    # print("len", len(loglist))
    # merged1 = merge_log.update_merge(update_loglist)
    # print(case_statistics.get_variant_statistics(merged1))
    # update_loglist = [loglist[2], loglist[3]]
    # merged2 = merge_log.update_merge(update_loglist)
    # # update_loglist = [loglist[4], loglist[5]]
    # # merged3 = merge_log.update_merge(update_loglist)
    # # update_loglist = [loglist[6], loglist[7]]
    # # merged4 = merge_log.update_merge(update_loglist)
    # # dist2= leven_dist_calc.leven_dist(merged1, loglist[2], percent, percent)
    # # dist3 = leven_dist_calc.leven_dist(merged1, loglist[3], percent, percent)
    # # dist4 = 0.5*act_dist_calc.act_sim_percent_avg(merged1, merged2, percent, percent)+0.5*suc_dist_calc.suc_sim_percent_avg(merged1, merged2, percent, percent)
    # dist4 = leven_dist_calc.leven_dist(merged1, loglist[2], percent, percent)
    # dist5 = leven_dist_calc.leven_dist(merged1, loglist[3], percent, percent)
    # dist6 = leven_dist_calc.leven_dist(merged1, merged2, percent, percent)
    # # print("dist2",dist2)
    # # print("dist3", dist3)
    # print("dist4", dist4)
    # print("dist4", dist5)
    # print("dist6", dist6)






