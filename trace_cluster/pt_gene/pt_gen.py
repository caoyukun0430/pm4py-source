import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
from pm4py.objects.log.importer.xes import factory as xes_importer
import filter_subsets
from trace_cluster.variant import act_dist_calc, suc_dist_calc
import time


def openAllXes(path,cate,iter):
    loglist = []

    for i in range(1,cate+1):
        for j in range(1,iter+1):
            log = xes_importer.apply(path + '\\log_1_'+str(i)+'_'+str(j) + ".xes")
            print(path + '\\log_1_'+str(i)+'_'+str(j) + ".xes")
            print(filter_subsets.sublog_percent(log, 1))
            loglist.append(log)

    return loglist



if __name__ == "__main__":


    '''
    log_1 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_1_1.xes")
    log_2 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_2_1.xes")
    log_3 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_3_1.xes")
    log_4 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_4_1.xes")
    log_5 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_1_2.xes")
    log_6 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_2_2.xes")
    log_7 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_3_2.xes")
    log_8 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_4_2.xes")

    print(filter_subsets.sublog_percent(log_1, 1))
    print(filter_subsets.sublog_percent(log_2, 1))
    print(filter_subsets.sublog_percent(log_3, 1))
    print(filter_subsets.sublog_percent(log_4, 1))
    print(filter_subsets.sublog_percent(log_5, 1))
    print(filter_subsets.sublog_percent(log_6, 1))
    print(filter_subsets.sublog_percent(log_7, 1))
    print(filter_subsets.sublog_percent(log_8, 1))

    #loglist = [log_1, log_2, log_3, log_4]
    loglist = [log_1,log_5,log_2,log_6,log_3,log_7,log_4,log_8]'''

    percent = 1
    alpha = 0.5
    loglist = openAllXes("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs",4,1)

    #log1 = variants_filter.filter_by_variants_percentage(loglist[0],percentage=0.2)

    #filter_subsets.sublog_percent(loglist[0],0.5)
    #(logl,freq)=filter_subsets.logslice_percent(loglist[0],1)
    #print(freq)
    '''
    varlist=['a,b,c,d,e,f,g', 'a,b,c,d,e,h']
    log1 = variants_filter.apply(loglist[0],varlist,parameters={"positive": True})
    variants_count = case_statistics.get_variant_statistics(log1)
    print(variants_count)
    '''




    lists = loglist
    size = len(lists)
    print(size)
    dist_mat = np.zeros((size, size))
    #sim_act = suc_dist_calc.suc_sim_percent(lists[0], lists[1], percent, percent)
    #print(sim_act)

    start = time.time()
    for i in range(0, size - 1):
        for j in range(i + 1, size):
            sim_act = act_dist_calc.act_sim_percent(lists[i], lists[j], percent, percent)
            #print([i, len(lists[i]), j, len(lists[j]), sim_act])

            sim_suc = suc_dist_calc.suc_sim_percent(lists[i], lists[j], percent, percent)
            #print([i, len(lists[i]), j, len(lists[j]), sim_suc])
            print([i,j, sim_act,sim_suc])

            # sim_suc = suc_dist_calc.suc_sim(lists[i], lists[j], lists[i+size],
            #                                lists[j+size],
            #                                freq, num, parameters={"single": True})
            # dist_mat[i][j] = sim_act
            dist_mat[i][j] = (sim_act * alpha + sim_suc * (1 - alpha))
            dist_mat[j][i] = dist_mat[i][j]

    end = time.time()
    print(end - start)
    print(dist_mat)

    y = squareform(dist_mat)
    print(y)
    Z = linkage(y, method='average')
    print(Z)
    print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    fig = plt.figure(figsize=(10, 8))
    dn = dendrogram(Z)
    #dn = dendrogram(Z)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.savefig('cluster.svg')
    plt.show()
