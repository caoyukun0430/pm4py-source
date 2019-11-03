import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
from trace_cluster.leven_dist import leven_dist_calc
from trace_cluster.merge_log import merge_log
from trace_cluster.evaluation import fake_log_eval
from trace_cluster.dfg import dfg_dist
from trace_cluster.variant import act_dist_calc
from trace_cluster.variant import suc_dist_calc


def linkage_dfg_update(loglist, dist_mat,alpha,percent):
    index_list = []
    for i in range(len(dist_mat)):
        for j in range(i + 1, len(dist_mat)):
            index_list.append([i, j])

    y = squareform(dist_mat)
    n = len(dist_mat)  # The number of observations.
    Z = []
    cluster_size = dict(zip(range(n), np.ones(n)))  # record merged cluster size every step
    k = 1
    logsindex = list(range(len(loglist)))
    while (k <= n - 2):
        min_index = np.argmin(y)

        # update Z
        temp = []
        temp.extend(index_list[min_index])
        temp.append(y[min_index])
        cluster_size[n - 1 + k] = cluster_size[temp[0]] + cluster_size[temp[1]]
        temp.append(cluster_size[n - 1 + k])

        Z.append(temp)

        # get index of min in y
        item = index_list[min_index][::]
        record1 = []
        record2 = []
        for ele in index_list:
            if item[0] in ele:
                record1.append(index_list.index(ele))
                inde = ele.index(item[0])
                ele[inde] = n - 1 + k
            if item[1] in ele:  # here if/elif both works
                record2.append(index_list.index(ele))
                inde = ele.index(item[1])
                ele[inde] = n - 1 + k
            ele.sort()

        record = list(set(record1).union(set(record2)))

        merged1 = merge_log.update_merge([loglist[item[0]], loglist[item[1]]])
        # here the logsindex is changing
        diff = list(set(logsindex).difference(set(item)))  # diff is the node number need to be updated

        update_dist = dict()
        for ele in diff:
            (dist_act, dist_dfg) = dfg_dist.dfg_dist_calc(merged1, loglist[ele])
            tempdist =dist_act * alpha + dist_dfg * (1 - alpha)
            # tempdist = leven_dist_calc.leven_dist_avg(merged1, loglist[ele], percent, percent)
            update_dist[ele] = tempdist

        loglist.append(merged1)
        diff.append(n - 1 + k)
        logsindex = diff

        del (record1[record1.index(min_index)])
        del (record2[record2.index(min_index)])

        # for i in range(len(record1)):
        #     y[record1[i]] = (y[record1[i]]*cluster_size[item[0]] + y[record2[i]]*cluster_size[item[1]]) / (cluster_size[item[0]]+cluster_size[item[1]])
        for ele in record1:
            uindex = index_list[ele][0]  # record1 is the location if nodes in diff in the index_list
            y[ele] = update_dist[uindex]

        diff1 = list(set(range(len(index_list))).difference(set(record)))
        newindex = record1 + diff1
        newindex.sort()

        range_newindex = range(len(newindex))
        tempy = list(range_newindex)
        templist = list(range_newindex)
        for i in range_newindex:
            tempy[i] = y[newindex[i]]
            templist[i] = index_list[newindex[i]]

        index_list = templist
        y = tempy
        k = k + 1

    temp = []
    temp.extend(index_list[0])
    temp.append(y[0])

    cluster_size[n - 1 + k] = cluster_size[temp[0]] + cluster_size[temp[1]]
    temp.append(cluster_size[n - 1 + k])
    Z.append(temp)

    return Z

def linkage_avg(loglist, dist_mat):
    index_list = []
    cluster_size = []
    for i in range(len(dist_mat)):
        cluster_size.append(len(loglist[i]))
        for j in range(i + 1, len(dist_mat)):
            index_list.append([i, j])

    y = squareform(dist_mat)
    n = len(dist_mat)  # The number of observations.
    Z = []
    cluster_size = dict(zip(range(n), cluster_size))  # record merged cluster size every step
    k = 1
    while (k <= n - 2):
        min_index = np.argmin(y)

        # update Z
        temp = []
        temp.extend(index_list[min_index])
        temp.append(y[min_index])
        cluster_size[n - 1 + k] = cluster_size[temp[0]] + cluster_size[temp[1]]
        temp.append(cluster_size[n - 1 + k])

        Z.append(temp)

        # get index of min in y
        item = index_list[min_index][::]
        record1 = []
        record2 = []
        for ele in index_list:
            if item[0] in ele:
                record1.append(index_list.index(ele))
                inde = ele.index(item[0])
                ele[inde] = n - 1 + k
            if item[1] in ele:  # here if/elif both works
                record2.append(index_list.index(ele))
                inde = ele.index(item[1])
                ele[inde] = n - 1 + k
            ele.sort()

        record = list(set(record1).union(set(record2)))

        del (record1[record1.index(min_index)])
        del (record2[record2.index(min_index)])

        for i in range(len(record1)):
            y[record1[i]] = (y[record1[i]]*cluster_size[item[0]] + y[record2[i]]*cluster_size[item[1]]) / (cluster_size[item[0]]+cluster_size[item[1]])
        # for ele in record1:
        #     uindex = index_list[ele][0]  # record1 is the location if nodes in diff in the index_list
        #     y[ele] = update_dist[uindex]

        diff1 = list(set(range(len(index_list))).difference(set(record)))
        newindex = record1 + diff1
        newindex.sort()

        range_newindex = range(len(newindex))
        tempy = list(range_newindex)
        templist = list(range_newindex)
        for i in range_newindex:
            tempy[i] = y[newindex[i]]
            templist[i] = index_list[newindex[i]]

        index_list = templist
        y = tempy
        k = k + 1

    temp = []
    temp.extend(index_list[0])
    temp.append(y[0])

    cluster_size[n - 1 + k] = cluster_size[temp[0]] + cluster_size[temp[1]]
    temp.append(cluster_size[n - 1 + k])
    Z.append(temp)

    return Z

def linkage_DMM_update(loglist, dist_mat,alpha,percent):
    index_list = []
    for i in range(len(dist_mat)):
        for j in range(i + 1, len(dist_mat)):
            index_list.append([i, j])

    y = squareform(dist_mat)
    n = len(dist_mat)  # The number of observations.
    Z = []
    cluster_size = dict(zip(range(n), np.ones(n)))  # record merged cluster size every step
    k = 1
    logsindex = list(range(len(loglist)))
    while (k <= n - 2):
        min_index = np.argmin(y)

        # update Z
        temp = []
        temp.extend(index_list[min_index])
        temp.append(y[min_index])
        cluster_size[n - 1 + k] = cluster_size[temp[0]] + cluster_size[temp[1]]
        temp.append(cluster_size[n - 1 + k])

        Z.append(temp)

        # get index of min in y
        item = index_list[min_index][::]
        record1 = []
        record2 = []
        for ele in index_list:
            if item[0] in ele:
                record1.append(index_list.index(ele))
                inde = ele.index(item[0])
                ele[inde] = n - 1 + k
            if item[1] in ele:  # here if/elif both works
                record2.append(index_list.index(ele))
                inde = ele.index(item[1])
                ele[inde] = n - 1 + k
            ele.sort()

        record = list(set(record1).union(set(record2)))

        merged1 = merge_log.update_merge([loglist[item[0]], loglist[item[1]]])
        # here the logsindex is changing
        diff = list(set(logsindex).difference(set(item)))  # diff is the node number need to be updated

        update_dist = dict()
        for ele in diff:
            dist_act = act_dist_calc.act_sim_percent(merged1, loglist[ele], percent, percent)
            dist_suc = suc_dist_calc.suc_sim_percent(merged1, loglist[ele], percent, percent)
            tempdist =dist_act * alpha + dist_suc * (1 - alpha)
            # tempdist = leven_dist_calc.leven_dist_avg(merged1, loglist[ele], percent, percent)
            update_dist[ele] = tempdist

        loglist.append(merged1)
        diff.append(n - 1 + k)
        logsindex = diff

        del (record1[record1.index(min_index)])
        del (record2[record2.index(min_index)])

        # for i in range(len(record1)):
        #     y[record1[i]] = (y[record1[i]]*cluster_size[item[0]] + y[record2[i]]*cluster_size[item[1]]) / (cluster_size[item[0]]+cluster_size[item[1]])
        for ele in record1:
            uindex = index_list[ele][0]  # record1 is the location if nodes in diff in the index_list
            y[ele] = update_dist[uindex]

        diff1 = list(set(range(len(index_list))).difference(set(record)))
        newindex = record1 + diff1
        newindex.sort()

        range_newindex = range(len(newindex))
        tempy = list(range_newindex)
        templist = list(range_newindex)
        for i in range_newindex:
            tempy[i] = y[newindex[i]]
            templist[i] = index_list[newindex[i]]

        index_list = templist
        y = tempy
        k = k + 1

    temp = []
    temp.extend(index_list[0])
    temp.append(y[0])

    cluster_size[n - 1 + k] = cluster_size[temp[0]] + cluster_size[temp[1]]
    temp.append(cluster_size[n - 1 + k])
    Z.append(temp)

    return Z

if __name__ == "__main__":
    percent = 1
    alpha = 0.5
    # loglist = pt_gen.openAllXes("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 1)
    (loglist, mergedlog) = merge_log.merge_log("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 3)
    loglisttemp = loglist[::]

    y= fake_log_eval.eval_avg_leven(loglist,percent,alpha)

    # y=[0.00124115,0.00024129, 0.00124275,0.00099058, 0.00192228,0.00054379,
    #  0.00806061, 0.00079721,0.00022188,0.00065901, 0.0055589,
    # 0.00799296,0.00857136,0.00882324, 0.0003645,
    # 0.00063645,0.00168869,0.00590032,
    #  0.00051635,0.00604253,
    # 0.00618876]
    # y= [0.00024129 ,0.00124275 ,0.00099058, 0.00192228 ,0.00054379 ,0.00668401,
    #  0.00799296 ,0.00857136, 0.00882324, 0.0003645 , 0.01116678, 0.00063645,
    #  0.00168869, 0.00590032 ,0.00657356 ,0.00051635 ,0.00604253, 0.00077648,
    #  0.00618876 ,0.00235414, 0.00681441]

    # dist_mat = [[0., 0.00200144, 0.01357403, 0.0084349, 0.00392522, 0.012375],
    #             [0.00200144, 0., 0.00893385, 0.00513085, 0.00127037, 0.01103749],
    #             [0.01357403, 0.00893385, 0., 0.00154929, 0.00766808, 0.00339415],
    #             [0.0084349, 0.00513085, 0.00154929, 0., 0.00387665, 0.00554735],
    #             [0.00392522, 0.00127037, 0.00766808, 0.00387665, 0., 0.01107205],
    #             [0.012375, 0.01103749, 0.00339415, 0.00554735, 0.01107205, 0.]]

    # dist_mat = [[0.    ,     0.00031656 ,0.0005384 , 0.00119314 ,0.00079948, 0.00076642],
    #  [0.00031656, 0.       ,  0.00021545 ,0.00053412, 0.00035174, 0.00030837],
    #  [0.0005384 , 0.00021545, 0.      ,   0.00101519 ,0.00063727 ,0.00053139],
    #  [0.00119314, 0.00053412, 0.00101519, 0.      ,   0.00090698, 0.00103637],
    #  [0.00079948, 0.00035174 ,0.00063727, 0.00090698 ,0.        , 0.00062803],
    #  [0.00076642, 0.00030837, 0.00053139, 0.00103637, 0.00062803, 0.        ]]
    # y= squareform(dist_mat)
    # Z = linkage(y, method='average')
    # print("Z",Z)
    #
    dist_mat= squareform(y)
    Z=linkage_avg(loglist,dist_mat)
    print("Zupdate",Z)

    loglist = loglisttemp
    Z = linkage_dfg_update(loglist, dist_mat,alpha,percent)
    print("Zupdate2", Z)
    # index_list = []
    # cluster_size =[]
    # for i in range(len(dist_mat)):
    #     cluster_size.append(len(loglist[i]))
    #     for j in range(i + 1, len(dist_mat)):
    #         index_list.append([i, j])
    #
    # n = len(dist_mat)  # The number of observations.
    # Z = []
    # cluster_size = dict(zip(range(n), cluster_size))  # record merged cluster size every step
    # print(cluster_size)
    #
    # print([0, index_list])
    # k = 1
    # logsindex = list(range(len(loglist)))
    # print("len(loglist)",len(loglist))
    #
    # while (k <= n - 2):
    #     print("y",y)
    #     min_index = np.argmin(y)
    #     # print("min_index",[k,index_list[min_index]])
    #     # print(index_list)
    #
    #     # update Z
    #     temp = []
    #     temp.extend(index_list[min_index])
    #     temp.append(y[min_index])
    #
    #     cluster_size[n - 1 + k] = cluster_size[temp[0]] + cluster_size[temp[1]]
    #     temp.append(cluster_size[n - 1 + k])
    #     # print(temp)
    #     Z.append(temp)
    #     # print(cluster_size)
    #
    #     item = index_list[min_index][::]
    #     record1 = []
    #     record2 = []
    #     for ele in index_list:
    #         if item[0] in ele:
    #             record1.append(index_list.index(ele))
    #             inde = ele.index(item[0])
    #             ele[inde] = n - 1 + k
    #         if item[1] in ele:  # here if/elif both works
    #             record2.append(index_list.index(ele))
    #             inde = ele.index(item[1])
    #             ele[inde] = n - 1 + k
    #         ele.sort()
    #
    #     # print(index_list)
    #
    #     record = list(set(record1).union(set(record2)))
    #     print("record", [k, record])
    #
    #     merged1 = merge_log.update_merge([loglist[item[0]], loglist[item[1]]])
    #     # # here the logsindex is changing
    #     diff = list(set(logsindex).difference(set(item)))# diff is the node number need to be updated
    #     # print("diff",diff)
    #     update_dist=dict()
    #
    #     for ele in diff:
    #         tempdist = leven_dist_calc.leven_dist_avg(merged1, loglist[ele], percent, percent)
    #         # print(tempdist)
    #         update_dist[ele]=tempdist
    #
    #     # print("update_dist",[k,update_dist])
    #     loglist.append(merged1)
    #     diff.append(n - 1 + k)
    #     logsindex = diff
    #     # # print(logsindex)
    #
    #     del (record1[record1.index(min_index)])
    #     del (record2[record2.index(min_index)])
    #
    #     # print("yold",y)
    #     print("record1", record1)
    #     print("record2", record2)
    #     print("index_list", index_list)
    #     for i in range(len(record1)):
    #         y[record1[i]] = (y[record1[i]]*cluster_size[item[0]] + y[record2[i]]*cluster_size[item[1]]) / (cluster_size[item[0]]+cluster_size[item[1]])
    #     # for ele in record1:
    #     #     uindex = index_list[ele][0] #record1 is the location if nodes in diff in the index_list
    #     #     # print("uindex",uindex)
    #     #     y[ele] = update_dist[uindex]
    #
    #     diff1 = list(set(range(len(index_list))).difference(set(record)))
    #     # print("ynew", y)
    #     newindex = record1 + diff1
    #     newindex.sort()
    #
    #     rang = range(len(newindex))
    #     # print("new", newindex)
    #     tempy = list(rang)
    #     templist = list(rang)
    #     for i in rang:
    #         tempy[i] = y[newindex[i]]
    #         templist[i] = index_list[newindex[i]]
    #
    #     index_list = templist
    #     y = tempy
    #     # y[min_index]=1
    #     # print("index_list",[k,index_list])
    #     k = k + 1
    #
    # temp = []
    # temp.extend(index_list[0])
    # temp.append(y[0])
    #
    # cluster_size[n - 1 + k] = cluster_size[temp[0]] + cluster_size[temp[1]]
    # temp.append(cluster_size[n - 1 + k])
    # # print(temp)
    # Z.append(temp)
    # print(Z)
    # # plt.figure(figsize=(12, 10))
    # dn = dendrogram(Z)
    # # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # plt.title('Hierarchical Clustering Dendrogram')
    # # plt.xlabel('Loan Amount')
    # plt.ylabel('Distance')
    # plt.savefig('cluster1.svg')
    # plt.show()
