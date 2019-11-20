import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
import numpy as np
from trace_cluster.pt_gene import pt_gen
from trace_cluster import filter_subsets
from trace_cluster.merge_log import merge_log
import Levenshtein

from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.filtering.log.attributes import attributes_filter
import string

def leven_preprocess(list1,list2):
    '''
    this function transform event name to alphabet for the sake of levenshtein distance
    :param list1:
    :param list2:
    :return:
    '''

    listsum = sorted(list(set(list1+list2)))
    alphabet = list(string.ascii_letters)[0:len(listsum)]
    str1 = [alphabet[listsum.index(item)] for item in list1]
    str2 = [alphabet[listsum.index(item)] for item in list2]
    # dictmap = dict(zip(listsum, alphabet[0:len(listsum)]))
    # print(dictmap)
    # str1 = [value for key, value in dictmap.items() if key in list1]
    # str2 = [value for key, value in dictmap.items() if key in list2]
    # print([str1,str2])
    str1 = ''.join(str1)
    str2 = ''.join(str2)
    return str1, str2


def leven_dist(log1, log2, percent_1, percent_2):
    '''

    this function compare the levenstein distance between two sublogs via the two lists of variants.
    '''

    (dataframe_1, var_list_1) = filter_subsets.sublog_percent(log1, percent_1)
    (dataframe_2, var_list_2) = filter_subsets.sublog_percent(log2, percent_2)

    if len(var_list_1) >= len(var_list_2):
        max_len = len(var_list_1)
        min_len = len(var_list_2)
        max_var = var_list_1
        min_var = var_list_2
        var_count_max = dataframe_1['count']
        var_count_min = dataframe_2['count']
    else:
        max_len = len(var_list_2)
        min_len = len(var_list_1)
        max_var = var_list_2
        min_var = var_list_1
        var_count_max = dataframe_2['count']
        var_count_min = dataframe_1['count']

    # print("list1:", max_len)
    # print("list2:", min_len)

    # print((var_count_max))
    # print((var_count_min))
    dist_matrix = np.zeros((max_len, min_len))
    max_per_var = np.zeros(max_len)
    max_freq = np.zeros(max_len)
    min_freq = np.zeros(min_len)
    min_per_var = np.zeros(min_len)
    col_sum = np.zeros(max_len)
    index_rec = set(list(range(min_len)))

    if var_list_1 == var_list_2:
        print("Please give different variant lists!")
        dist = 0
    else:
        for i in range(max_len):
            dist_vec = np.zeros(min_len)
            # str1 = ''.join(max_var[i])
            # print(str1)
            for j in range(min_len):
                # str2 = ''.join(min_var[j])
                # print(str2)
                (str1, str2) = leven_preprocess(max_var[i], min_var[j])
                max_len = np.max([len(str1), len(str2)])
                # levenstein distance between variants
                dist_vec[j] = (Levenshtein.distance(str1, str2)) / max_len
                # print([i, j, dist_vec[j]])
                dist_matrix[i][j] = dist_vec[j]
                if j == (min_len - 1):
                    # max_loc_col = np.argmax(dist_matrix[i, :])  # location of max value
                    max_loc_col = np.argmin(dist_vec)
                    if abs(dist_vec[max_loc_col]) <= 1e-8:
                        index_rec.discard(max_loc_col)
                        # print("skip:",[i,max_loc_col])
                        max_freq[i] = var_count_max.iloc[i] * var_count_min.iloc[max_loc_col] * 2
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i] * 2
                    else:
                        max_freq[i] = var_count_max.iloc[i] * var_count_min.iloc[max_loc_col]
                        # print("max", [i, max_loc_col])
                        max_per_var[i] = dist_vec[max_loc_col] * max_freq[i]
                        # print(type(dist_vec)) # location of max value
                        # print([i,max_loc_col])
                        # max_per_var[i] = dist_vec[max_loc_col]
                        # max_per_var[i] = dist_matrix[i][max_loc_col] * var_count_max.iloc[i] * var_count_min.iloc[
                        #    max_loc_col]

        if (len(index_rec) != 0):
            # print(index_rec)
            for i in list(index_rec):
                min_loc_row = np.argmin(dist_matrix[:, i])
                min_freq[i] = var_count_max.iloc[min_loc_row] * var_count_min.iloc[i]
                min_per_var[i] = dist_matrix[min_loc_row, i] * min_freq[i]
            # print("dual",[i,min_loc_row])
            # print("dual",dist_matrix[min_loc_row, i])

        # print(max_freq)
        # single linkage
        # print(max_freq, max_per_var)
        # print(min_freq, min_per_var)
        dist = (np.sum(max_per_var) + np.sum(min_per_var)) / (np.sum(max_freq) + np.sum(min_freq))

    # print(index_rec)
    # print(dist_matrix)
    # print(max_per_var)
    # print(max_freq)

    return dist

def leven_dist_avg(log1, log2, percent_1, percent_2):
    '''

    this function compare the levenstein distance between two sublogs via the two lists of variants.
    avg doesn't have: if var_list_1 == var_list_2:print("Please give different variant lists!")
    '''

    (dataframe_1, var_list_1) = filter_subsets.sublog_percent(log1, percent_1)
    (dataframe_2, var_list_2) = filter_subsets.sublog_percent(log2, percent_2)

    if len(var_list_1) >= len(var_list_2):
        max_len = len(var_list_1)
        min_len = len(var_list_2)
        max_var = var_list_1
        min_var = var_list_2
        var_count_max = dataframe_1['count']
        var_count_min = dataframe_2['count']
    else:
        max_len = len(var_list_2)
        min_len = len(var_list_1)
        max_var = var_list_2
        min_var = var_list_1
        var_count_max = dataframe_2['count']
        var_count_min = dataframe_1['count']

    # print("list1:", max_var)
    # print("list2:", min_var)
    #
    # print((var_count_max))
    # print((var_count_min))
    dist_matrix = np.zeros((max_len, min_len))
    max_per_var = np.zeros(max_len)
    max_freq = np.zeros(max_len)
    min_freq = np.zeros(min_len)
    min_per_var = np.zeros(min_len)
    col_sum = np.zeros(max_len)
    index_rec = set(list(range(min_len)))

    for i in range(max_len):
        dist_vec = np.zeros(min_len)
        # join all strings into one
        # str1 = ''.join(max_var[i])
        # print('str1',str1)
        for j in range(min_len):
            # str2 = ''.join(min_var[j])
            (str1, str2) = leven_preprocess(max_var[i], min_var[j])
            # print('str', [i, j, str1, str2])
            max_len = np.max([len(str1), len(str2)])
            # levenstein distance between variants
            dist_vec[j] = (Levenshtein.distance(str1, str2)) / max_len
            # print([i, j, dist_vec[j]])
            col_sum[i] += dist_vec[j] * var_count_max.iloc[i] * var_count_min.iloc[j]
            dist_matrix[i][j] = dist_vec[j]

    # print(max_freq)
    # single linkage
    # print(max_freq, max_per_var)
    # print(min_freq, min_per_var)
    vmax_vec = (var_count_max.values).reshape(-1, 1)
    # print(vmax_vec)
    vmin_vec = (var_count_min.values).reshape(1, -1)
    # print(vmin_vec)
    vec_sum = np.sum(np.dot(vmax_vec, vmin_vec))
    # dist = np.sum(dist_matrix) / vec_sum
    dist = np.sum(col_sum) / vec_sum

    # print(index_rec)
    #print(dist_matrix)
    # print(max_per_var)
    # print(max_freq)

    return dist


if __name__ == "__main__":

    LOG_PATH = "D:/Sisc/19SS/thesis/Dataset/Receipt4.xes"
    ATTR_NAME = 'responsible'
    METHOD = 'DMM'

    # PIC_PATH = 'D:/Sisc/19SS/thesis/Dataset/'
    log = xes_importer.apply(LOG_PATH)
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # runtime = dict()
    # F1val = dict()
    # METHOD = 'dfg'
    # ATTR_NAME = 'RequestedAmount'

    # sublog = xes_importer.apply(
    #     "D:\\Sisc\\19SS\\thesis\\Dataset\\BPIC2017\\sublog_598.xes")
    # log1 = xes_importer.apply(
    #     "C:\\Users\\yukun\\PycharmProjects\\pm4py-source\\trace_cluster\\merge_log\\log_3_0_dfg.xes")

    # percent = 1
    # alpha = 0.5
    # # ATTR_NAME = 'amount_applied0'
    # TYPE = METHOD + ATTR_NAME

    list_of_vals = []
    list_log = []
    list_of_vals_dict = attributes_filter.get_trace_attribute_values(log, ATTR_NAME)

    list_of_vals_keys = list(list_of_vals_dict.keys())
    for i in range(len(list_of_vals_keys)):
        list_of_vals.append(list_of_vals_keys[i])

    print(list_of_vals)
    for i in range(len(list_of_vals)):
        logsample = merge_log.log2sublog(log, list_of_vals[i], ATTR_NAME)
        list_log.append(logsample)
    print(len(list_log))

    dist = leven_dist_avg(list_log[0], list_log[1], 1, 1)

    # percent = 1
    # alpha = 0.5
    # loglist = pt_gen.openAllXes("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs", 4, 1)
    #
    #
    # dist = leven_dist(loglist[2],loglist[3],1,1)

    print(dist)
    #dist = leven_dist_avg(loglist[2], loglist[3], 1, 1)

    #print(dist)




    # lists = loglist
    # size = len(lists)
    # print(size)
    # dist_mat = np.zeros((size, size))
    #
    #
    # for i in range(0, size - 1):
    #     for j in range(i + 1, size):
    #         dist_mat[i][j] = leven_dist(lists[i], lists[j],percent,percent)
    #         #print([i,j,dist_mat[i][j]])
    #         dist_mat[j][i] = dist_mat[i][j]
    #
    # print(dist_mat)
    #
    # y = squareform(dist_mat)
    # print(y)
    # Z = linkage(y, method='average')
    # print(Z)
    # print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    # fig = plt.figure(figsize=(10, 8))
    # # dn = fancy_dendrogram(Z, max_d=0.35)
    # dn = dendrogram(Z)
    # # dn = dendrogram(Z,labels=np.array(list_of_vals))
    # #plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Distance')
    # plt.savefig('cluster.svg')
    # plt.show()

