from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.discovery.alpha import factory as alpha_miner

if __name__ == "__main__":


    '''
    #pt 1
    root = ProcessTree(operator=Operator.SEQUENCE)
    child1 = ProcessTree(label='a')
    child2 = ProcessTree(label='b')
    child3 = ProcessTree(operator=Operator.PARALLEL)
    child4 = ProcessTree(label='i')
    child5 = ProcessTree(label='j')
    root.children.append(child1)
    root.children.append(child2)
    root.children.append(child3)
    root.children.append(child4)
    root.children.append(child5)
    child1.parent = root
    child2.parent = root
    child3.parent = root
    child4.parent = root
    child5.parent = root

    child2_1 = ProcessTree(operator=Operator.SEQUENCE)
    child2_2 = ProcessTree(operator=Operator.SEQUENCE)
    child3.children.append(child2_1)
    child3.children.append(child2_2)
    child2_1.parent = child3
    child2_2.parent = child3


    child3_1 = ProcessTree(label='c')
    child3_2 = ProcessTree(label='d')
    child3_3 = ProcessTree(label='e')
    child2_1.children.append(child3_1)
    child2_1.children.append(child3_2)
    child2_1.children.append(child3_3)
    child3_1.parent = child2_1
    child3_2.parent = child2_1
    child3_3.parent = child2_1

    child3_4 = ProcessTree(label='f')
    child3_5 = ProcessTree(label='g')
    child2_2.children.append(child3_4)
    child2_2.children.append(child3_5)
    child3_4.parent = child2_2
    child3_5.parent = child2_2

    print(root)




    
    #pt2

    root2 = copy.deepcopy(root)
    modify_child = ProcessTree(operator=Operator.SEQUENCE,parent=root2)
    modify_child.parent = root2
    mo3_1 = ProcessTree(label='c')
    mo3_2 = ProcessTree(operator=Operator.PARALLEL)
    modify_child.children.append(mo3_1)
    modify_child.children.append(mo3_2)
    mo4_1 = ProcessTree(label='d')
    mo4_2 = ProcessTree(label='e')
    mo3_2.children.append(mo4_1)
    mo3_2.children.append(mo4_2)


    (root2.children[2]).children[0] = modify_child


    #pt3
    root3 = copy.deepcopy(root)
    ((root3.children)[2].children)[0].operator = Operator.XOR





    # pt4
    root4 = copy.deepcopy(root)
    (root4.children)[2].operator = Operator.XOR

    
    print(root)
    print(root2)
    print(root3)
    print(root4)

    root5 = ProcessTree(operator=Operator.SEQUENCE)
    c1 = ProcessTree(operator=Operator.PARALLEL)
    c2 = ProcessTree(operator=Operator.PARALLEL)
    root5.children.append(c1)
    root5.children.append(c2)
    child3_1 = ProcessTree(label='c')
    child3_2 = ProcessTree(label='d')
    c1.children.append(child3_1)
    c1.children.append(child3_2)
    child3_4 = ProcessTree(label='f')
    child3_5 = ProcessTree(label='g')
    c2.children.append(child3_4)
    c2.children.append(child3_5)

    print(root)'''

    log_1 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_1_1.xes")
    log_2 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_2_1.xes")
    log_3 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_3_1.xes")
    log_4 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_4_1.xes")
    log_5 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_1_2.xes")
    log_6 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_2_2.xes")
    log_7 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_3_2.xes")
    log_8 = xes_importer.apply("C:\\Users\\yukun\\PycharmProjects\\PTandLogGenerator\\data\\logs\\log_1_4_2.xes")

    net, initial_marking, final_marking = alpha_miner.apply(log_5)
    gviz = pn_vis_factory.apply(net, initial_marking, final_marking)
    pn_vis_factory.view(gviz)
    '''
    #loglist = [log_1,log_2,log_3,log_4,log_5,log_6,log_7,log_8]
    loglist = [log_1, log_2, log_3, log_4]
    percent = 1
    alpha =0.5
    print(filter_subsets.sublog_percent(log_1, 1))
    print(filter_subsets.sublog_percent(log_2, 1))
    print(filter_subsets.sublog_percent(log_3, 1))
    print(filter_subsets.sublog_percent(log_4, 1))
    print(filter_subsets.sublog_percent(log_5, 1))
    print(filter_subsets.sublog_percent(log_6, 1))
    print(filter_subsets.sublog_percent(log_7, 1))
    print(filter_subsets.sublog_percent(log_8, 1))

    

    sim_act = act_dist_calc.act_sim_percent(log_3, log_4, percent, percent)
    print(sim_act)'''



    '''
    lists = loglist
    size = len(lists)
    print(size)
    dist_mat = np.zeros((size, size))

    for i in range(0, size - 1):
        for j in range(i + 1, size):
            sim_act = act_dist_calc.act_sim_percent(lists[i], lists[j], percent, percent)
            print([i, len(lists[i]), j, len(lists[j]), sim_act])

            sim_suc = suc_dist_calc.suc_sim_percent(lists[i], lists[j], percent, percent)
            print([i, len(lists[i]), j, len(lists[j]), sim_suc])


            # sim_suc = suc_dist_calc.suc_sim(lists[i], lists[j], lists[i+size],
            #                                lists[j+size],
            #                                freq, num, parameters={"single": True})
            #dist_mat[i][j] = sim_act
            dist_mat[i][j] = (sim_act * alpha + sim_suc * (1 - alpha))
            dist_mat[j][i] = dist_mat[i][j]

    print(dist_mat)

    y = squareform(dist_mat)
    print(y)
    Z = linkage(y, method='average')
    print(Z)
    print(cophenet(Z, y))  # return vector is the pairwise dist generated from Z
    fig = plt.figure(figsize=(10, 8))
    # dn = fancy_dendrogram(Z, max_d=0.35)
    dn = dendrogram(Z)
    plt.savefig('cluster.svg')
    plt.show()'''


    '''
    variants_count = case_statistics.get_variant_statistics(log_1)
    print(variants_count)
    print(filter_subsets.sublog_percent(log_1,1))
    
    print(log)
    log_end = end_activities_filter.get_end_activities(log)
    
    print(log_end)
    net, initial_marking, final_marking = alpha_miner.apply(log)
    gviz = pn_vis_factory.apply(net, initial_marking, final_marking)
    pn_vis_factory.view(gviz)
    '''






    # log
    #log1 = semantics.generate_log(root5,no_traces=100)
    #print((log1[0]))






    '''
    for i,child in enumerate(root.children):
        if (isinstance(child.label,str)):
            print(type(child.label))
    
    parameters = {"format": "svg"}
    gviz = pt_vis_factory.apply(root,parameters = {"format":"svg"})
    pt_vis_factory.save(gviz, "pt.svg")
    '''

