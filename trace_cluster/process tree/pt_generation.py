import string
from pm4py.objects.process_tree.process_tree import ProcessTree
from pm4py.objects.process_tree import semantics
from pm4py.objects.process_tree.pt_operator import Operator
from pm4py.visualization.process_tree import factory as pt_vis_factory
from pm4py.statistics.traces.log import case_statistics
import copy

if __name__ == "__main__":

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

    '''
    print(root)
    print(root2)
    print(root3)
    print(root4)'''

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

    print(root5)





    # log
    log1 = semantics.generate_log(root5,no_traces=100)
    print((log1[0]))
    from pm4py.algo.filtering.log.end_activities import end_activities_filter
    log_end = end_activities_filter.get_end_activities(log1)
    variants_count = case_statistics.get_variant_statistics(log1)
    print(variants_count)
    print(log_end)





    '''
    for i,child in enumerate(root.children):
        if (isinstance(child.label,str)):
            print(type(child.label))
    
    parameters = {"format": "svg"}
    gviz = pt_vis_factory.apply(root,parameters = {"format":"svg"})
    pt_vis_factory.save(gviz, "pt.svg")
    '''

