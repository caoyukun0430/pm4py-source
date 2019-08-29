from pm4py.objects.process_tree.process_tree import ProcessTree
from pm4py.objects.process_tree import semantics
from pm4py.objects.process_tree.pt_operator import Operator
from pm4py.visualization.process_tree import factory as pt_vis_factory
from pm4py.statistics.traces.log import case_statistics

if __name__ == "__main__":

    #pt 1
    root = ProcessTree(operator=Operator.SEQUENCE)
    child1 = ProcessTree(label='a')
    child3 = ProcessTree(operator=Operator.PARALLEL)
    child4 = ProcessTree(label='i')
    root.children.append(child1)
    root.children.append(child3)
    root.children.append(child4)
    child1.parent = root
    child3.parent = root
    child4.parent = root

    child2_1 = ProcessTree(operator=Operator.SEQUENCE)
    child2_2 = ProcessTree(operator=Operator.SEQUENCE)
    child3.children.append(child2_1)
    child3.children.append(child2_2)
    child2_1.parent = child3
    child2_2.parent = child3


    child3_1 = ProcessTree(label='c')
    child3_2 = ProcessTree(label='d')
    child2_1.children.append(child3_1)
    child2_1.children.append(child3_2)
    child3_1.parent = child2_1
    child3_2.parent = child2_1

    child3_4 = ProcessTree(label='f')
    child3_5 = ProcessTree(label='g')
    child2_2.children.append(child3_4)
    child2_2.children.append(child3_5)
    child3_4.parent = child2_2
    child3_5.parent = child2_2

    print(root)

    log = semantics.generate_log(root,no_traces=100)
    variants_count = case_statistics.get_variant_statistics(log)
    print(variants_count)
