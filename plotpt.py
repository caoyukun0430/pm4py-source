import os
import pandas as pd
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.log import case_statistics
from IPython.display import display
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.visualization.process_tree import factory as pt_vis_factory
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.util import constants
from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def protree():
    log = xes_import_factory.apply("D:\\Sisc\\19SS\\APM\\APM_A1\\APM_Assignment_1.xes")
    variants = variants_filter.get_variants(log)
    variants_count = case_statistics.get_variant_statistics(log)
    variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
    #print(variants_count)

    '''
    filtered_var_list = []
    dis_list=[]
    for i in range(len(variants_count)):
        if variants_count[i]['count'] >= 5:
            filtered_var_list.append(variants_count[i])
            dis_list.append(variants_count[i]['variant'])

    df = pd.DataFrame(filtered_var_list)
    #df2= pd.DataFrame.from_dict(variants_count)
    #print(df2)
    #df['variant'].str.split(',',expand=True)
    #print(df['variant'].str.split(',',expand=True))
    #dis_list = list(set(dis_list))
    string=[]
    for str in dis_list:
        string.extend(str.split(','))
    print(string)
    #df3 = pd.DataFrame(string)
    #print(type(df3.iloc[:,[0]]))
    df2=pd.DataFrame({'var':string}).drop_duplicates('var','first')#delete duplicate in variants
    print(df2)
    #print(list(set(string)))
    '''
    '''
    #log = xes_import_factory.apply(os.path.join("tests", "input_data", "receipt.xes"))
    activities = attributes_filter.get_attribute_values(log, "org:resource")
    tracefilter_log_alex = attributes_filter.apply_events(log, ["Alex"],parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "org:resource","positive": True})
    dfg = dfg_factory.apply(tracefilter_log_alex)
    #print(dfg)
    #gviz = dfg_vis_factory.apply(dfg, log=log)
    #dfg_vis_factory.view(gviz)
    tree_alex = inductive_miner.apply_tree(tracefilter_log_alex)
    gviz_alex = pt_vis_factory.apply(tree_alex, parameters={"format": "svg"})
    #pt_vis_factory.view(gviz_alex)
    tracefilter_log_alfred = attributes_filter.apply_events(log, ["Alfred"],parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "org:resource","positive": True})

log = xes_importer.apply("D:\\Sisc\\19SS\\thesis\\Dataset\\BPI_Challenge_2012.xes")
list_of_vals=['5000','7000']
tracefilter_log = filter_subsets.apply_trace_attributes(log, list_of_vals,
                                                 parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                             "positive": True})
tracefilter_log_7000 = filter_subsets.apply_trace_attributes(tracefilter_log,['7000'],
                                                 parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "AMOUNT_REQ",
                                                             "positive": True})
dfg = dfg_factory.apply(tracefilter_log_7000)
print(dfg)

    #net, initial_marking, final_marking = alpha_miner.apply(log)
    #gviz=pn_vis_factory.apply(net, initial_marking, final_marking)
    #pn_vis_factory.view(gviz)
    tree_org = inductive_miner.apply_tree(log)
    tree_alex = inductive_miner.apply_tree(tracefilter_log_alex)
    tree_alfred = inductive_miner.apply_tree(tracefilter_log_alfred)

    gviz_org = pt_vis_factory.apply(tree_org, parameters={"format": "svg"})
    gviz_alex= pt_vis_factory.apply(tree_alex, parameters={"format": "svg"})
    gviz_alfred = pt_vis_factory.apply(tree_alfred, parameters={"format": "svg"})
    #pt_vis_factory.view(gviz)
    pt_vis_factory.save(gviz_org, "pt_org.svg")
    pt_vis_factory.save(gviz_alex,"pt_alex.svg")
    pt_vis_factory.save(gviz_alfred, "pt_alfred.svg")

    from IPython.display import Image
    image = Image(open(gviz_org.render(), "rb").read())
    from IPython.display import display
    #return display(image)
    return pt_vis_factory.view(gviz_org)
    '''

if __name__ == "__main__":
    protree()
