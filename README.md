# Welcome to Process Mining for Python!

PM4Py is a python library that supports (state-of-the-art) process mining algorithms in python. It is completely open source and intended to be used in both academia and industry projects.

The official website of the library is http://pm4py.org/

You can always check out (changes to) the source code at the github repo.

A very simple example, to whet your appetite:

from pm4py.algo.discovery.alpha import factory as alpha_miner
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.visualization.petrinet import factory as vis_factory

log = xes_importer.import_log('<path-to-xes-log-file>')
net, initial_marking, final_marking = alpha_miner.apply(log)
gviz = vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz)
*****
## Trace clustering
Before is general information of PM4py and here we introduce also the usage of trace_cluster folder.  

Before start playing with our package, make sure you follow the instruction above to correctly install PM4py.  

One simple example to run trace clustering is the `python -m trace_cluster.merge_log.plots_indiv`  
This command will returns you a set of plots containing the cluster dedrogram and plots for fitness, precision and F1-score at each cluster step.

### Input  
* LOG_PATH: xes log file you would to do trace clustering based on case attribute (here we default use receipt log located in `trace_cluster/example/real_log`) and a set of parameters.  
* ATTR_NAME: case attribute name used for clustering  
* METHOD: we offer several options in the algorithm, namlely 'dfg','avg','DMM'(it means direct follow graph method, average intertrace method and dual minimal match intertrace method)  
* PIC_PATH: location of generated plots  
* plot_clu: number of cluster wanted to have at the end  

### Output  
* clustering dendrogram (svg file)  
* plots of fitness, precision and F1-score w.r.t. cluster steps. (svg file)  
* all the data of fitness, precision and F1-score at each cluster step.
