from trace_cluster.merge_log import merge_log
from pm4py.objects.log.importer.xes import factory as xes_importer
from matplotlib import rc
import matplotlib.pyplot as plt
if __name__ == "__main__":

    F1all = dict()
    precall = dict()
    fitall = dict()
    PIC_PATH = '/home/yukun/resultlog/'
    percent = 1
    alpha = 0.5
    runtime = dict()

    LOG_PATH = "/home/yukun/dataset/Receipt4.xes"
    ATTR_NAME = 'responsible'
    METHOD = 'dfg'

    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime)
    fitall['Receipt'] = list(plot_fit.values())
    precall['Receipt'] = list(plot_prec.values())
    F1all['Receipt'] = list(plot_F1.values())

    LOG_PATH = "/home/yukun/dataset/filteredbpic2017.xes"
    ATTR_NAME = 'CreditScore'
    METHOD = 'dfg'

    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime)
    fitall['filteredbpic2017'] = list(plot_fit.values())
    precall['filteredbpic2017'] = list(plot_prec.values())
    F1all['filteredbpic2017'] = list(plot_F1.values())

    LOG_PATH = "/home/yukun/dataset/BPIC2012_A.xes"
    ATTR_NAME = 'AMOUNT_REQ'
    METHOD = 'dfg'

    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime)
    fitall['BPIC2012_A'] = list(plot_fit.values())
    precall['BPIC2012_A'] = list(plot_prec.values())
    F1all['BPIC2012_A'] = list(plot_F1.values())

    LOG_PATH = "/home/yukun/dataset/document_logs/Control_summary.xes"
    ATTR_NAME = 'amount_applied0'
    METHOD = 'dfg'

    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime)
    fitall['Control_summary'] = list(plot_fit.values())
    precall['Control_summary'] = list(plot_prec.values())
    F1all['Control_summary'] = list(plot_F1.values())


    LOG_PATH = "/home/yukun/dataset/document_logs/Payment_application.xes"
    ATTR_NAME = 'amount_applied0'
    METHOD = 'dfg'
    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime)
    fitall['Payment_application'] = list(plot_fit.values())
    precall['Payment_application'] = list(plot_prec.values())
    F1all['Payment_application'] = list(plot_F1.values())

    LOG_PATH = "/home/yukun/dataset/document_logs/Geo_parcel_document.xes"
    ATTR_NAME = 'amount_applied0'
    METHOD = 'dfg'
    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime)
    fitall['Geo'] = list(plot_fit.values())
    precall['Geo'] = list(plot_prec.values())
    F1all['Geo'] = list(plot_F1.values())

    x_axis = range(1, 24)
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, F1all['filteredbpic2017'], linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, F1all['BPIC2012_A'], linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(x_axis, F1all['Control_summary'], linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(x_axis, F1all['Geo'], linestyle="-", marker="s", linewidth=1)
    l5 = plt.plot(x_axis, F1all['Payment_application'], linestyle="-", marker="s", linewidth=1)
    l6 = plt.plot(x_axis, F1all['Receipt'], linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2,l3,l4,l5,l6], labels=['BPIC2017', 'BPIC2012','BPIC2018-Control','BPIC2018-Geo','BPIC2018-Payment','Receipt'], loc='best')
    plt.title('F1-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'F1allreallogs' + '.svg')
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, F1all['filteredbpic2017'], linestyle="-", linewidth=2)
    l2 = plt.plot(x_axis, F1all['BPIC2012_A'], linestyle="-", linewidth=2)
    l3 = plt.plot(x_axis, F1all['Control_summary'], linestyle="-", linewidth=2)
    l4 = plt.plot(x_axis, F1all['Geo'], linestyle="-", linewidth=2)
    l5 = plt.plot(x_axis, F1all['Payment_application'], linestyle="-", linewidth=2)
    l6 = plt.plot(x_axis, F1all['Receipt'], linestyle="-", linewidth=2)
    plt.xticks(x_axis)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('F1-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'F1allreallogs-line' + '.svg')



    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, fitall['filteredbpic2017'], linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, fitall['BPIC2012_A'], linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(x_axis, fitall['Control_summary'], linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(x_axis, fitall['Geo'], linestyle="-", marker="s", linewidth=1)
    l5 = plt.plot(x_axis, fitall['Payment_application'], linestyle="-", marker="s", linewidth=1)
    l6 = plt.plot(x_axis, fitall['Receipt'], linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('Fitness-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Fitness")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Fitnessallreallogs' + '.svg')

    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, fitall['filteredbpic2017'], linestyle="-", linewidth=2)
    l2 = plt.plot(x_axis, fitall['BPIC2012_A'], linestyle="-", linewidth=2)
    l3 = plt.plot(x_axis, fitall['Control_summary'], linestyle="-", linewidth=2)
    l4 = plt.plot(x_axis, fitall['Geo'], linestyle="-", linewidth=2)
    l5 = plt.plot(x_axis, fitall['Payment_application'], linestyle="-", linewidth=2)
    l6 = plt.plot(x_axis, fitall['Receipt'], linestyle="-", linewidth=2)
    plt.xticks(x_axis)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('Fitness-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Fitness")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Fitnessallreallogs-line' + '.svg')



    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, precall['filteredbpic2017'], linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, precall['BPIC2012_A'], linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(x_axis, precall['Control_summary'], linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(x_axis, precall['Geo'], linestyle="-", marker="s", linewidth=1)
    l5 = plt.plot(x_axis, precall['Payment_application'], linestyle="-", marker="s", linewidth=1)
    l6 = plt.plot(x_axis, precall['Receipt'], linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('Precision-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Precision")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Precisionallreallogs' + '.svg')

    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, precall['filteredbpic2017'], linestyle="-", linewidth=2)
    l2 = plt.plot(x_axis, precall['BPIC2012_A'], linestyle="-", linewidth=2)
    l3 = plt.plot(x_axis, precall['Control_summary'], linestyle="-", linewidth=2)
    l4 = plt.plot(x_axis, precall['Geo'], linestyle="-", linewidth=2)
    l5 = plt.plot(x_axis, precall['Payment_application'], linestyle="-", linewidth=2)
    l6 = plt.plot(x_axis, precall['Receipt'], linestyle="-", linewidth=2)
    plt.xticks(x_axis)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('Precision-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Precision")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Precisionallreallogs-line' + '.svg')



    # individual


