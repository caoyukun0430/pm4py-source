from trace_cluster.merge_log import merge_log
from pm4py.objects.log.importer.xes import factory as xes_importer
from matplotlib import rc
import matplotlib.pyplot as plt

def standard_plt(LOG_PATH,ATTR_NAME,PIC_PATH,plot_clu):
    # PIC_PATH = '/home/yukun/resultlog/'
    percent = 1
    alpha = 0.5
    runtime = dict()

    log = xes_importer.apply(LOG_PATH)

    METHOD = 'avg'
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME + 'update'
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    F1valup = list(plot_F1.values())
    x_axis = range(1, plot_clu + 1)
    merge_log.five_plots(plot_fit, plot_prec, plot_F1,plot_boxfit,plot_boxprec,plot_box,plot_length,plot_clu,x_axis,PIC_PATH,TYPE)

    # ATTR_NAME = 'responsible'
    METHOD = 'avg'
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    F1val = list(plot_F1.values())
    x_axis = range(1, plot_clu + 1)
    merge_log.five_plots(plot_fit, plot_prec, plot_F1,plot_boxfit,plot_boxprec,plot_box,plot_length,plot_clu,x_axis,PIC_PATH,TYPE)
    F1avg = [F1val, F1valup]
    print('F1compare', F1avg)

    # DMM
    # ATTR_NAME = 'responsible'
    # PIC_PATH = '/home/yukun/resultlog/Receipt/leven/' + ATTR_NAME + '/'
    METHOD = 'DMM'
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME + 'update'
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    F1valup = list(plot_F1.values())
    x_axis = range(1, plot_clu + 1)
    merge_log.five_plots(plot_fit, plot_prec, plot_F1,plot_boxfit,plot_boxprec,plot_box,plot_length,plot_clu,x_axis,PIC_PATH,TYPE)

    # ATTR_NAME = 'responsible'
    METHOD = 'DMM'
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    F1val = list(plot_F1.values())
    x_axis = range(1, plot_clu + 1)
    merge_log.five_plots(plot_fit, plot_prec, plot_F1,plot_boxfit,plot_boxprec,plot_box,plot_length,plot_clu,x_axis,PIC_PATH,TYPE)
    F1DMM = [F1val, F1valup]
    print('F1compare', F1DMM)

    # FT
    # ATTR_NAME = 'responsible'
    # PIC_PATH = '/home/yukun/resultlog/Receipt/leven/' + ATTR_NAME + '/'
    METHOD = 'avg'
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME + 'updateFT'
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    F1valup = list(plot_F1.values())
    x_axis = range(1, plot_clu + 1)
    merge_log.five_plots(plot_fit, plot_prec, plot_F1,plot_boxfit,plot_boxprec,plot_box,plot_length,plot_clu,x_axis,PIC_PATH,TYPE)

    # ATTR_NAME = 'responsible'
    METHOD = 'avg'
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME+ 'FT'
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    F1val = list(plot_F1.values())
    x_axis = range(1, plot_clu + 1)
    merge_log.five_plots(plot_fit, plot_prec, plot_F1,plot_boxfit,plot_boxprec,plot_box,plot_length,plot_clu,x_axis,PIC_PATH,TYPE)
    F1avg_FT = [F1val, F1valup]
    print('F1compare', F1avg_FT)

    # DMM
    # ATTR_NAME = 'responsible'
    # PIC_PATH = '/home/yukun/resultlog/Receipt/leven/' + ATTR_NAME + '/'
    METHOD = 'DMM'
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME + 'updateFT'
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    F1valup = list(plot_F1.values())
    x_axis = range(1, plot_clu + 1)
    merge_log.five_plots(plot_fit, plot_prec, plot_F1,plot_boxfit,plot_boxprec,plot_box,plot_length,plot_clu,x_axis,PIC_PATH,TYPE)

    # ATTR_NAME = 'responsible'
    METHOD = 'DMM'
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME+ 'FT'
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    F1val = list(plot_F1.values())
    x_axis = range(1, plot_clu + 1)
    merge_log.five_plots(plot_fit, plot_prec, plot_F1,plot_boxfit,plot_boxprec,plot_box,plot_length,plot_clu,x_axis,PIC_PATH,TYPE)
    F1DMM_FT = [F1val, F1valup]
    print('F1compare', F1DMM_FT)

    print('runtime',runtime)

    fig9 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, F1DMM[1], color="b", linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, F1avg[1], color="r", linestyle="-", marker="o", linewidth=1)
    plt.xticks(x_axis)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2], labels=['Leven-DMM', 'Leven-AVG'], loc='best')
    plt.title('Leven-Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Leven-Recomputing' + '.svg')

    fig9 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, F1DMM_FT[1], color="b", linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, F1avg_FT[1], color="r", linestyle="-", marker="o", linewidth=1)
    plt.xticks(x_axis)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2], labels=['Feature Vector-DMM', 'Feature Vector-AVG'], loc='best')
    plt.title('Feature Vector-Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'FT-Recomputing' + '.svg')

    fig10 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, F1DMM[1], linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, F1DMM_FT[1], linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(x_axis, F1avg[1], linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(x_axis, F1avg_FT[1], linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4], labels=['Leven-DMM', 'Feature Vector-DMM', 'Leven-AVG', 'Feature Vector-AVG'],
               loc='best')
    plt.title('Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Recomputing' + '.svg')

    fig11 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, F1DMM[1], linestyle="-", linewidth=2)
    l2 = plt.plot(x_axis, F1DMM_FT[1], linestyle="-", linewidth=2)
    l3 = plt.plot(x_axis, F1avg[1], linestyle="-", linewidth=2)
    l4 = plt.plot(x_axis, F1avg_FT[1], linestyle="-", linewidth=2)
    plt.xticks(x_axis)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4], labels=['Leven-DMM', 'Feature Vector-DMM', 'Leven-AVG', 'Feature Vector-AVG'],
               loc='best')
    plt.title('Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Recomputing-line' + '.svg')

    METHOD = 'dfg'
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME + 'dfg'
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    x_axis = range(1, plot_clu + 1)

    fig10 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, list(plot_F1.values()), color="b", linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, F1DMM[1], color="r", linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(x_axis, F1DMM_FT[1], color="g", linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3], labels=['DFG', 'Leven-DMM', 'Feature Vector-DMM'], loc='best')
    plt.title('DFG,Leven and Feature Vector-Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    # plt.show()
    plt.savefig(PIC_PATH + 'DFG-Leven-FT-DMM' + '.svg')

    fig11 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, list(plot_F1.values()), color="b", linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, F1avg[1], color="r", linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(x_axis, F1avg_FT[1], color="g", linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3], labels=['DFG', 'Leven-AVG', 'Feature Vector-AVG'], loc='best')
    plt.title('DFG,Leven and Feature Vector-Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    # plt.show()
    plt.savefig(PIC_PATH + 'DFG-Leven-FT-AVG' + '.svg')

if __name__ == "__main__":
    # LOG_PATH = "/home/yukun/dataset/Receipt4.xes"
    # ATTR_NAME = 'responsible'
    # PIC_PATH = '/home/yukun/resultlog/Receipt/leven/' + ATTR_NAME + '/'
    plot_clu = 23
    # standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH,plot_clu)

    LOG_PATH = "/home/yukun/dataset/document_logs/Control_summary.xes"
    ATTR_NAME = 'amount_applied0'
    PIC_PATH = '/home/yukun/resultlog/Control_summary/' + ATTR_NAME + '/'
    standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH,plot_clu)

    LOG_PATH = "/home/yukun/dataset/filteredbpic2017.xes"
    ATTR_NAME = 'CreditScore'
    PIC_PATH = '/home/yukun/resultlog/filteredbpic2017/' + ATTR_NAME + '/'
    standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH,plot_clu)

    LOG_PATH = "/home/yukun/dataset/document_logs/Geo_parcel_document.xes"
    ATTR_NAME = 'amount_applied0'
    PIC_PATH = '/home/yukun/resultlog/Geo_parcel_document/' + ATTR_NAME + '/'
    standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH, plot_clu)

    LOG_PATH = "/home/yukun/dataset/BPIC2012_A.xes"
    ATTR_NAME = 'AMOUNT_REQ'
    PIC_PATH = '/home/yukun/resultlog/BPIC2012_A/' + ATTR_NAME + '/'
    standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH,plot_clu)


    #
    # LOG_PATH = "/home/yukun/dataset/document_logs/Payment_application.xes"
    # ATTR_NAME = 'amount_applied0'
    # PIC_PATH = '/home/yukun/resultlog/Payment_application/' + ATTR_NAME + '/'
    # standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH)
    #



