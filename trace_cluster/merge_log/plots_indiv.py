from trace_cluster.merge_log import merge_log
from pm4py.objects.log.importer.xes import factory as xes_importer
from matplotlib import rc
import matplotlib.pyplot as plt


# standard plots generating procedure for individual logs

def standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH, plot_clu):
    # PIC_PATH = '/home/yukun/resultlog/'
    # percent = 1
    # alpha = 0.5
    # runtime = dict()
    #
    # log = xes_importer.apply(LOG_PATH)
    #
    # METHOD = 'dfg'
    # # plot_clu = 23
    #
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME + 'dfg'
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha, runtime, plot_clu)
    # x_axis = range(1, plot_clu + 1)
    # merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
    #                      x_axis, PIC_PATH, TYPE)
    # F1dfg = list(plot_F1.values())
    #
    # METHOD = 'avg'
    # # plot_clu = 23
    #
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME + 'update'
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha,runtime,plot_clu)
    # F1valup = list(plot_F1.values())
    # x_axis = range(1, plot_clu + 1)
    # merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
    #                      x_axis, PIC_PATH, TYPE)
    #
    # # ATTR_NAME = 'responsible'
    # METHOD = 'avg'
    # # plot_clu = 23
    #
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_leven(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha,runtime,plot_clu)
    # F1val = list(plot_F1.values())
    # x_axis = range(1, plot_clu + 1)
    # merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
    #                      x_axis, PIC_PATH, TYPE)
    # F1avg = [F1val, F1valup]
    # print('F1compare', F1avg)
    #
    # # DMM
    # # ATTR_NAME = 'responsible'
    # # PIC_PATH = '/home/yukun/resultlog/Receipt/leven/' + ATTR_NAME + '/'
    # METHOD = 'DMM'
    # # plot_clu = 23
    #
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME + 'update'
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha,runtime,plot_clu)
    # F1valup = list(plot_F1.values())
    # x_axis = range(1, plot_clu + 1)
    # merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
    #                      x_axis, PIC_PATH, TYPE)
    #
    # # ATTR_NAME = 'responsible'
    # METHOD = 'DMM'
    # # plot_clu = 23
    #
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_leven(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha,runtime,plot_clu)
    # F1val = list(plot_F1.values())
    # x_axis = range(1, plot_clu + 1)
    # merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
    #                      x_axis, PIC_PATH, TYPE)
    # F1DMM = [F1val, F1valup]
    # print('F1compare', F1DMM)
    #
    # # FT
    # # ATTR_NAME = 'responsible'
    # # PIC_PATH = '/home/yukun/resultlog/Receipt/leven/' + ATTR_NAME + '/'
    # METHOD = 'avg'
    # # plot_clu = 23
    #
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME + 'updateFT'
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha,runtime,plot_clu)
    # F1valup = list(plot_F1.values())
    # x_axis = range(1, plot_clu + 1)
    # merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
    #                      x_axis, PIC_PATH, TYPE)
    #
    # # ATTR_NAME = 'responsible'
    # METHOD = 'avg'
    # # plot_clu = 23
    #
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME + 'FT'
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha,runtime,plot_clu)
    # F1val = list(plot_F1.values())
    # x_axis = range(1, plot_clu + 1)
    # merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
    #                      x_axis, PIC_PATH, TYPE)
    # F1avg_FT = [F1val, F1valup]
    # print('F1compare', F1avg_FT)
    #
    # # DMM
    # # ATTR_NAME = 'responsible'
    # # PIC_PATH = '/home/yukun/resultlog/Receipt/leven/' + ATTR_NAME + '/'
    # METHOD = 'DMM'
    # # plot_clu = 23
    #
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME + 'updateFT'
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha,runtime,plot_clu)
    # F1valup = list(plot_F1.values())
    # x_axis = range(1, plot_clu + 1)
    # merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
    #                      x_axis, PIC_PATH, TYPE)
    #
    # # ATTR_NAME = 'responsible'
    # METHOD = 'DMM'
    # # plot_clu = 23
    #
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME + 'FT'
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha,runtime,plot_clu)
    # F1val = list(plot_F1.values())
    # x_axis = range(1, plot_clu + 1)
    # merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
    #                      x_axis, PIC_PATH, TYPE)
    # F1DMM_FT = [F1val, F1valup]
    # print('F1compare', F1DMM_FT)
    #
    # print('runtime', runtime)

    # fig9 = plt.figure()
    # rc('text', usetex=True)
    # rc('font', family='serif')
    # l1 = plt.plot(x_axis, F1DMM[1], color="b", linestyle="-", marker="s", linewidth=1)
    # l2 = plt.plot(x_axis, F1avg[1], color="r", linestyle="-", marker="o", linewidth=1)
    # plt.xticks(x_axis,fontsize=7)
    # # plt.gca().invert_xaxis()
    # plt.ylim(0, 1.04)
    # plt.legend([l1, l2], labels=['Leven-DMM', 'Leven-AVG'], loc='best')
    # plt.title('Leven-Recomputing')
    # plt.xlabel("Num. of Cluster")
    # plt.ylabel("F1-Score")
    # plt.grid(axis='y')
    # plt.savefig(PIC_PATH + 'Leven-Recomputing' + '.svg')
    #
    # fig9 = plt.figure()
    # rc('text', usetex=True)
    # rc('font', family='serif')
    # l1 = plt.plot(x_axis, F1DMM_FT[1], color="b", linestyle="-", marker="s", linewidth=1)
    # l2 = plt.plot(x_axis, F1avg_FT[1], color="r", linestyle="-", marker="o", linewidth=1)
    # plt.xticks(x_axis,fontsize=7)
    # # plt.gca().invert_xaxis()
    # plt.ylim(0, 1.04)
    # plt.legend([l1, l2], labels=['Feature Vector-DMM', 'Feature Vector-AVG'], loc='best')
    # plt.title('Feature Vector-Recomputing')
    # plt.xlabel("Num. of Cluster")
    # plt.ylabel("F1-Score")
    # plt.grid(axis='y')
    # plt.savefig(PIC_PATH + 'FT-Recomputing' + '.svg')

    x_axis = range(1, 38)

    fitdfg = list({'1': 0.9997709473232487, '2': 0.7836985414090225, '3': 0.6344546056221994, '4': 0.5598326377287878,
                   '5': 0.6297353983582362, '6': 0.6185451951640831, '7': 0.6355197750090118, '8': 0.6418506336807615,
                   '9': 0.6785480506290226, '10': 0.6852800564935537, '11': 0.7118803718660606,
                   '12': 0.7118803718660606, '13': 0.7276588309578341, '14': 0.7296349033560376,
                   '15': 0.7375607123997523, '16': 0.7375607123997523, '17': 0.7369376123475584,
                   '18': 0.7342489666444912, '19': 0.7423798549348837, '20': 0.7505493335454727,
                   '21': 0.7599845218793149, '22': 0.7685676400282323, '23': 0.7713130917214615, '24': 0.77584711640831,
                   '25': 0.7827121813960869, '26': 0.7820883129907791, '27': 0.7916496807211623,
                   '28': 0.7981935481410068, '29': 0.7970031899638278, '30': 0.7942539179555976,
                   '31': 0.8000784552156083, '32': 0.8055696463177586, '33': 0.8084650217566973,
                   '34': 0.8132786207113566, '35': 0.8153800335070021, '36': 0.8137322793064841,
                   '37': 0.8138490595482556}.values())
    fitavg = list({'1': 0.9997709473232487, '2': 0.9309378054510451, '3': 0.9237406839260401, '4': 0.8334540577428595, '5': 0.8338100035143444, '6': 0.8162114563172015, '7': 0.8150394071590469, '8': 0.8317455173336861, '9': 0.8150156556594508, '10': 0.8240910328081718, '11': 0.8257566306053937, '12': 0.8309176087676436, '13': 0.839883465101112, '14': 0.8346523338014125, '15': 0.8441986389926217, '16': 0.8390095356910356, '17': 0.8330305779491267, '18': 0.8471903374090306, '19': 0.8516035441870451, '20': 0.8578314568500567, '21': 0.8634488432135976, '22': 0.8645982523476136, '23': 0.868259768382739, '24': 0.8636994223897103, '25': 0.856679614846518, '26': 0.8513907195864774, '27': 0.844947046474002, '28': 0.8498030702569347, '29': 0.8462191060851209, '30': 0.8480490370757081, '31': 0.8520515644673018, '32': 0.8489991994650803, '33': 0.8404727429811323, '34': 0.8418361349098915, '35': 0.8411566210053815, '36': 0.8271235685899141, '37': 0.8271235685899141}.values())
    fitDMM = list({'1': 0.9997709473232487, '2': 0.7811825632582832, '3': 0.7811825632582832, '4': 0.8179792502456606, '5': 0.8018568510122449, '6': 0.7925253940559268, '7': 0.7597893029540238, '8': 0.7592724656566405, '9': 0.7801837184839839, '10': 0.7908481534914805, '11': 0.8054500637536673, '12': 0.7956795779250202, '13': 0.807459081250442, '14': 0.8132645643155494, '15': 0.8242367208058162, '16': 0.8238502475212051, '17': 0.8286690227932908, '18': 0.8141666401655967, '19': 0.8018298056331247, '20': 0.8001589780175772, '21': 0.8085400646570868, '22': 0.8120894850649023, '23': 0.8171549177496749, '24': 0.8284362231862467, '25': 0.8295436259751193, '26': 0.8105598993625591, '27': 0.8105598993625591, '28': 0.8105598993625591, '29': 0.8105598993625591, '30': 0.8145202134323581, '31': 0.8171007454382717, '32': 0.8184365575205939, '33': 0.8184365575205939, '34': 0.8205441232228035, '35': 0.8248750735940122, '36': 0.8271235685899141, '37': 0.8271235685899141}.values())
    fitavg_FT = list({'1': 0.9997709473232487, '2': 0.9309378054510451, '3': 0.9832606197098297, '4': 0.8793519986560714, '5': 0.7706749457345676, '6': 0.698223577120232, '7': 0.6788479488449585, '8': 0.7076602603488424, '9': 0.7218350965990026, '10': 0.7402285296537686, '11': 0.7353138128109724, '12': 0.7533267589413529, '13': 0.7520991047211397, '14': 0.7661979089380874, '15': 0.774367104619678, '16': 0.787084592560295, '17': 0.7982067810830586, '18': 0.7860271104032431, '19': 0.784938895150885, '20': 0.7944817789175616, '21': 0.7989611709513197, '22': 0.8049235692062553, '23': 0.8066119316508444, '24': 0.8046202455216448, '25': 0.8119363583082309, '26': 0.8082040491235625, '27': 0.8064292754090416, '28': 0.8114627160829928, '29': 0.8072124235908171, '30': 0.8051676131136013, '31': 0.809758133371315, '32': 0.812613099072297, '33': 0.8148620375415894, '34': 0.8194874889731636, '35': 0.8176271574138223, '36': 0.8137322793064838, '37': 0.8138490595482558}.values())
    fitDMM_FT = list({'1': 0.9997709473232487, '2': 0.7836985414090225, '3': 0.6344546056221994, '4': 0.616489499014979, '5': 0.6887683043302091, '6': 0.7359948179776928, '7': 0.7288765074641665, '8': 0.7299378507909687, '9': 0.7317078795113426, '10': 0.7494717356478114, '11': 0.7429208092752506, '12': 0.7506135573856699, '13': 0.7612383486761527, '14': 0.771561997138332, '15': 0.7834233549130035, '16': 0.7955748272102879, '17': 0.8061975901654048, '18': 0.8059260810333702, '19': 0.8077811424181994, '20': 0.7963407207396086, '21': 0.8048862373941232, '22': 0.8027919295375745, '23': 0.8034554607860295, '24': 0.8096089168504973, '25': 0.8047527295288737, '26': 0.8111902163244983, '27': 0.8094924683350144, '28': 0.8101118377897661, '29': 0.8148484551453854, '30': 0.8148484551453854, '31': 0.8155410775437181, '32': 0.8155410775437181, '33': 0.8227049377077368, '34': 0.8269131257236081, '35': 0.824888775911172, '36': 0.827123568589914, '37': 0.827123568589914}.values())

    precdfg = list({'1': 0.15551664221436445, '2': 0.5809342867708414, '3': 0.7206228578472276, '4': 0.7904671433854207, '5': 0.7033893092314674, '6': 0.7503384082004511, '7': 0.7850649580553306, '8': 0.8108258823208572, '9': 0.7617581021134117, '10': 0.7850951800693122, '11': 0.7413479731805152, '12': 0.7413479731805152, '13': 0.7235022610469328, '14': 0.743344738064137, '15': 0.7043403683230319, '16': 0.7043403683230319, '17': 0.6715375435300593, '18': 0.6537322868885153, '19': 0.6337732582779062, '20': 0.6137229117361529, '21': 0.5946927171814821, '22': 0.5831072781637627, '23': 0.5676966887657086, '24': 0.5532547674169525, '25': 0.5458932111384283, '26': 0.5323757152775764, '27': 0.52410215769548, '28': 0.5164105409872158, '29': 0.5064333489463827, '30': 0.5021615459302496, '31': 0.4956022325500163, '32': 0.49168419001122166, '33': 0.48454399849039026, '34': 0.48115199443049195, '35': 0.48110673492026645, '36': 0.4767366385448141, '37': 0.4906776446828658}.values())
    precavg = list({'1': 0.16362552815368858, '2': 0.17594613644760992, '3': 0.23550002217903365, '4': 0.4228959923956177, '5': 0.37691421147517506, '6': 0.3522787456577938, '7': 0.35838361994494694, '8': 0.33993085273301044, '9': 0.41119267469699455, '10': 0.39335003997157914, '11': 0.3751335218770738, '12': 0.3626621377614801, '13': 0.3631662702762777, '14': 0.35301215435324146, '15': 0.34955266902006266, '16': 0.347336152469638, '17': 0.38534133509993035, '18': 0.39089432753763614, '19': 0.38655309001775917, '20': 0.38124093621679267, '21': 0.38071636169735446, '22': 0.37588883358779024, '23': 0.3743203216401323, '24': 0.3709962985259346, '25': 0.3814410778736636, '26': 0.37865745857887945, '27': 0.4010622814596801, '28': 0.39396095805657266, '29': 0.41469088645711905, '30': 0.41035881103080507, '31': 0.40903161907670976, '32': 0.4276471259578972, '33': 0.4449379113951729, '34': 0.445956440812156, '35': 0.46157351009331526, '36': 0.47652980147961205, '37': 0.47652980147961205}.values())
    precDMM = list({'1': 0.16362552815368858, '2': 0.5743547155995292, '3': 0.5743547155995292, '4': 0.5216857094368681, '5': 0.6160334189864161, '6': 0.6792159961007496, '7': 0.7810115101731107, '8': 0.807495321322383, '9': 0.7587976034481012, '10': 0.7308746342605513, '11': 0.688714577715407, '12': 0.6839980114240531, '13': 0.6575222458369456, '14': 0.6301642813690883, '15': 0.6082279875681864, '16': 0.63224835494847, '17': 0.6087494121540211, '18': 0.6303879473336863, '19': 0.6155321769330699, '20': 0.5937863255178258, '21': 0.5788588821121403, '22': 0.5627953202958579, '23': 0.5473010885039723, '24': 0.5447175097737499, '25': 0.5321290531511297, '26': 0.550124089568394, '27': 0.5500831132070881, '28': 0.550124089568394, '29': 0.550124089568394, '30': 0.5157800509203719, '31': 0.5055215007989177, '32': 0.49834642670338164, '33': 0.49834642670338164, '34': 0.48792512780447433, '35': 0.48453343445159924, '36': 0.476529801479612, '37': 0.476529801479612}.values())
    precavg_FT = list({'1': 0.15551664221436445, '2': 0.17594613644760992, '3': 0.27906249062461824, '4': 0.45885762931546215, '5': 0.5670861034523698, '6': 0.6392384195436415, '7': 0.6886449171867455, '8': 0.6468902767436375, '9': 0.5964565890825004, '10': 0.5600875629185345, '11': 0.5992752710331716, '12': 0.5715951661357377, '13': 0.6040435577548815, '14': 0.5738932632957655, '15': 0.5539344286285107, '16': 0.5381335189864513, '17': 0.5229674304577442, '18': 0.5148001962726644, '19': 0.4993764694132292, '20': 0.4929188895079973, '21': 0.48038102223625956, '22': 0.4729462812645649, '23': 0.4601953530283593, '24': 0.4532932036063189, '25': 0.443585000816335, '26': 0.43939250890429093, '27': 0.43330792983448313, '28': 0.43015109114066163, '29': 0.4371153908330237, '30': 0.4557158405276701, '31': 0.4529255185898049, '32': 0.44766936562348913, '33': 0.4486359882280077, '34': 0.446300102117003, '35': 0.46225518032150886, '36': 0.4767366385448141, '37': 0.4906776446828658}.values())
    precDMM_FT = list({'1': 0.15551664221436445, '2': 0.5809342867708414, '3': 0.7206228578472276, '4': 0.7867381191467632, '5': 0.6902597119706742, '6': 0.6367524033845364, '7': 0.6873809674981087, '8': 0.7256363787089212, '9': 0.7555799901493097, '10': 0.7154827704985667, '11': 0.739855943758742, '12': 0.6942835390949639, '13': 0.6619933240328084, '14': 0.6313342528478108, '15': 0.6013745983464406, '16': 0.5826086780970106, '17': 0.5648264037382704, '18': 0.5612638817832433, '19': 0.582868356552131, '20': 0.57252279948084, '21': 0.5628895648059707, '22': 0.5506920285374899, '23': 0.5699034127598913, '24': 0.5576873424109157, '25': 0.5606644800032454, '26': 0.5533010715590378, '27': 0.5410048320917927, '28': 0.5292969219200974, '29': 0.5237768853178129, '30': 0.5237768853178129, '31': 0.5088844859147336, '32': 0.5088844859147336, '33': 0.4989901017338015, '34': 0.48889338111806163, '35': 0.4841759492405164, '36': 0.476529801479612, '37': 0.476529801479612}.values())

    F1dfg = list({'1': 0.26916418408582554, '2': 0.5024746519741774, '3': 0.5026353466832513, '4': 0.5027156940377882, '5': 0.5038812673081683, '6': 0.5392631887865204, '7': 0.5831477253750573, '8': 0.6116207387332719, '9': 0.6031264634832043, '10': 0.6280793174564742, '11': 0.6133485798457504, '12': 0.6133485798457504, '13': 0.6127876824383685, '14': 0.6311914267583875, '15': 0.6068457042178631, '16': 0.6068457042178631, '17': 0.594582383691367, '18': 0.5875035242040896, '19': 0.5786591128443818, '20': 0.5682440317807216, '21': 0.5577853493809855, '22': 0.555177264355282, '23': 0.5467903168144113, '24': 0.5384486642757597, '25': 0.5381653752741437, '26': 0.5294119967988663, '27': 0.5271233033358338, '28': 0.5250733048525962, '29': 0.5190441658881388, '30': 0.5190714278825467, '31': 0.5170998598755626, '32': 0.5177151180100732, '33': 0.513725869221786, '34': 0.514356645464753, '35': 0.5174466034680868, '36': 0.5156674896327189, '37': 0.5259709246856307}.values())
    F1avg = list({'1': 0.28122493532811643, '2': 0.2955789767190374, '3': 0.36637012083352954, '4': 0.4538207896697173, '5': 0.42576274483533344, '6': 0.41394799567205814, '7': 0.43034629821692194, '8': 0.41969498499923064, '9': 0.4620775483768625, '10': 0.45290554269629413, '11': 0.4402784956186591, '12': 0.4335827376220445, '13': 0.44110456082306865, '14': 0.4340486711194545, '15': 0.43580865659369983, '16': 0.4363633470140713, '17': 0.4604863821842012, '18': 0.47334111642054527, '19': 0.47218582309499146, '20': 0.4703540143011347, '21': 0.4735175431525046, '22': 0.4710603688441925, '23': 0.47233633014729326, '24': 0.47033749630245314, '25': 0.47788274295423044, '26': 0.4769993446772038, '27': 0.48904679509286997, '28': 0.48346497806851785, '29': 0.4961956685908414, '30': 0.4934042050257624, '31': 0.49475156168566237, '32': 0.506621456574169, '33': 0.5132003326967771, '34': 0.5164151810169141, '35': 0.5272860211761581, '36': 0.5266102077018594, '37': 0.5266102077018594}.values())
    F1DMM = list({'1': 0.28122493532811643, '2': 0.49869886575319833, '3': 0.49869886575319833, '4': 0.5451150596615892, '5': 0.6053830367105274, '6': 0.6465950311280898, '7': 0.6883233683084616, '8': 0.7095549158592249, '9': 0.6897561962343509, '10': 0.6830310941690592, '11': 0.658857811744176, '12': 0.658867049315235, '13': 0.6466778515844869, '14': 0.6304479842119978, '15': 0.6191146821467403, '16': 0.6364759261744721, '17': 0.6208219006095727, '18': 0.626538704387943, '19': 0.6176947726167346, '20': 0.6014956160927946, '21': 0.5935934270987497, '22': 0.5829816911274222, '23': 0.5723109254038301, '24': 0.5772926201136038, '25': 0.5685933166065238, '26': 0.566068832740942, '27': 0.5660197819864436, '28': 0.566068832740942, '29': 0.566068832740942, '30': 0.5478400164993481, '31': 0.5404549043211199, '32': 0.5366223122591083, '33': 0.5366223122591083, '34': 0.5324002672203704, '35': 0.5324794670564359, '36': 0.5266102077018592, '37': 0.5266102077018592}.values())
    F1avg_FT = list({'1': 0.26916418408582554, '2': 0.2955789767190374, '3': 0.4276765172921566, '4': 0.501688480124177, '5': 0.5019421313196215, '6': 0.5021112321165844, '7': 0.5326914555539695, '8': 0.5295990747425375, '9': 0.5055914626042332, '10': 0.4920680655009278, '11': 0.5210556806590957, '12': 0.5123947895621163, '13': 0.5380917245885903, '14': 0.5214683841225672, '15': 0.5146678057390386, '16': 0.5112790962074196, '17': 0.5068244227374452, '18': 0.5062487566639601, '19': 0.4976659109767936, '20': 0.4996220214369491, '21': 0.4932396964790382, '22': 0.49190337324417005, '23': 0.483344165775664, '24': 0.48088667211297514, '25': 0.47556721739262436, '26': 0.47573817570138593, '27': 0.47287056873142164, '28': 0.47407291496461085, '29': 0.4804486304002065, '30': 0.49285550949406304, '31': 0.4940977975502338, '32': 0.4915463589279912, '33': 0.4955148079174894, '34': 0.49668120361058277, '35': 0.5074785463107911, '36': 0.5156674896327189, '37': 0.5259709246856307}.values())
    F1DMM_FT = list({'1': 0.26916418408582554, '2': 0.5024746519741774, '3': 0.5026353466832513, '4': 0.5560197090570085, '5': 0.5381422382587449, '6': 0.537647242129398, '7': 0.5766878978595946, '8': 0.6104087787404302, '9': 0.6373250259043715, '10': 0.6243877642201857, '11': 0.6405598316456407, '12': 0.6133073927999871, '13': 0.5983960324591912, '14': 0.5821075652529045, '15': 0.5636580581069953, '16': 0.557207457802379, '17': 0.5500511160032895, '18': 0.5537317900368078, '19': 0.5700815734922065, '20': 0.5664006164883316, '21': 0.5649904976165018, '22': 0.5585983985371668, '23': 0.5733069284525271, '24': 0.5673317287484575, '25': 0.5709972061023947, '26': 0.5696193236170204, '27': 0.5612144100259046, '28': 0.5530667682869246, '29': 0.5523184810782499, '30': 0.5523184810782499, '31': 0.5445802845859945, '32': 0.5445802845859945, '33': 0.5421456093499656, '34': 0.5344848404795344, '35': 0.5321680880485432, '36': 0.5266102077018592, '37': 0.5266102077018592}.values())

    fig10 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, F1DMM, linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, F1DMM_FT, linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(x_axis, F1avg, linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(x_axis, F1avg_FT, linestyle="-", marker="s", linewidth=1)
    l5 = plt.plot(x_axis, F1dfg, color="b", linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis, fontsize=7)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5], labels=['Leven-DMM', 'Behav. TR.-DMM', 'Leven-UPGMA', 'Behav. TR.-UPGMA'],
               loc='best')
    # plt.title('Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'F1-allmethods' + '.svg')

    fig11 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, F1DMM, linestyle="-", linewidth=2)
    l2 = plt.plot(x_axis, F1DMM_FT, linestyle="-", linewidth=2)
    l3 = plt.plot(x_axis, F1avg, linestyle="-", linewidth=2)
    l4 = plt.plot(x_axis, F1avg_FT, linestyle="-", linewidth=2)
    l5 = plt.plot(x_axis, F1dfg, color="b", linestyle="-", linewidth=2)
    plt.xticks(x_axis, fontsize=7)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5],
               labels=['Leven-DMM', 'Behav. TR.-DMM', 'Leven-UPGMA', 'Behav. TR.-UPGMA', 'Behav. CL.'],
               loc='best')
    # plt.title('Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'F1-allmethods-line' + '.svg')

    #fit
    fig10 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, fitDMM, linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, fitDMM_FT, linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(x_axis, fitavg, linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(x_axis, fitavg_FT, linestyle="-", marker="s", linewidth=1)
    l5 = plt.plot(x_axis, fitdfg, color="b", linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis, fontsize=7)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5], labels=['Leven-DMM', 'Behav. TR.-DMM', 'Leven-UPGMA', 'Behav. TR.-UPGMA'],
               loc='best')
    # plt.title('Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Fitness")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Fitness-allmethods' + '.svg')

    fig11 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, fitDMM, linestyle="-", linewidth=2)
    l2 = plt.plot(x_axis, fitDMM_FT, linestyle="-", linewidth=2)
    l3 = plt.plot(x_axis, fitavg, linestyle="-", linewidth=2)
    l4 = plt.plot(x_axis, fitavg_FT, linestyle="-", linewidth=2)
    l5 = plt.plot(x_axis, fitdfg, color="b", linestyle="-", linewidth=2)
    plt.xticks(x_axis, fontsize=7)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5],
               labels=['Leven-DMM', 'Behav. TR.-DMM', 'Leven-UPGMA', 'Behav. TR.-UPGMA', 'Behav. CL.'],
               loc='best')
    # plt.title('Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Fitness")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Fitness-allmethods-line' + '.svg')

    # prec
    fig10 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, precDMM, linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, precDMM_FT, linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(x_axis, precavg, linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(x_axis, precavg_FT, linestyle="-", marker="s", linewidth=1)
    l5 = plt.plot(x_axis, precdfg, color="b", linestyle="-", marker="s", linewidth=1)
    plt.xticks(x_axis, fontsize=7)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5], labels=['Leven-DMM', 'Behav. TR.-DMM', 'Leven-UPGMA', 'Behav. TR.-UPGMA'],
               loc='best')
    # plt.title('Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Precision")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Precision-allmethods' + '.svg')

    fig11 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, precDMM, linestyle="-", linewidth=2)
    l2 = plt.plot(x_axis, precDMM_FT, linestyle="-", linewidth=2)
    l3 = plt.plot(x_axis, precavg, linestyle="-", linewidth=2)
    l4 = plt.plot(x_axis, precavg_FT, linestyle="-", linewidth=2)
    l5 = plt.plot(x_axis, precdfg, color="b", linestyle="-", linewidth=2)
    plt.xticks(x_axis, fontsize=7)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5],
               labels=['Leven-DMM', 'Behav. TR.-DMM', 'Leven-UPGMA', 'Behav. TR.-UPGMA', 'Behav. CL.'],
               loc='best')
    # plt.title('Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Precision")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Precision-allmethods-line' + '.svg')

    # fig10 = plt.figure()
    # rc('text', usetex=True)
    # rc('font', family='serif')
    # l1 = plt.plot(x_axis, F1DMM[1], linestyle="-", marker="s", linewidth=1)
    # l2 = plt.plot(x_axis, F1DMM_FT[1], linestyle="-", marker="s", linewidth=1)
    # l3 = plt.plot(x_axis, F1avg[1], linestyle="-", marker="s", linewidth=1)
    # l4 = plt.plot(x_axis, F1avg_FT[1], linestyle="-", marker="s", linewidth=1)
    # l5 = plt.plot(x_axis, F1dfg, color="b", linestyle="-", marker="s", linewidth=1)
    # plt.xticks(x_axis, fontsize=7)
    # # plt.gca().invert_xaxis()
    # plt.ylim(0, 1.04)
    # plt.legend([l1, l2, l3, l4, l5], labels=['Leven-DMM', 'Behav. TR.-DMM', 'Leven-UPGMA', 'Behav. TR.-UPGMA'],
    #            loc='best')
    # # plt.title('Recomputing')
    # plt.xlabel("Num. of Cluster")
    # plt.ylabel("F1-Score")
    # plt.grid(axis='y')
    # plt.savefig(PIC_PATH + 'Recomputing' + '.svg')
    #
    # fig11 = plt.figure()
    # rc('text', usetex=True)
    # rc('font', family='serif')
    # l1 = plt.plot(x_axis, F1DMM[1], linestyle="-", linewidth=2)
    # l2 = plt.plot(x_axis, F1DMM_FT[1], linestyle="-", linewidth=2)
    # l3 = plt.plot(x_axis, F1avg[1], linestyle="-", linewidth=2)
    # l4 = plt.plot(x_axis, F1avg_FT[1], linestyle="-", linewidth=2)
    # l5 = plt.plot(x_axis, F1dfg, color="b", linestyle="-", linewidth=2)
    # plt.xticks(x_axis, fontsize=7)
    # # plt.gca().invert_xaxis()
    # plt.ylim(0, 1.04)
    # plt.legend([l1, l2, l3, l4, l5],
    #            labels=['Leven-DMM', 'Behav. TR.-DMM', 'Leven-UPGMA', 'Behav. TR.-UPGMA', 'Behav. CL.'],
    #            loc='best')
    # # plt.title('Recomputing')
    # plt.xlabel("Num. of Cluster")
    # plt.ylabel("F1-Score")
    # plt.grid(axis='y')
    # plt.savefig(PIC_PATH + 'Recomputing-line' + '.svg')

    # fig10 = plt.figure()
    # rc('text', usetex=True)
    # rc('font', family='serif')
    # l1 = plt.plot(x_axis, F1dfg, color="b", linestyle="-", marker="s", linewidth=1)
    # l2 = plt.plot(x_axis, F1DMM[1], color="r", linestyle="-", marker="s", linewidth=1)
    # l3 = plt.plot(x_axis, F1DMM_FT[1], color="g", linestyle="-", marker="s", linewidth=1)
    # plt.xticks(x_axis,fontsize=7)
    # plt.ylim(0, 1.04)
    # plt.legend([l1, l2, l3], labels=['DFG', 'Leven-DMM', 'Feature Vector-DMM'], loc='best')
    # plt.title('DFG,Leven and Feature Vector-Recomputing')
    # plt.xlabel("Num. of Cluster")
    # plt.ylabel("F1-Score")
    # plt.grid(axis='y')
    # # plt.show()
    # plt.savefig(PIC_PATH + 'DFG-Leven-FT-DMM' + '.svg')
    #
    # fig11 = plt.figure()
    # rc('text', usetex=True)
    # rc('font', family='serif')
    # l1 = plt.plot(x_axis, F1dfg, color="b", linestyle="-", marker="s", linewidth=1)
    # l2 = plt.plot(x_axis, F1avg[1], color="r", linestyle="-", marker="s", linewidth=1)
    # l3 = plt.plot(x_axis, F1avg_FT[1], color="g", linestyle="-", marker="s", linewidth=1)
    # plt.xticks(x_axis,fontsize=7)
    # plt.ylim(0, 1.04)
    # plt.legend([l1, l2, l3], labels=['DFG', 'Leven-AVG', 'Feature Vector-AVG'], loc='best')
    # plt.title('DFG,Leven and Feature Vector-Recomputing')
    # plt.xlabel("Num. of Cluster")
    # plt.ylabel("F1-Score")
    # plt.grid(axis='y')
    # # plt.show()
    # plt.savefig(PIC_PATH + 'DFG-Leven-FT-AVG' + '.svg')


def example_run(LOG_PATH, ATTR_NAME, METHOD, PIC_PATH, plot_clu):
    percent = 1
    alpha = 0.5
    runtime = dict()

    log = xes_importer.apply(LOG_PATH)

    METHOD = METHOD
    # plot_clu = 23

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME + 'update'
    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha, runtime,
                                                    plot_clu)
    F1valup = list(plot_F1.values())
    x_axis = range(1, plot_clu + 1)
    merge_log.five_plots(plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length, plot_clu,
                         x_axis, PIC_PATH, TYPE)


if __name__ == "__main__":

    LOG_PATH = '/home/yukun/dataset/Receipt.xes'
    ATTR_NAME = 'responsible'
    PIC_PATH = '/home/yukun/resultlog/'
    METHOD = 'dfg'
    plot_clu = 37
    # example_run(LOG_PATH, ATTR_NAME, METHOD, PIC_PATH, plot_clu)
    percent = 1
    alpha = 0.5
    runtime = dict()

    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    log = xes_importer.apply(LOG_PATH)
    TYPE = METHOD + ATTR_NAME + 'update'

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha, runtime, plot_clu)
    F1valup = list(plot_F1.values())

    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc(log, ATTR_NAME, METHOD, TYPE, PIC_PATH, percent, alpha, runtime, plot_clu)

    F1val = list(plot_F1.values())
    F1dfg = [F1val, F1valup]
    print('F1compare', F1dfg)
    x_axis = range(1, plot_clu + 1)

    fig9 = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(x_axis, F1dfg[0], color="b", linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(x_axis, F1dfg[1], color="r", linestyle="-", marker="o", linewidth=1)
    plt.xticks(x_axis, fontsize=7)
    # plt.gca().invert_xaxis()
    plt.ylim(0, 1.04)
    plt.legend([l1, l2], labels=['Behav. CL.', 'Behav. CL-recomputation'], loc='best')
    # plt.title('Leven-Recomputing')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Behav. CL-recomputationornot' + '.svg')


    # LevenDMMre=list( {'1': 0.28122493532811643, '2': 0.49869886575319833, '3': 0.49869886575319833, '4': 0.5451150596615892, '5': 0.6053830367105274, '6': 0.6465950311280898, '7': 0.6883233683084616, '8': 0.7095549158592249, '9': 0.6897561962343509, '10': 0.6830310941690592, '11': 0.658857811744176, '12': 0.658867049315235, '13': 0.6466778515844869, '14': 0.6304479842119978, '15': 0.6191146821467403, '16': 0.6364759261744721, '17': 0.6208219006095727, '18': 0.626538704387943, '19': 0.6176947726167346, '20': 0.6014956160927946, '21': 0.5935934270987497, '22': 0.5829816911274222, '23': 0.5723109254038301, '24': 0.5772926201136038, '25': 0.5685933166065238, '26': 0.566068832740942, '27': 0.5660197819864436, '28': 0.566068832740942, '29': 0.566068832740942, '30': 0.5478400164993481, '31': 0.5404549043211199, '32': 0.5366223122591083, '33': 0.5366223122591083, '34': 0.5324002672203704, '35': 0.5324794670564359, '36': 0.5266102077018592, '37': 0.5266102077018592}.values())
    # LevenDMMno = list({'1': 0.26916418408582554, '2': 0.49869886575319833, '3': 0.5737073667088783, '4': 0.5389721590465493, '5': 0.580836921453319, '6': 0.6251065920288125, '7': 0.6576135079126824, '8': 0.6927484933474337, '9': 0.7111292994993375, '10': 0.6817288682989381, '11': 0.6468064899990202, '12': 0.6187898947986361, '13': 0.6096835551076263, '14': 0.6301986287849711, '15': 0.6321166310390985, '16': 0.6258197440513649, '17': 0.6102183362403376, '18': 0.6050255087226127, '19': 0.5979299626603916, '20': 0.5851357498157108, '21': 0.5744443698637763, '22': 0.563792425241625, '23': 0.5675286436584379, '24': 0.5582790165196277, '25': 0.5478538278522304, '26': 0.5427399561491275, '27': 0.536421478113581, '28': 0.5326593069178867, '29': 0.5321235426033298, '30': 0.5305531375126018, '31': 0.5307542659857405, '32': 0.5283368984672749, '33': 0.5332924954360968, '34': 0.5324002672203704, '35': 0.5324794670564359, '36': 0.5266102077018593, '37': 0.5266102077018593}.values())
    # FTDMMre= list({'1': 0.26916418408582554, '2': 0.5024746519741774, '3': 0.5026353466832513, '4': 0.5560197090570085,
    #  '5': 0.5381422382587449, '6': 0.537647242129398, '7': 0.5766878978595946, '8': 0.6104087787404302,
    #  '9': 0.6373250259043715, '10': 0.6243877642201857, '11': 0.6405598316456407, '12': 0.6133073927999871,
    #  '13': 0.5983960324591912, '14': 0.5821075652529045, '15': 0.5636580581069953, '16': 0.557207457802379,
    #  '17': 0.5500511160032895, '18': 0.5537317900368078, '19': 0.5700815734922065, '20': 0.5664006164883316,
    #  '21': 0.5649904976165018, '22': 0.5585983985371668, '23': 0.5733069284525271, '24': 0.5673317287484575,
    #  '25': 0.5709972061023947, '26': 0.5696193236170204, '27': 0.5612144100259046, '28': 0.5530667682869246,
    #  '29': 0.5523184810782499, '30': 0.5523184810782499, '31': 0.5445802845859945, '32': 0.5445802845859945,
    #  '33': 0.5421456093499656, '34': 0.5344848404795344, '35': 0.5321680880485432, '36': 0.5266102077018592,
    #  '37': 0.5266102077018592}.values())
    # FTDMMno =list({'1': 0.26916418408582554, '2': 0.5024746519741774, '3': 0.5026353466832513, '4': 0.5560197090570085, '5': 0.5518502195421394, '6': 0.537647242129398, '7': 0.5766878978595946, '8': 0.6104087787404302, '9': 0.5990247376651062, '10': 0.6243877642201857, '11': 0.6405598316456407, '12': 0.613307392799987, '13': 0.5983960324591912, '14': 0.5821075652529045, '15': 0.5739969577849199, '16': 0.5766411009612762, '17': 0.5683416036822515, '18': 0.553731790036808, '19': 0.5471874506506291, '20': 0.5446768394816057, '21': 0.5443011861815246, '22': 0.5457497175722271, '23': 0.5596031070555816, '24': 0.5571366225455786, '25': 0.5518257303586007, '26': 0.543781919831612, '27': 0.5465297023305362, '28': 0.5389065144378195, '29': 0.5512509307189447, '30': 0.5465816970693044, '31': 0.5435816084434187, '32': 0.542482458416047, '33': 0.5421456093499655, '34': 0.5395354004101091, '35': 0.5321680880485432, '36': 0.5266102077018593, '37': 0.5266102077018593}.values())
    #
    # x_axis = range(1,38)
    # font1 = {'family': 'Times New Roman',
    #          'weight': 'normal'}
    # fig9,ax = plt.subplots()
    # l1 = plt.plot(x_axis, FTDMMre, color="b", linestyle="-", marker="s", linewidth=1)
    # l2 = plt.plot(x_axis, FTDMMno, color="r", linestyle="-", marker="o", linewidth=1)
    # plt.xticks(x_axis,fontsize=7)
    # # plt.gca().invert_xaxis()
    # plt.ylim(0, 1.04)
    # plt.legend([l1, l2], labels=['Behav. TR.-DMM-recomputation', 'Behav. TR.-DMM'], loc='best',prop=font1)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]
    # # plt.title('Leven-Recomputing')
    # plt.xlabel("Num. of Cluster",font1)
    # plt.ylabel("F1-Score",font1)
    # plt.grid(axis='y')
    # plt.savefig('Behav. TR.-DMM-recomputeornot' + '.svg')
    # plt.show()

    # LOG_PATH = "/home/yukun/pm4py-source/tests/input_data/receipt.xes"
    # ATTR_NAME = 'responsible'
    # PIC_PATH = '/home/yukun/resultlog/receipt_all/' + ATTR_NAME + '/'
    # # LOG_PATH = "../../tests/input_data/receipt.xes"
    # # ATTR_NAME = 'responsible'
    # # PIC_PATH = '../example/real_log/'
    # plot_clu = 37
    # # METHOD = 'dfg'
    # # example_run(LOG_PATH, ATTR_NAME, METHOD, PIC_PATH, plot_clu)
    # standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH, plot_clu)
    #
    # LOG_PATH = "/home/yukun/dataset/filteredbpic2017.xes"
    # ATTR_NAME = 'CreditScore'
    # PIC_PATH = '/home/yukun/resultlog/filteredbpic2017/' + ATTR_NAME + '/'
    # standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH, plot_clu)
    #
    # LOG_PATH = "/home/yukun/dataset/document_logs/Control_summary.xes"
    # ATTR_NAME = 'amount_applied0'
    # PIC_PATH = '/home/yukun/resultlog/Control_summary/' + ATTR_NAME + '/'
    # standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH,plot_clu)
    #
    # LOG_PATH = "/home/yukun/dataset/document_logs/Geo_parcel_document.xes"
    # ATTR_NAME = 'amount_applied0'
    # PIC_PATH = '/home/yukun/resultlog/Geo_parcel_document/' + ATTR_NAME + '/'
    # standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH, plot_clu)
    #
    # LOG_PATH = "/home/yukun/dataset/BPIC2012_A.xes"
    # ATTR_NAME = 'AMOUNT_REQ'
    # PIC_PATH = '/home/yukun/resultlog/BPIC2012_A/' + ATTR_NAME + '/'
    # standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH,plot_clu)
    #
    # LOG_PATH = "/home/yukun/dataset/document_logs/Payment_application.xes"
    # ATTR_NAME = 'amount_applied0'
    # PIC_PATH = '/home/yukun/resultlog/Payment_application/' + ATTR_NAME + '/'
    # standard_plt(LOG_PATH, ATTR_NAME, PIC_PATH,plot_clu)
