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
    plot_clu = 23
    runtime = dict()

    LOG_PATH = "/home/yukun/dataset/Receipt4.xes"
    # LOG_PATH = "D:/Sisc/19SS/thesis/Dataset/Receipt4.xes"
    ATTR_NAME = 'responsible'
    METHOD = 'DMM'

    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    fitall['Receipt'] = list(plot_fit.values())
    precall['Receipt'] = list(plot_prec.values())
    F1all['Receipt'] = list(plot_F1.values())

    LOG_PATH = "/home/yukun/dataset/filteredbpic2017.xes"
    ATTR_NAME = 'CreditScore'
    METHOD = 'dfg'
    plot_clu = 50

    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    fitall['filteredbpic2017'] = list(plot_fit.values())
    precall['filteredbpic2017'] = list(plot_prec.values())
    F1all['filteredbpic2017'] = list(plot_F1.values())

    LOG_PATH = "/home/yukun/dataset/BPIC2012_A.xes"
    ATTR_NAME = 'AMOUNT_REQ'
    METHOD = 'dfg'
    plot_clu = 50

    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    fitall['BPIC2012_A'] = list(plot_fit.values())
    precall['BPIC2012_A'] = list(plot_prec.values())
    F1all['BPIC2012_A'] = list(plot_F1.values())

    LOG_PATH = "/home/yukun/dataset/document_logs/Control_summary.xes"
    ATTR_NAME = 'amount_applied0'
    METHOD = 'dfg'
    plot_clu = 50

    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    fitall['Control_summary'] = list(plot_fit.values())
    precall['Control_summary'] = list(plot_prec.values())
    F1all['Control_summary'] = list(plot_F1.values())

    LOG_PATH = "/home/yukun/dataset/document_logs/Geo_parcel_document.xes"
    ATTR_NAME = 'amount_applied0'
    METHOD = 'dfg'
    plot_clu = 50
    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha, runtime, plot_clu)
    fitall['Geo'] = list(plot_fit.values())
    precall['Geo'] = list(plot_prec.values())
    F1all['Geo'] = list(plot_F1.values())



    LOG_PATH = "/home/yukun/dataset/document_logs/Payment_application.xes"
    ATTR_NAME = 'amount_applied0'
    METHOD = 'dfg'
    plot_clu = 50
    log = xes_importer.apply(LOG_PATH)
    print(LOG_PATH)
    print(ATTR_NAME)
    print(METHOD)
    TYPE = METHOD + ATTR_NAME

    (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
     runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    fitall['Payment_application'] = list(plot_fit.values())
    precall['Payment_application'] = list(plot_prec.values())
    F1all['Payment_application'] = list(plot_F1.values())

    print('fitall', fitall)
    print('precall', precall)
    print('F1all', F1all)

    # fitall['Receipt'] = list(
    #     {'1': 0.9871623451004897, '2': 0.7252854590998924, '3': 0.622014337783979, '4': 0.7084464941912509,
    #      '5': 0.6906560391814989, '6': 0.6969888284425486, '7': 0.7204002817447476, '8': 0.7531007010167994,
    #      '9': 0.7286199467492621, '10': 0.7536801188727913, '11': 0.7738693163406307, '12': 0.7921887839661829,
    #      '13': 0.8035380323701417, '14': 0.8076835640902577, '15': 0.8192885707113948, '16': 0.8277898308555318,
    #      '17': 0.8056427698986632, '18': 0.8152860420145461, '19': 0.823761663413301, '20': 0.8291697032472223,
    #      '21': 0.8363150349013811, '22': 0.8380166532007617, '23': 0.8421102029193948}.values())
    # precall['Receipt'] = list(
    #     {'1': 0.13982676330210364, '2': 0.48924671498438516, '3': 0.7035819178559121, '4': 0.5957366971770063,
    #      '5': 0.674726003704338, '6': 0.6459338196372995, '7': 0.6973211044713399, '8': 0.6444129530805764,
    #      '9': 0.6129890875263273, '10': 0.5858799483949363, '11': 0.5528996170091997, '12': 0.5223845121749725,
    #      '13': 0.5250330255360288, '14': 0.5033096777080136, '15': 0.5327862934597235, '16': 0.5330304862635773,
    #      '17': 0.5660093757949332, '18': 0.5535587269292378, '19': 0.5452238487399109, '20': 0.5689419548070729,
    #      '21': 0.5581303234454227, '22': 0.5448693092568266, '23': 0.5645114898245286}.values())
    # F1all['Receipt'] = list(
    #     {'1': 0.2449566096779114, '2': 0.42095976845683253, '3': 0.4867194773602561, '4': 0.4712624038473022,
    #      '5': 0.5294707675068432, '6': 0.5414056490672706, '7': 0.5980646943382399, '8': 0.5768717207909952,
    #      '9': 0.5679462682101176, '10': 0.5618349696996708, '11': 0.5437736353333467, '12': 0.5242328280156981,
    #      '13': 0.5376983725140169, '14': 0.5246877858440896, '15': 0.5539263954608077, '16': 0.562260682886589,
    #      '17': 0.58465881804464, '18': 0.5803351772146961, '19': 0.5784845899730752, '20': 0.5986430883437075,
    #      '21': 0.5942711928561809, '22': 0.5842300138225129, '23': 0.6007124650633123}.values())
    #
    # fitall['filteredbpic2017'] = list(
    #     {'1': 1.0, '2': 0.8104569668568641, '3': 0.8553512598093528, '4': 0.8553512598093528, '5': 0.8586785040569399,
    #      '6': 0.8513375879090131, '7': 0.8446819224953785, '8': 0.8446819224953785, '9': 0.8446819224953785,
    #      '10': 0.8446819224953785, '11': 0.8446819224953785, '12': 0.8446819224953785, '13': 0.8446819224953785,
    #      '14': 0.8446819224953785, '15': 0.8446819224953785, '16': 0.8446819224953785, '17': 0.8446819224953785,
    #      '18': 0.8446819224953785, '19': 0.8446819224953785, '20': 0.8446819224953785, '21': 0.8446819224953785,
    #      '22': 0.8446819224953785, '23': 0.8446819224953785, '24': 0.8446819224953785, '25': 0.8446819224953785,
    #      '26': 0.8446819224953785, '27': 0.8446819224953785, '28': 0.8446819224953785, '29': 0.8446819224953785,
    #      '30': 0.8446819224953785, '31': 0.8446819224953785, '32': 0.8446819224953785, '33': 0.8446819224953785,
    #      '34': 0.8446819224953785, '35': 0.8446819224953785, '36': 0.8446819224953785, '37': 0.8446819224953785,
    #      '38': 0.8446819224953785, '39': 0.8446819224953785, '40': 0.8446819224953785, '41': 0.8446819224953785,
    #      '42': 0.8446819224953785, '43': 0.8446819224953785, '44': 0.8446819224953785, '45': 0.8446819224953785,
    #      '46': 0.8446819224953785, '47': 0.8446819224953785, '48': 0.8446819224953785, '49': 0.8446819224953785,
    #      '50': 0.8446819224953785}.values())
    # precall['filteredbpic2017'] = list(
    #     {'1': 0.7625744049007841, '2': 0.7270718288726079, '3': 0.8180478859150719, '4': 0.8180478859150719,
    #      '5': 0.8884890732929508, '6': 0.9070742277441256, '7': 0.9184295981108496, '8': 0.9184295981108496,
    #      '9': 0.9184295981108496, '10': 0.9184295981108496, '11': 0.9184295981108496, '12': 0.9184295981108496,
    #      '13': 0.9184295981108496, '14': 0.9184295981108496, '15': 0.9184295981108496, '16': 0.9184295981108496,
    #      '17': 0.9184295981108496, '18': 0.9184295981108496, '19': 0.9184295981108496, '20': 0.9184295981108496,
    #      '21': 0.9184295981108496, '22': 0.9184295981108496, '23': 0.9184295981108496, '24': 0.9184295981108496,
    #      '25': 0.9184295981108496, '26': 0.9184295981108496, '27': 0.9184295981108496, '28': 0.9184295981108496,
    #      '29': 0.9184295981108496, '30': 0.9184295981108496, '31': 0.9184295981108496, '32': 0.9184295981108496,
    #      '33': 0.9184295981108496, '34': 0.9184295981108496, '35': 0.9184295981108496, '36': 0.9184295981108496,
    #      '37': 0.9184295981108496, '38': 0.9184295981108496, '39': 0.9184295981108496, '40': 0.9184295981108496,
    #      '41': 0.9184295981108496, '42': 0.9184295981108496, '43': 0.9184295981108496, '44': 0.9184295981108496,
    #      '45': 0.9184295981108496, '46': 0.9184295981108496, '47': 0.9184295981108496, '48': 0.9184295981108496,
    #      '49': 0.9184295981108496, '50': 0.9184295981108496}.values())
    # F1all['filteredbpic2017'] = list(
    #     {'1': 0.8652961290944307, '2': 0.759817958696615, '3': 0.8304774034638758, '4': 0.8304774034638758,
    #      '5': 0.8679499699712541, '6': 0.8729331033874778, '7': 0.8748609494364434, '8': 0.8748609494364434,
    #      '9': 0.8748609494364434, '10': 0.8748609494364434, '11': 0.8748609494364434, '12': 0.8748609494364434,
    #      '13': 0.8748609494364434, '14': 0.8748609494364434, '15': 0.8748609494364434, '16': 0.8748609494364434,
    #      '17': 0.8748609494364434, '18': 0.8748609494364434, '19': 0.8748609494364434, '20': 0.8748609494364434,
    #      '21': 0.8748609494364434, '22': 0.8748609494364434, '23': 0.8748609494364434, '24': 0.8748609494364434,
    #      '25': 0.8748609494364434, '26': 0.8748609494364434, '27': 0.8748609494364434, '28': 0.8748609494364434,
    #      '29': 0.8748609494364434, '30': 0.8748609494364434, '31': 0.8748609494364434, '32': 0.8748609494364434,
    #      '33': 0.8748609494364434, '34': 0.8748609494364434, '35': 0.8748609494364434, '36': 0.8748609494364434,
    #      '37': 0.8748609494364434, '38': 0.8748609494364434, '39': 0.8748609494364434, '40': 0.8748609494364434,
    #      '41': 0.8748609494364434, '42': 0.8748609494364434, '43': 0.8748609494364434, '44': 0.8748609494364434,
    #      '45': 0.8748609494364434, '46': 0.8748609494364434, '47': 0.8748609494364434, '48': 0.8748609494364434,
    #      '49': 0.8748609494364434, '50': 0.8748609494364434}.values())
    #
    # fitall['BPIC2012_A'] = list(
    #     {'1': 1.0, '2': 0.7870184946099976, '3': 0.7263611528912973, '4': 0.7382501875008753, '5': 0.7382501875008753,
    #      '6': 0.7382501875008753, '7': 0.7382501875008753, '8': 0.6807429400428815, '9': 0.6940954909731176,
    #      '10': 0.670442765885155, '11': 0.670442765885155, '12': 0.6747738445403774, '13': 0.6911595691771494,
    #      '14': 0.7101695849212469, '15': 0.7101695849212469, '16': 0.7299148480035216, '17': 0.7473829010509813,
    #      '18': 0.7473829010509813, '19': 0.7473829010509813, '20': 0.742802084342006, '21': 0.750701303839001,
    #      '22': 0.750701303839001, '23': 0.7595031165653922, '24': 0.7595031165653922, '25': 0.7580081514700687,
    #      '26': 0.7580081514700687, '27': 0.7580081514700687, '28': 0.7497470879533229, '29': 0.7434738888364375,
    #      '30': 0.7434738888364375, '31': 0.7434738888364375, '32': 0.7488834591127954, '33': 0.7541880086217432,
    #      '34': 0.7541880086217432, '35': 0.7712045415969156, '36': 0.7761173804915987, '37': 0.7761173804915987,
    #      '38': 0.7705747515158572, '39': 0.7705747515158572, '40': 0.7704515530171308, '41': 0.7704515530171308,
    #      '42': 0.7745254518417252, '43': 0.7745254518417252, '44': 0.7745254518417252, '45': 0.7868070746959818,
    #      '46': 0.7868070746959818, '47': 0.7944896989213828, '48': 0.7983884752402365, '49': 0.7983884752402365,
    #      '50': 0.7983884752402365}.values())
    # precall['BPIC2012_A'] = list(
    #     {'1': 0.5213162855289565, '2': 0.7606581427644783, '3': 0.8090888742056728, '4': 0.83330423992627,
    #      '5': 0.83330423992627, '6': 0.83330423992627, '7': 0.83330423992627, '8': 0.9041405281122707,
    #      '9': 0.9137264753010437, '10': 0.92085562473997, '11': 0.92085562473997, '12': 0.9264011607768075,
    #      '13': 0.9254879906862173, '14': 0.8813600260616996, '15': 0.8813600260616996, '16': 0.8557876579065,
    #      '17': 0.825324643725972, '18': 0.825324643725972, '19': 0.825324643725972, '20': 0.8214852689012175,
    #      '21': 0.8045660980880079, '22': 0.8045660980880079, '23': 0.8078970117591103, '24': 0.8078970117591103,
    #      '25': 0.7998198834810466, '26': 0.7998198834810466, '27': 0.7998198834810466, '28': 0.8282284764406874,
    #      '29': 0.8337694933296975, '30': 0.8337694933296975, '31': 0.8337694933296975, '32': 0.8376497529835437,
    #      '33': 0.8262852061521329, '34': 0.8262852061521329, '35': 0.7921345803024243, '36': 0.7828341307543838,
    #      '37': 0.7828341307543838, '38': 0.7889899410710677, '39': 0.7889899410710677, '40': 0.7950410927560758,
    #      '41': 0.7950410927560758, '42': 0.7964591955852419, '43': 0.7964591955852419, '44': 0.7964591955852419,
    #      '45': 0.777543773285053, '46': 0.777543773285053, '47': 0.7598919147752967, '48': 0.7561007588825255,
    #      '49': 0.7561007588825255, '50': 0.7561007588825255}.values())
    # F1all['BPIC2012_A'] = list(
    #     {'1': 0.6853489842813278, '2': 0.7073654067451904, '3': 0.7134226080007423, '4': 0.743753229578779,
    #      '5': 0.743753229578779, '6': 0.743753229578779, '7': 0.743753229578779, '8': 0.7523563577120047,
    #      '9': 0.7668834524384994, '10': 0.7549021386295834, '11': 0.7549021386295834, '12': 0.7626790796850955,
    #      '13': 0.7731213495989305, '14': 0.7605115671346501, '15': 0.7605115671346501, '16': 0.7598481345462275,
    #      '17': 0.7499588102273879, '18': 0.7499588102273879, '19': 0.7499588102273879, '20': 0.7465664693292351,
    #      '21': 0.7397227707306248, '22': 0.7397227707306248, '23': 0.7461841964482369, '24': 0.7461841964482369,
    #      '25': 0.7437991747038967, '26': 0.7437991747038967, '27': 0.7437991747038967, '28': 0.7584776757533029,
    #      '29': 0.7570447247668975, '30': 0.7570447247668975, '31': 0.7570447247668975, '32': 0.7625392934285601,
    #      '33': 0.757844032916094, '34': 0.757844032916094, '35': 0.7441058887712979, '36': 0.7403611567895203,
    #      '37': 0.7403611567895203, '38': 0.7416911020537674, '39': 0.7416911020537674, '40': 0.7461492294706787,
    #      '41': 0.7461492294706787, '42': 0.7514111555392637, '43': 0.7514111555392637, '44': 0.7514111555392637,
    #      '45': 0.7454507644161456, '46': 0.7454507644161456, '47': 0.736898972766727, '48': 0.7366325501701166,
    #      '49': 0.7366325501701166, '50': 0.7366325501701166}.values())
    #
    # fitall['Control_summary'] = list(
    #     {'1': 0.9665845411315319, '2': 0.9129895957235955, '3': 0.9192972150989572, '4': 0.9218410942311387,
    #      '5': 0.9374642663391397, '6': 0.9365334326641938, '7': 0.9455902089846256, '8': 0.9523869399199769,
    #      '9': 0.9576695984741149, '10': 0.9585538876144083, '11': 0.9620974879769684, '12': 0.9652509670425178,
    #      '13': 0.9678844942249398, '14': 0.9701735208451027, '15': 0.969929452113899, '16': 0.969929452113899,
    #      '17': 0.969929452113899, '18': 0.969929452113899, '19': 0.969929452113899, '20': 0.969929452113899,
    #      '21': 0.9765690318937088, '22': 0.9758795368555352, '23': 0.9769252468577823, '24': 0.9778849013541727,
    #      '25': 0.9774309709365943, '26': 0.978296351550849, '27': 0.978296351550849, '28': 0.978296351550849,
    #      '29': 0.978296351550849, '30': 0.978296351550849, '31': 0.978296351550849, '32': 0.978296351550849,
    #      '33': 0.9817997994187563, '34': 0.9823330661095935, '35': 0.9828358604180969, '36': 0.9828358604180969,
    #      '37': 0.9828358604180969, '38': 0.9828358604180969, '39': 0.9845664228019447, '40': 0.9849505339045763,
    #      '41': 0.9853159078802505, '42': 0.9856638830951784, '43': 0.9859853467537558, '44': 0.9863022903936073,
    #      '45': 0.9866051476494653, '46': 0.986894837198547, '47': 0.987172199532774, '48': 0.9874380051030748,
    #      '49': 0.9876440354397992, '50': 0.9876440354397992}.values())
    # precall['Control_summary'] = list(
    #     {'1': 0.8673615228254425, '2': 0.9338208428867203, '3': 0.9167725376580095, '4': 0.9375794032435071,
    #      '5': 0.9497453617281953, '6': 0.9498151790110878, '7': 0.9570952611342326, '8': 0.9531481553592378,
    #      '9': 0.9583687137567227, '10': 0.9585646681666613, '11': 0.9623307427417324, '12': 0.9654695875866262,
    #      '13': 0.9681494554568313, '14': 0.9703651224581294, '15': 0.9696816237865061, '16': 0.9696816237865061,
    #      '17': 0.9696816237865061, '18': 0.9696816237865061, '19': 0.9696816237865061, '20': 0.9696816237865061,
    #      '21': 0.9726532029038063, '22': 0.9738962391354513, '23': 0.9750176116892575, '24': 0.9760035388408314,
    #      '25': 0.9714192271688706, '26': 0.9724925322945516, '27': 0.9724925322945516, '28': 0.9724925322945516,
    #      '29': 0.9724925322945516, '30': 0.9724925322945516, '31': 0.9724925322945516, '32': 0.9724925322945516,
    #      '33': 0.9786830303590964, '34': 0.9793100000544172, '35': 0.9799015361026503, '36': 0.9799015361026503,
    #      '37': 0.9799015361026503, '38': 0.9799015361026503, '39': 0.9819775119171578, '40': 0.9824296819094258,
    #      '41': 0.9828590106282116, '42': 0.9832548850381203, '43': 0.9836443063163036, '44': 0.9840160266272967,
    #      '45': 0.9843819394750449, '46': 0.9847116803315246, '47': 0.9850344253369201, '48': 0.9853447745061811,
    #      '49': 0.9855535862396566, '50': 0.9855535862396566}.values())
    # F1all['Control_summary'] = list(
    #     {'1': 0.914288872516205, '2': 0.9194128066851874, '3': 0.9151459029063848, '4': 0.9272212741475842,
    #      '5': 0.9416105330428348, '6': 0.9414976010120539, '7': 0.9499117243141296, '8': 0.9513350182426918,
    #      '9': 0.9567466178687616, '10': 0.9574131210165249, '11': 0.9611720170821894, '12': 0.9644050417268852,
    #      '13': 0.967136524765592, '14': 0.9694484648805463, '15': 0.9690380214684278, '16': 0.9690380214684278,
    #      '17': 0.9690380214684278, '18': 0.9690380214684278, '19': 0.9690380214684278, '20': 0.9690380214684278,
    #      '21': 0.9739884096684276, '22': 0.9742762207195816, '23': 0.9753856017038692, '24': 0.9763827850091392,
    #      '25': 0.9737650992560266, '26': 0.9747583835205496, '27': 0.9747583835205496, '28': 0.9747583835205496,
    #      '29': 0.9747583835205496, '30': 0.9747583835205496, '31': 0.9747583835205496, '32': 0.9747583835205496,
    #      '33': 0.9798142964310506, '34': 0.9804069768965702, '35': 0.9809660084442509, '36': 0.9809660084442509,
    #      '37': 0.9809660084442509, '38': 0.9809660084442509, '39': 0.9829113874737766, '40': 0.9833386316141404,
    #      '41': 0.9837445989825787, '42': 0.9841240069254443, '43': 0.984487247330708, '44': 0.9848390242608901,
    #      '45': 0.9851811186211616, '46': 0.9854970822390459, '47': 0.9858035088309867, '48': 0.9860977521161439,
    #      '49': 0.9863100844259117, '50': 0.9863100844259117}.values())
    #
    # fitall['Payment_application'] = list(
    #     {'1': 0.999997717364012, '2': 0.9999689507366609, '3': 0.9999330737843194, '4': 0.9998078584394293,
    #      '5': 0.9402946909443131, '6': 0.9497879263865388, '7': 0.9569511785313873, '8': 0.9622433216246642,
    #      '9': 0.9664371997391661, '10': 0.956350102369389, '11': 0.9602412225425175, '12': 0.9635534727185728,
    #      '13': 0.9635534727185728, '14': 0.9528593537397996, '15': 0.9559974429171783, '16': 0.9586168016984542,
    #      '17': 0.9610504148135557, '18': 0.9631671931169028, '19': 0.9651051121105491, '20': 0.9668388488416868,
    #      '21': 0.9684165949791883, '22': 0.969847206324811, '23': 0.9652866154745812}.values())
    # precall['Payment_application'] = list(
    #     {'1': 0.10949444840182976, '2': 0.1094497281960658, '3': 0.11528754957115032, '4': 0.12188386057497144,
    #      '5': 0.19805744733263225, '6': 0.19344721805733597, '7': 0.18214180293889498, '8': 0.17816476540162518,
    #      '9': 0.17105423651306126, '10': 0.16661747463485757, '11': 0.16610505645898635, '12': 0.1617771355377833,
    #      '13': 0.1617771355377833, '14': 0.1576393462153843, '15': 0.15473418232527095, '16': 0.156785714285755,
    #      '17': 0.15427974082342022, '18': 0.15464879875833834, '19': 0.15282747469473046, '20': 0.15156195943110526,
    #      '21': 0.15003795291129013, '22': 0.1486784134573035, '23': 0.14840993482311268}.values())
    # F1all['Payment_application'] = list(
    #     {'1': 0.19737714576881488, '2': 0.19730392238193312, '3': 0.2066411568450716, '4': 0.21702263977056796,
    #      '5': 0.2908148746964441, '6': 0.29085574367622163, '7': 0.27861417410433315, '8': 0.27645457197969725,
    #      '9': 0.2685086813878963, '10': 0.26376017522017337, '11': 0.2649898066642269, '12': 0.2599857858299987,
    #      '13': 0.2599857858299987, '14': 0.25594771175673053, '15': 0.2525368275324907, '16': 0.2564906877500734,
    #      '17': 0.2534596658645162, '18': 0.25477839839838495, '19': 0.2526291985957428, '20': 0.2513059830734679,
    #      '21': 0.24951034085998483, '22': 0.24791767469567239, '23': 0.24777723223720055}.values())
    #
    # fitall['Geo'] = list(
    #     {'1': 1.0, '2': 0.997829362281979, '3': 0.9881652033942424, '4': 0.8614631457006638, '5': 0.8501770482691604,
    #      '6': 0.8676382297700136, '7': 0.8865264094249486, '8': 0.8892348897462392, '9': 0.9015351904892311,
    #      '10': 0.9113672201757639, '11': 0.9189779565170599, '12': 0.9257244975871429, '13': 0.9312824878526483,
    #      '14': 0.9113830345896945, '15': 0.9149892380473807, '16': 0.9202933786290793, '17': 0.9212667967996406,
    #      '18': 0.9253995810533703, '19': 0.9291857302460993, '20': 0.9324836439228188, '21': 0.9354947755652762,
    #      '22': 0.9382295005197616, '23': 0.9407289747760513}.values())
    # precall['Geo'] = list(
    #     {'1': 0.167448424820266, '2': 0.21461141452477195, '3': 0.20726121151444912, '4': 0.22191753026982317,
    #      '5': 0.22629578343778398, '6': 0.2304491465052004, '7': 0.2358433626399529, '8': 0.2377529140771279,
    #      '9': 0.2367955437837006, '10': 0.23735842837736917, '11': 0.2423763899869843, '12': 0.23856394268593548,
    #      '13': 0.2357517385404925, '14': 0.24521588384240145, '15': 0.24503217618029058, '16': 0.24593298395817095,
    #      '17': 0.24492970843395648, '18': 0.24667091208560554, '19': 0.2494011949042691, '20': 0.250024117745606,
    #      '21': 0.24848756033812874, '22': 0.2499768626863452, '23': 0.24942459259939587}.values())
    # F1all['Geo'] = list(
    #     {'1': 0.2868622223650616, '2': 0.350708908681431, '3': 0.3410299903170004, '4': 0.34141145862717553,
    #      '5': 0.3479828227856281, '6': 0.35628357127503785, '7': 0.36536899217686303, '8': 0.36873725388662726,
    #      '9': 0.3685034010235446, '10': 0.37067589507522725, '11': 0.37808502466226607, '12': 0.3742221610278135,
    #      '13': 0.3712999135036916, '14': 0.3784100255051652, '15': 0.3793625910382152, '16': 0.3812002909799301,
    #      '17': 0.380702736325684, '18': 0.3831581204285938, '19': 0.3871784764190982, '20': 0.3885499173409183,
    #      '21': 0.3873193758520034, '22': 0.38948601210410805, '23': 0.38921190228301367}.values())

    fig = plt.figure()
    x_axis = range(1, 24)
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(range(1, 51), F1all['filteredbpic2017'], linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(range(1, 51), F1all['BPIC2012_A'], linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(range(1, 51), F1all['Control_summary'], linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(range(1, 51), F1all['Geo'], linestyle="-", marker="s", linewidth=1)
    l5 = plt.plot(range(1, 51), F1all['Payment_application'], linestyle="-", marker="s", linewidth=1)
    l6 = plt.plot(x_axis, F1all['Receipt'], linestyle="-", marker="s", linewidth=1)
    plt.xticks(range(1, 51), fontsize=6)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('F1-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'F1allreallogs' + '.svg')

    fig = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(range(1, 51), F1all['filteredbpic2017'], linestyle="-", linewidth=2)
    l2 = plt.plot(range(1, 51), F1all['BPIC2012_A'], linestyle="-", linewidth=2)
    l3 = plt.plot(range(1, 51), F1all['Control_summary'], linestyle="-", linewidth=2)
    l4 = plt.plot(range(1, 51), F1all['Geo'], linestyle="-", linewidth=2)
    l5 = plt.plot(range(1, 51), F1all['Payment_application'], linestyle="-", linewidth=2)
    l6 = plt.plot(x_axis, F1all['Receipt'], linestyle="-", linewidth=2)
    plt.xticks(range(1, 51), fontsize=6)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('F1-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("F1-Score")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'F1allreallogs-line' + '.svg')

    fig = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(range(1, 51), fitall['filteredbpic2017'], linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(range(1, 51), fitall['BPIC2012_A'], linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(range(1, 51), fitall['Control_summary'], linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(range(1, 51), fitall['Geo'], linestyle="-", marker="s", linewidth=1)
    l5 = plt.plot(range(1, 51), fitall['Payment_application'], linestyle="-", marker="s", linewidth=1)
    l6 = plt.plot(x_axis, fitall['Receipt'], linestyle="-", marker="s", linewidth=1)
    plt.xticks(range(1, 51), fontsize=6)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('Fitness-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Fitness")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Fitnessallreallogs' + '.svg')

    fig = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(range(1, 51), fitall['filteredbpic2017'], linestyle="-", linewidth=2)
    l2 = plt.plot(range(1, 51), fitall['BPIC2012_A'], linestyle="-", linewidth=2)
    l3 = plt.plot(range(1, 51), fitall['Control_summary'], linestyle="-", linewidth=2)
    l4 = plt.plot(range(1, 51), fitall['Geo'], linestyle="-", linewidth=2)
    l5 = plt.plot(range(1, 51), fitall['Payment_application'], linestyle="-", linewidth=2)
    l6 = plt.plot(x_axis, fitall['Receipt'], linestyle="-", linewidth=2)
    plt.xticks(range(1, 51), fontsize=6)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('Fitness-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Fitness")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Fitnessallreallogs-line' + '.svg')

    fig = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(range(1, 51), precall['filteredbpic2017'], linestyle="-", marker="s", linewidth=1)
    l2 = plt.plot(range(1, 51), precall['BPIC2012_A'], linestyle="-", marker="s", linewidth=1)
    l3 = plt.plot(range(1, 51), precall['Control_summary'], linestyle="-", marker="s", linewidth=1)
    l4 = plt.plot(range(1, 51), precall['Geo'], linestyle="-", marker="s", linewidth=1)
    l5 = plt.plot(range(1, 51), precall['Payment_application'], linestyle="-", marker="s", linewidth=1)
    l6 = plt.plot(x_axis, precall['Receipt'], linestyle="-", marker="s", linewidth=1)
    plt.xticks(range(1, 51), fontsize=6)
    plt.ylim(0, 1.04)
    plt.legend([l1, l2, l3, l4, l5, l6],
               labels=['BPIC2017', 'BPIC2012', 'BPIC2018-Control', 'BPIC2018-Geo', 'BPIC2018-Payment', 'Receipt'],
               loc='best')
    plt.title('Precision-Real logs')
    plt.xlabel("Num. of Cluster")
    plt.ylabel("Precision")
    plt.grid(axis='y')
    plt.savefig(PIC_PATH + 'Precisionallreallogs' + '.svg')

    fig = plt.figure()
    rc('text', usetex=True)
    rc('font', family='serif')
    l1 = plt.plot(range(1, 51), precall['filteredbpic2017'], linestyle="-", linewidth=2)
    l2 = plt.plot(range(1, 51), precall['BPIC2012_A'], linestyle="-", linewidth=2)
    l3 = plt.plot(range(1, 51), precall['Control_summary'], linestyle="-", linewidth=2)
    l4 = plt.plot(range(1, 51), precall['Geo'], linestyle="-", linewidth=2)
    l5 = plt.plot(range(1, 51), precall['Payment_application'], linestyle="-", linewidth=2)
    l6 = plt.plot(x_axis, precall['Receipt'], linestyle="-", linewidth=2)
    plt.xticks(range(1, 51), fontsize=6)
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