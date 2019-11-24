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
    plot_clu=23
    runtime = dict()

    # LOG_PATH = "/home/yukun/dataset/Receipt4.xes"
    # LOG_PATH = "D:/Sisc/19SS/thesis/Dataset/Receipt4.xes"
    # ATTR_NAME = 'responsible'
    # METHOD = 'DMM'
    #
    # log = xes_importer.apply(LOG_PATH)
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME
    #
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_leven_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    # fitall['Receipt'] = list(plot_fit.values())
    # precall['Receipt'] = list(plot_prec.values())
    # F1all['Receipt'] = list(plot_F1.values())
    #
    # LOG_PATH = "/home/yukun/dataset/filteredbpic2017.xes"
    # ATTR_NAME = 'CreditScore'
    # METHOD = 'dfg'
    #
    # log = xes_importer.apply(LOG_PATH)
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME
    #
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    # fitall['filteredbpic2017'] = list(plot_fit.values())
    # precall['filteredbpic2017'] = list(plot_prec.values())
    # F1all['filteredbpic2017'] = list(plot_F1.values())
    #
    # LOG_PATH = "/home/yukun/dataset/BPIC2012_A.xes"
    # ATTR_NAME = 'AMOUNT_REQ'
    # METHOD = 'dfg'
    #
    # log = xes_importer.apply(LOG_PATH)
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME
    #
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    # fitall['BPIC2012_A'] = list(plot_fit.values())
    # precall['BPIC2012_A'] = list(plot_prec.values())
    # F1all['BPIC2012_A'] = list(plot_F1.values())
    #
    # LOG_PATH = "/home/yukun/dataset/document_logs/Control_summary.xes"
    # ATTR_NAME = 'amount_applied0'
    # METHOD = 'dfg'
    #
    # log = xes_importer.apply(LOG_PATH)
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME
    #
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    # fitall['Control_summary'] = list(plot_fit.values())
    # precall['Control_summary'] = list(plot_prec.values())
    # F1all['Control_summary'] = list(plot_F1.values())
    #
    #
    # LOG_PATH = "/home/yukun/dataset/document_logs/Payment_application.xes"
    # ATTR_NAME = 'amount_applied0'
    # METHOD = 'dfg'
    # log = xes_importer.apply(LOG_PATH)
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME
    #
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    # fitall['Payment_application'] = list(plot_fit.values())
    # precall['Payment_application'] = list(plot_prec.values())
    # F1all['Payment_application'] = list(plot_F1.values())
    #
    # LOG_PATH = "/home/yukun/dataset/document_logs/Geo_parcel_document.xes"
    # ATTR_NAME = 'amount_applied0'
    # METHOD = 'dfg'
    # log = xes_importer.apply(LOG_PATH)
    # print(LOG_PATH)
    # print(ATTR_NAME)
    # print(METHOD)
    # TYPE = METHOD + ATTR_NAME
    #
    # (plot_fit, plot_prec, plot_F1, plot_boxfit, plot_boxprec, plot_box, plot_length,
    #  runtime) = merge_log.main_calc_recompute(log, ATTR_NAME, METHOD, TYPE, percent, alpha,runtime,plot_clu)
    # fitall['Geo'] = list(plot_fit.values())
    # precall['Geo'] = list(plot_prec.values())
    # F1all['Geo'] = list(plot_F1.values())

    fitall['Receipt'] = list(
        {'1': 0.9871623451004897, '2': 0.7252854590998924, '3': 0.622014337783979, '4': 0.7084464941912509,
         '5': 0.6906560391814989, '6': 0.6969888284425486, '7': 0.7204002817447476, '8': 0.7531007010167994,
         '9': 0.7286199467492621, '10': 0.7536801188727913, '11': 0.7738693163406307, '12': 0.7921887839661829,
         '13': 0.8035380323701417, '14': 0.8076835640902577, '15': 0.8192885707113948, '16': 0.8277898308555318,
         '17': 0.8056427698986632, '18': 0.8152860420145461, '19': 0.823761663413301, '20': 0.8291697032472223,
         '21': 0.8363150349013811, '22': 0.8380166532007617, '23': 0.8421102029193948}.values())
    precall['Receipt'] = list(
        {'1': 0.13982676330210364, '2': 0.48924671498438516, '3': 0.7035819178559121, '4': 0.5957366971770063,
         '5': 0.674726003704338, '6': 0.6459338196372995, '7': 0.6973211044713399, '8': 0.6444129530805764,
         '9': 0.6129890875263273, '10': 0.5858799483949363, '11': 0.5528996170091997, '12': 0.5223845121749725,
         '13': 0.5250330255360288, '14': 0.5033096777080136, '15': 0.5327862934597235, '16': 0.5330304862635773,
         '17': 0.5660093757949332, '18': 0.5535587269292378, '19': 0.5452238487399109, '20': 0.5689419548070729,
         '21': 0.5581303234454227, '22': 0.5448693092568266, '23': 0.5645114898245286}.values())
    F1all['Receipt'] = list(
        {'1': 0.2449566096779114, '2': 0.42095976845683253, '3': 0.4867194773602561, '4': 0.4712624038473022,
         '5': 0.5294707675068432, '6': 0.5414056490672706, '7': 0.5980646943382399, '8': 0.5768717207909952,
         '9': 0.5679462682101176, '10': 0.5618349696996708, '11': 0.5437736353333467, '12': 0.5242328280156981,
         '13': 0.5376983725140169, '14': 0.5246877858440896, '15': 0.5539263954608077, '16': 0.562260682886589,
         '17': 0.58465881804464, '18': 0.5803351772146961, '19': 0.5784845899730752, '20': 0.5986430883437075,
         '21': 0.5942711928561809, '22': 0.5842300138225129, '23': 0.6007124650633123}.values())


    fitall['filteredbpic2017'] = list(
        {'1': 1.0, '2': 0.9073165035846898, '3': 0.9199242842945701, '4': 0.8451716966493595, '5': 0.8390639587533636,
         '6': 0.8349921334893663, '7': 0.8475062302541811, '8': 0.8433970773685809, '9': 0.8402010695686697,
         '10': 0.8484400436961097, '11': 0.8453666767391343, '12': 0.8428055376083216, '13': 0.8406384198822491,
         '14': 0.8387808904027585, '15': 0.8371710315205332, '16': 0.835762404998586, '17': 0.8345194992439268,
         '18': 0.8328655231755736, '19': 0.8319059170699845, '20': 0.8310422715749542, '21': 0.8302608780318318,
         '22': 0.8295505202653566, '23': 0.8289019327394446}.values())
    precall['filteredbpic2017'] = list(
        {'1': 0.7625744049007841, '2': 0.881287202450392, '3': 0.9208581349669279, '4': 0.863535914436304,
         '5': 0.8908287315490432, '6': 0.9090239429575359, '7': 0.9220205225350308, '8': 0.9317679572181519,
         '9': 0.9393492953050239, '10': 0.9454143657745215, '11': 0.9503766961586559, '12': 0.9545119714787679,
         '13': 0.9580110505957857, '14': 0.9610102612675153, '15': 0.9636095771830143, '16': 0.9658839786090759,
         '17': 0.9678908033967775, '18': 0.9690247425813754, '19': 0.9706550192876188, '20': 0.9721222683232378,
         '21': 0.9734497793554646, '22': 0.9746566075665798, '23': 0.9757584941941199}.values())
    F1all['filteredbpic2017'] = list(
        {'1': 0.8652961290944307, '2': 0.8815724497815136, '3': 0.9116470641871414, '4': 0.8473202452150559,
         '5': 0.8574259502657642, '6': 0.8641630869662361, '7': 0.8778165094045388, '8': 0.8805040711332712,
         '9': 0.8824312599483073, '10': 0.8901618383569121, '11': 0.8908606503670652, '12': 0.8914429937088596,
         '13': 0.891935745767301, '14': 0.8923581046745364, '15': 0.8927241490608071, '16': 0.8930444378987936,
         '17': 0.8933270456970174, '18': 0.8929817855285435, '19': 0.8932379426306516, '20': 0.8934684840225489,
         '21': 0.8936770690914082, '22': 0.8938666918812803, '23': 0.8940398257329027}.values())


    fitall['BPIC2012_A'] = list(
        {'1': 1.0, '2': 0.8025232347269484, '3': 0.7495518857863169, '4': 0.7622868863283971, '5': 0.7805697135616139,
         '6': 0.7738709620147306, '7': 0.8061751102983405, '8': 0.7911716359742073, '9': 0.7911716359742073,
         '10': 0.811883568194793, '11': 0.811883568194793, '12': 0.811883568194793, '13': 0.811883568194793,
         '14': 0.7723286359254169, '15': 0.7875067268637224, '16': 0.7859184618922337, '17': 0.7842330315941308,
         '18': 0.7776242184474909, '19': 0.7717727205536522, '20': 0.7609479802924632, '21': 0.7512242117392811,
         '22': 0.7511284382032244, '23': 0.7502308471411563}.values())
    precall['BPIC2012_A'] = list(
        {'1': 0.5213162855289565, '2': 0.713633311308509, '3': 0.7777389865683598, '4': 0.8009177943603003,
         '5': 0.8219243029058525, '6': 0.8359286419362206, '7': 0.7909840195923258, '8': 0.8171110171432852,
         '9': 0.8171110171432852, '10': 0.8065818413318795, '11': 0.8065818413318795, '12': 0.8065818413318795,
         '13': 0.8065818413318795, '14': 0.8176166637322038, '15': 0.7978633051853206, '16': 0.7860870734548306,
         '17': 0.79313785366855, '18': 0.8046301951314084, '19': 0.8149128164402817, '20': 0.8241671756182678,
         '21': 0.832540167255493, '22': 0.8441762250754702, '23': 0.8226475837800717}.values())
    F1all['BPIC2012_A'] = list(
        {'1': 0.6853489842813278, '2': 0.705442997396587, '3': 0.7211528393092134, '4': 0.7493667283110089,
         '5': 0.7753033747565122, '6': 0.781892450053617, '7': 0.7681005263718612, '8': 0.7737392667751104,
         '9': 0.7737392667751104, '10': 0.7835920100716472, '11': 0.7835920100716472, '12': 0.7835920100716472,
         '13': 0.7835920100716472, '14': 0.7603189941057727, '15': 0.7553209934508099, '16': 0.7504430621526215,
         '17': 0.7548265518953551, '18': 0.7580278895676066, '19': 0.7602285756435168, '20': 0.7579199566200778,
         '21': 0.7558890655883755, '22': 0.7650676678352665, '23': 0.7525218991627797}.values())


    fitall['Control_summary'] = list(
        {'1': 0.9665845411315319, '2': 0.9665845411315319, '3': 0.930854577526241, '4': 0.9397690556139429,
         '5': 0.9451307323782026, '6': 0.9542684361103594, '7': 0.9607916405099106, '8': 0.9656840438095742,
         '9': 0.9694872001183023, '10': 0.9725315667971938, '11': 0.9750229335997935, '12': 0.9743137299562314,
         '13': 0.9760998654674882, '14': 0.9751894086387624, '15': 0.9768405783809212, '16': 0.9782837214138147,
         '17': 0.9795570829134268, '18': 0.9806889598019706, '19': 0.9815755141072651, '20': 0.9824939823645173,
         '21': 0.9833243101998842, '22': 0.9840791536865813, '23': 0.9847683586092175}.values())
    precall['Control_summary'] = list(
        {'1': 0.8668475960724564, '2': 0.8677129042950131, '3': 0.9115010245404785, '4': 0.9236182403750918,
         '5': 0.9128273883824475, '6': 0.9271713410706122, '7': 0.937589887171329, '8': 0.945278055493742,
         '9': 0.9498126799641469, '10': 0.9548350809309539, '11': 0.9546771720456523, '12': 0.9551461819995182,
         '13': 0.9585958209605125, '14': 0.9609161537754504, '15': 0.9634337344122221, '16': 0.9657191260114584,
         '17': 0.9677356480107842, '18': 0.9695281120101852, '19': 0.9711314476145144, '20': 0.972013181387996,
         '21': 0.9733458870361866, '22': 0.9745574376254509, '23': 0.9756636359895616}.values())
    F1all['Control_summary'] = list(
        {'1': 0.9140032716547106, '2': 0.9144840509326577, '3': 0.9176121808755857, '4': 0.9290122894037355,
         '5': 0.9263226883182414, '6': 0.9385081318405892, '7': 0.9472957694405698, '8': 0.9538166538581005,
         '9': 0.9581327032479707, '10': 0.96231801558621, '11': 0.9635606510748864, '12': 0.9635471711946781,
         '13': 0.9662559255442285, '14': 0.9669779789223557, '15': 0.9691339803962216, '16': 0.9710609461376278,
         '17': 0.9727612100271038, '18': 0.9742725557066384, '19': 0.975561420024122, '20': 0.9764835721949211,
         '21': 0.9776017560074829, '22': 0.9786182867461757, '23': 0.9795464235075907}.values())


    fitall['Payment_application'] = list(
        {'1': 0.999997717364012, '2': 0.9999689507366609, '3': 0.9999330737843194, '4': 0.9998078584394293,
         '5': 0.9402946909443131, '6': 0.9497879263865388, '7': 0.9569511785313873, '8': 0.9622433216246642,
         '9': 0.9664371997391661, '10': 0.956350102369389, '11': 0.9602412225425175, '12': 0.9635534727185728,
         '13': 0.9635534727185728, '14': 0.9528593537397996, '15': 0.9559974429171783, '16': 0.9586168016984542,
         '17': 0.9610504148135557, '18': 0.9631671931169028, '19': 0.9651051121105491, '20': 0.9668388488416868,
         '21': 0.9684165949791883, '22': 0.969847206324811, '23': 0.9652866154745812}.values())
    precall['Payment_application'] = list(
        {'1': 0.10949444840182976, '2': 0.1094497281960658, '3': 0.11528754957115032, '4': 0.12188386057497144,
         '5': 0.19805744733263225, '6': 0.19344721805733597, '7': 0.18214180293889498, '8': 0.17816476540162518,
         '9': 0.17105423651306126, '10': 0.16661747463485757, '11': 0.16610505645898635, '12': 0.1617771355377833,
         '13': 0.1617771355377833, '14': 0.1576393462153843, '15': 0.15473418232527095, '16': 0.156785714285755,
         '17': 0.15427974082342022, '18': 0.15464879875833834, '19': 0.15282747469473046, '20': 0.15156195943110526,
         '21': 0.15003795291129013, '22': 0.1486784134573035, '23': 0.14840993482311268}.values())
    F1all['Payment_application'] = list(
        {'1': 0.19737714576881488, '2': 0.19730392238193312, '3': 0.2066411568450716, '4': 0.21702263977056796,
         '5': 0.2908148746964441, '6': 0.29085574367622163, '7': 0.27861417410433315, '8': 0.27645457197969725,
         '9': 0.2685086813878963, '10': 0.26376017522017337, '11': 0.2649898066642269, '12': 0.2599857858299987,
         '13': 0.2599857858299987, '14': 0.25594771175673053, '15': 0.2525368275324907, '16': 0.2564906877500734,
         '17': 0.2534596658645162, '18': 0.25477839839838495, '19': 0.2526291985957428, '20': 0.2513059830734679,
         '21': 0.24951034085998483, '22': 0.24791767469567239, '23': 0.24777723223720055}.values())


    fitall['Geo'] = list({'1': 1.0, '2': 0.997829362281979, '3': 0.9881652033942424, '4': 0.8614631457006638, '5': 0.8501770482691604,
     '6': 0.8676382297700136, '7': 0.8865264094249486, '8': 0.8892348897462392, '9': 0.9015351904892311,
     '10': 0.9113672201757639, '11': 0.9189779565170599, '12': 0.9257244975871429, '13': 0.9312824878526483,
     '14': 0.9113830345896945, '15': 0.9149892380473807, '16': 0.9202933786290793, '17': 0.9212667967996406,
     '18': 0.9253995810533703, '19': 0.9291857302460993, '20': 0.9324836439228188, '21': 0.9354947755652762,
     '22': 0.9382295005197616, '23': 0.9407289747760513}.values())
    precall['Geo'] = list({'1': 0.167448424820266, '2': 0.21461141452477195, '3': 0.20726121151444912, '4': 0.22191753026982317,
     '5': 0.22629578343778398, '6': 0.2304491465052004, '7': 0.2358433626399529, '8': 0.2377529140771279,
     '9': 0.2367955437837006, '10': 0.23735842837736917, '11': 0.2423763899869843, '12': 0.23856394268593548,
     '13': 0.2357517385404925, '14': 0.24521588384240145, '15': 0.24503217618029058, '16': 0.24593298395817095,
     '17': 0.24492970843395648, '18': 0.24667091208560554, '19': 0.2494011949042691, '20': 0.250024117745606,
     '21': 0.24848756033812874, '22': 0.2499768626863452, '23': 0.24942459259939587}.values())
    F1all['Geo'] =list({'1': 0.2868622223650616, '2': 0.350708908681431, '3': 0.3410299903170004, '4': 0.34141145862717553,
     '5': 0.3479828227856281, '6': 0.35628357127503785, '7': 0.36536899217686303, '8': 0.36873725388662726,
     '9': 0.3685034010235446, '10': 0.37067589507522725, '11': 0.37808502466226607, '12': 0.3742221610278135,
     '13': 0.3712999135036916, '14': 0.3784100255051652, '15': 0.3793625910382152, '16': 0.3812002909799301,
     '17': 0.380702736325684, '18': 0.3831581204285938, '19': 0.3871784764190982, '20': 0.3885499173409183,
     '21': 0.3873193758520034, '22': 0.38948601210410805, '23': 0.38921190228301367}.values())

    fig = plt.figure()
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

    fig = plt.figure()
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

    fig = plt.figure()
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

    fig = plt.figure()
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

    fig = plt.figure()
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

    fig = plt.figure()
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


