from __future__ import division, print_function, absolute_import
from matplotlib import rc
# rc('font',**{'family':'sans-serif'})
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd

'''
data= np.array([0,1,2,3,4,5])
print(np.argsort(data)[len(data)//2])
df1 = pd.DataFrame([1,0,0,1,1,1])
df2 = pd.DataFrame([0,0,0,1,1,1])
#print(df1.iloc[:,0])
#print(np.array(df1))
Y = pdist(np.array([df1.iloc[:,0].values,df2.iloc[:,0].values]),'cosine')
#print(Y[0])
'''
'''
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

y= np.array([2.915,1.0000,3.0414,3.0414,2.5495,3.3541,2.5000,2.0616,2.0616,1.0000])
#print(y)
Z = linkage(y, method='average')
print(Z)
print(cophenet(Z,y)) # return vector is the pairwise dist generated from Z
fig = plt.figure(figsize=(10, 8))
dn = fancy_dendrogram(Z, max_d=2.1,show_contracted=True)
plt.show()

y=np.array([1,2,3,4,5,6])
Z = linkage(y, method='average')
print(Z)
print(cophenet(Z,y))
dn = dendrogram(Z)
plt.show()
li=set(list(range(3)))
li.discard(1)
for i in list(li):
    print(i)'''
# 17500 5000
'''
y1 = np.array([0.4414063607149643, 0.0048976483394648366, 0.00043966824490588723, 0.00025432121708286, 0.00020127817828047406, 0.00018169185247520983, 0.00018419072181842473])
y2 = np.array([0.017780763341636936, 0.0001976497516203453, 0.0001225928684826915, 0.00012943275311029845, 0.00014861258697941818, 0.00016034946827371508, 0.00018419072181842473])
y3 = np.array([0.06881426510242147, 0.0004312526202765457, 0.00014358985759517064, 0.0001410058167710789, 0.0001532826728349846, 0.00015605720216941053, 0.00018419072181842473])
percent_value = np.array(range(4, 11)) * 0.1
error1 = np.zeros(len(percent_value))
error2 = np.zeros(len(percent_value))
error3 = np.zeros(len(percent_value))
for i in range(0, len(y1)):
    error1[i] = abs(y1[i] - y1[-1]) / abs(y1[-1])
    error2[i] = abs(y2[i] - y2[-1]) / abs(y2[-1])
    error3[i] = abs(y3[i] - y3[-1]) / abs(y3[-1])
print(error1)
print(error2)
print(error3)
fig, ax = plt.subplots()
#ax.plot(percent_value, y1, label='8000')
#ax.plot(percent_value, y2, label='15000')
ax.plot(percent_value, y3, label='both')
ax.legend()
ax.set(ylabel='distance', xlabel='percentage of variants', title='distance-variant percentage')
plt.yscale('log')
plt.show()


fig2, ax2 = plt.subplots()
#ax2.plot(percent_value, error1, label='8000')
#ax2.plot(percent_value, error2, label='15000')
ax2.plot(percent_value, error3, label='both')
ax2.set(ylabel='relative error', xlabel='percentage of variants', title='error-variant percentage')
ax2.legend()
plt.yscale('log')
plt.show()
'''

# y = np.array([0.00200144 ,0.01357403 ,0.0084349 , 0.00392522 ,0.012375 ,  0.00893385,
#  0.00513085, 0.00127037 ,0.01103749 ,0.00154929 ,0.00766808, 0.00339415,
#  0.00387665 ,0.00554735 ,0.01107205])
# Z = linkage(y, method='average')
# print(Z)
# print(cophenet(Z, y))
# dn = dendrogram(Z,labels=np.array(['25000', '15000', '7000', '10000','12000','8000']))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Loan Amount')
# plt.ylabel('Distance')
# plt.show()

# x= range(1,8)
# fit =[1, 0.980399804,0.980360337,0.980360337,0.980360337,0.980360337,0.980345158]
# prec= [0.843,0.898949651,0.898949651,0.898949651,0.898949651,0.898949651,0.898949651]
# f1 = [0.914812805,0.935093506,0.935071146,0.935071146,0.935071146,0.935071146,0.935062546]

# DMM
# fit=[1,0.999964503
# ,
# 0.980369445
# ,
# 0.978596109
# ,
# 0.971961187
# ,
# 0.939703183
# ,
# 0.933902291
# ]
# prec = [0.843,0.843014666
# ,
# 0.898949651
# ,
# 0.898949651
# ,
# 0.898949651
# ,
# 0.898949651
# ,
# 0.898949651
# ]
# f1 = [0.914812805,0.914798851,
# 0.935074236
# ,
# 0.934121031
# ,
# 0.930572662
# ,
# 0.912087997
# ,
# 0.908728049
# ]

#DFG
# fit=[1,0.999964503
# ,
# 0.980369445
# ,
# 0.980156932
# ,
# 0.980156932
# ,
# 0.980156932
# ,
# 0.979983793
# ]
#
# prec=[0.843,0.843014666
# ,
# 0.898949651
# ,
# 0.898949651
# ,
# 0.898949651
# ,
# 0.898949651
# ,
# 0.898949651
# ]
#
# f1=[0.914812805,0.914798851
# ,
# 0.935074236
# ,
# 0.934953838
# ,
# 0.934953838
# ,
# 0.934953838
# ,
# 0.934861256]


#BPIC 2017
# x= range(0,4)
# fit=[1,0.855,0.615,0.6125]
# prec = [0.83,0.915,1,1]
# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.plot(x, fit, color="r", linestyle="-", marker="o", linewidth=1
# plt.xlabel("Depth")
# plt.ylabel("Fitness")
# plt.figure(1)
# plt.subplot(1, 2, 2)
# plt.plot(x, prec, color="b", linestyle="-", marker="o", linewidth=1)
# plt.xlabel("Depth")
# plt.ylabel("Precision")
# plt.show()
# fig = plt.figure()
#
# ax1 = fig.add_subplot(111)
# ax1.plot(x, fit, color="r", linestyle="-", marker="s", linewidth=1,label='Fitness') # 画图
# # ax1.set_ylim(0,1.02)
# ax1.set_ylabel('Fitness')
# ax1.set_xlabel('Num. of Cluster')
# ax1.set_xticks(x)
# ax1.yaxis.label.set_color('r')
# for tl in ax1.get_yticklabels():
#     tl.set_color('r')
# ax2 = ax1.twinx()
#
# ax2.plot(x, prec, color="b", linestyle="-", marker="s", linewidth=1,label='Precision') # 画图
# # ax2.set_ylim(0,1.02)
# ax2.set_ylim(np.min(prec)-0.01,1)
# ax2.set_ylabel('Precision')
# ax2.yaxis.label.set_color('b')
# for tl in ax2.get_yticklabels():
#     tl.set_color('b')
# fig.show()
# fig2 = plt.figure()
# plt.plot(x, f1, color="b", linestyle="-", marker="s", linewidth=1)
# plt.ylim(np.min(f1)-0.01,1)
# # plt.ylim(0,1)
# plt.xlabel("Num. of Cluster")
# plt.ylabel("F1-Score")
# plt.show()



# # fig for f1 scores
# x= range(1,8)
# f1dfg=[0.914812805,0.914798851
# ,
# 0.935074236
# ,
# 0.934953838
# ,
# 0.934953838
# ,
# 0.934953838
# ,
# 0.934861256]
# f1dfg_update=[0.914812805,0.914810069
# ,
# 0.935081466
# ,
# 0.935069076
# ,
# 0.934953838
# ,
# 0.934953838
# ,
# 0.934892025
# ]
# f1DMM = [0.914812805,0.914798851,
# 0.935074236
# ,
# 0.934121031
# ,
# 0.930572662
# ,
# 0.912087997
# ,
# 0.908728049
# ]
# f1avg = [0.914812805,0.935093506,0.935071146,0.935071146,0.935071146,0.935071146,0.935062546]
# fig = plt.figure()
# l1=plt.plot(x, f1DMM, color="b", linestyle="--", marker="s", linewidth=1)
# l2=plt.plot(x, f1dfg, color="r", linestyle="-", marker="s", linewidth=1)
#
# l3=plt.plot(x, f1avg, color="g", linestyle="-", marker="s", linewidth=1)
# plt.ylim(np.min(f1DMM)-0.01,1)
# plt.legend([l1, l2,l3], labels=['DMM', 'DFG','AVG'],  loc='best')
# # plt.ylim(0,1)
# plt.xlabel("Num. of Cluster")
# plt.ylabel("F1-Score")
# plt.show()

# #compare update or not
# l1=plt.plot(x, f1dfg, color="b", linestyle="-", marker="s", linewidth=1)
# l2=plt.plot(x, f1dfg_update, color="r", linestyle="-", marker="o", linewidth=1)
# # plt.ylim(np.min(f1dfg)-0.01,1)
# plt.legend([l1, l2], labels=['DFG', 'DFG_update'],  loc='best')
# # plt.ylim(0,1)
# plt.xlabel("Num. of Cluster")
# plt.ylabel("F1-Score")
# plt.show()


# df = pd.DataFrame(np.random.rand(10, 1), columns=['A', 'B', 'C', 'D', 'E'])
#
# plt.figure(figsize=(10, 4))
#
# f = df.boxplot(sym='o',vert = True,showfliers = True)
#
# plt.plot(range(1,6), np.random.rand(5, 1), color="g", linestyle="-", marker="s", linewidth=1)
# plt.show()



# s1 = pd.Series(np.array([0.62,1,0.945]))
# s2 = pd.Series(np.array([0.62,1,0.945,0.815]))
# s3 = pd.Series(np.array([0.62,1,0.945,0.815,0.815]))
# s4 = pd.Series(np.array([0.62,1,0.945,0.815,0.815,0.815]))
# s5 = pd.Series(np.array([0.62,1,0.945,0.815,0.815,0.815,0.923]))

#boxplot DMM
# s0 = pd.Series(np.array([0.914812805]))
# s1 = pd.Series(np.array([0.765432099,0.914812805,0.971722365]))
# s2 = pd.Series(np.array([0.765432099,0.914812805,0.971722365,0.954521694]))
# s3 = pd.Series(np.array([0.765432099,0.914812805,0.971722365,0.954521694,0.911268372]))
# s4 = pd.Series(np.array([0.765432099,0.914812805,0.971722365,0.954521694,0.911268372,0.959958398]))
# s5 = pd.Series(np.array([0.765432099,0.914812805,0.971722365,0.954521694,0.911268372,0.959958398,0.954521694]))

#DFG
# s0 = pd.Series(np.array([0.914812805]))
# s1 = pd.Series(np.array([0.765432099,0.914812805,0.971722365]))
# s2 = pd.Series(np.array([0.765432099,0.914812805,0.971722365,0.898071625]))
# s3 = pd.Series(np.array([0.765432099,0.914812805,0.971722365,0.898071625,0.898071625]))
# s4 = pd.Series(np.array([0.765432099,0.914812805,0.971722365,0.898071625,0.898071625,0.898071625]))
# s5 = pd.Series(np.array([0.765432099,0.914812805,0.971722365,0.898071625,0.898071625,0.898071625,0.959958398]))

#AVG
# s0 = pd.Series(np.array([0.914812805]))
# s1 = pd.Series(np.array([0.898071625,0.914812805,0.971722365]))
# s2 = pd.Series(np.array([0.898071625,0.898071625,0.914812805,0.971722365]))
# s3 = pd.Series(np.array([0.898071625,0.898071625,0.914812805,0.971722365,0.898071625]))
# s4 = pd.Series(np.array([0.898071625,0.898071625,0.914812805,0.971722365,0.898071625,0.898071625]))
# s5 = pd.Series(np.array([0.898071625,0.898071625,0.914812805,0.971722365,0.898071625,0.898071625,0.898071625]))
# data = pd.DataFrame({"1":s0, "2":s0,"3": s1, "4": s2, "5": s3, "6": s4,"7":s5})
# print(data)
# plt.plot(range(1,8), f1DMM, color="b", linestyle="-", marker="s", linewidth=1)
# data.boxplot(sym='o')
#
# plt.ylim(0.74,1)
# plt.xlabel("Num. of Cluster")
# plt.ylabel("F1-Score")
# plt.grid(axis='x')
# plt.show()



#Receipt
# dfg= {'1': 0.2449566096779114, '2': 0.24640818212144894, '3': 0.24645205009656512, '4': 0.24719404833834047, '5': 0.24826074444652005, '6': 0.2617804581738966, '7': 0.27639472285599204, '8': 0.3447425563887006, '9': 0.3464888826844202, '10': 0.34719989283229297, '11': 0.3070841921534124, '12': 0.3066532744259339, '13': 0.3126335605058499, '14': 0.4301175913668744, '15': 0.4366424958973945, '16': 0.4423642683882837, '17': 0.42636556323608443, '18': 0.6442167152515513, '19': 0.6682346218043699, '20': 0.6776970885418656, '21': 0.6638192565766592, '22': 0.617234025519495, '23': 0.617280114110985}
# f1dfg = list(dfg.values())
# dfgup ={'1': 0.2449566096779114, '2': 0.24640818212144894, '3': 0.24645205009656512, '4': 0.24751874620474468, '5': 0.24826074444652005, '6': 0.2617804581738966, '7': 0.27639472285599204, '8': 0.3447425563887006, '9': 0.30432644424743194, '10': 0.3038955265199534, '11': 0.3124008538270108, '12': 0.3174471485417448, '13': 0.3192898582238419, '14': 0.5441174386220617, '15': 0.5697821100720859, '16': 0.6004824849611563, '17': 0.6088445043616922, '18': 0.699916370449059, '19': 0.6999163704490591, '20': 0.6545515749623987, '21': 0.6250981302997787, '22': 0.6251442188912688, '23': 0.617280114110985}
# f1dfg_update = list(dfgup.values())
#
# avg ={'1': 0.2449566096779114, '2': 0.24640818212144894, '3': 0.24645205009656512, '4': 0.24751874620474468, '5': 0.24826074444652005, '6': 0.2617804581738966, '7': 0.26243566225903336, '8': 0.31070647848218863, '9': 0.31310057209381903, '10': 0.3460579649569415, '11': 0.30573823620205054, '12': 0.3474779536222246, '13': 0.34839299184610795, '14': 0.3556723004173724, '15': 0.3605924760757178, '16': 0.36648758750587235, '17': 0.3740585596880312, '18': 0.3874190988330173, '19': 0.39231796318617895, '20': 0.6008454164566367, '21': 0.608872006118959, '22': 0.617234025519495, '23': 0.617280114110985}
# f1avg = list(avg.values())
# x= range(1,24)
# # plt.rc('text', usetex=True)
# # plt.rc('font', family='serif')
# l1=plt.plot(x, f1dfg, color="b", linestyle="-", linewidth=2)
# l2=plt.plot(x, f1dfg_update, color="r", linestyle="-" ,linewidth=2)
# l3=plt.plot(x, f1avg, color="g", linestyle="-", linewidth=2)
# plt.xticks(x)
# plt.ylim(0,1.04)
# plt.legend([l1, l2,l3], labels=['DFG-update', 'DMM','AVG'],  loc='best')
# # plt.ylim(0,1)
# plt.xlabel("Num. of Cluster")
# plt.ylabel("F1-Score")
# plt.savefig('font.svg')

rtdfg = [5.670706033706665,12.101333856582642]
rtdmm = [74.23017835617065,238.69585061073303]
rtavg=[74.34487009048462,74.5996880531311]
xindex = ['No', 'Yes']
x= range(1,3)

rc('text', usetex=True)
rc('font', family='serif')
l1=plt.plot(x, rtdfg, color="b", linestyle="dashed",marker="s", linewidth=1)
l2=plt.plot(x, rtdmm, color="r", linestyle="dashed" ,marker="s",linewidth=1)
l3=plt.plot(x, rtavg, color="g", linestyle="dashed", marker="s",linewidth=1)
plt.xticks(x,xindex)
plt.legend([l1, l2,l3], labels=['DFG', 'DMM','AVG'],  loc='best')
# plt.ylim(4,250)
plt.yscale('log')
plt.xlabel("Recomputering Or Not")
plt.ylabel("Runtime (sec)")
plt.savefig('runtime1.svg')

fig2 = plt.figure()
rc('text', usetex=True)
rc('font', family='serif')
l1=plt.plot(x, rtdfg, color="b", linestyle="dashed",marker="s", linewidth=1)
l2=plt.plot(x, rtdmm, color="r", linestyle="dashed" ,marker="s",linewidth=1)
l3=plt.plot(x, rtavg, color="g", linestyle="dashed", marker="s",linewidth=1)
plt.xticks(x,xindex)
plt.legend([l1, l2,l3], labels=['DFG', 'DMM','AVG'],  loc='best')
# plt.ylim(4,250)
plt.xlabel("Recomputering Or Not")
plt.ylabel("Runtime (sec)")
plt.savefig('runtime2.svg')
