from __future__ import division, print_function, absolute_import
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

y = np.array([0.00031656 ,0.0005384  ,0.00119314, 0.00079948 ,0.00076642, 0.00021545,
 0.00053412, 0.00035174 ,0.00030837 ,0.00101519, 0.00063727, 0.00053139,
 0.00090698, 0.00103637 ,0.00062803])
Z = linkage(y, method='average')
print(Z)
print(cophenet(Z, y))
dn = dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Attribute Value')
plt.ylabel('Distance')
plt.show()
