from scipy.cluster.hierarchy import dendrogram, linkage,cophenet
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd

df1 = pd.DataFrame([1,0,0,1,1,1])
df2 = pd.DataFrame([0,0,0,1,1,1])
print(df1.iloc[:,0])
print(np.array(df1))
Y = pdist(np.array([df1.iloc[:,0].values,df2.iloc[:,0].values]),'cosine')
print(Y[0])
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
'''