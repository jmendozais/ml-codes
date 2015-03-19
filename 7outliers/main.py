import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn import svm
from operator import itemgetter

path = "/Users/mac/research/courses/mo444/exercise7/dados7b.csv"
data = np.genfromtxt(path, delimiter=',')
noutliers = 7


# Finding the limits that separed normal data of outliers
# in heach dimension
res = pl.boxplot(data, whis=2.2)
fl = res['fliers']
limits = len(fl)*[1e100]
for i in range(0,len(fl)):
    if i%2 == 1:
        limits[i] *= -1
    outs = fl[i].get_data()[1]
    for j in range(0,len(outs)):
        if (i%2 == 0) and (limits[i] > outs[j]):
                limits[i] = outs[j]
        elif (i%2 == 1) and (limits[i] < outs[j]):
                limits[i] = outs[j]
# Selecting points
boxcand = set()
for j in range(0, len(data[0])):
    res[j] = []
    for i in range(0, len(data)):
        if (limits[2*j] <= data[i][j]):
            boxcand.add(i)
        if (limits[2*j+1] >= data[i][j]):
            boxcand.add(i)
# Showing the points
boxcand
# selected using boxplot: [33, 98, 233, 428, 463, 561, 312]
kmcand = set()
T = 2
for k in range(2, 40):
    kmeans = KMeans(n_clusters=k)
    _ = kmeans.fit(data)
    labels = kmeans.labels_
    memb = [[] for i in range(0,k)]
    for i in range(0, len(labels)):
        memb[labels[i]].append(i)
    for i in range(0, k):
        if(len(memb[i]) <= T):
            for index in memb[i]:
                kmcand.add(index)
kmcand
kmcand & boxcand
# knn

for kv in range(2, 600):
    kdt = KDTree(data, leaf_size=30, metric='euclidean')
    indeg = len(data)*[0]
    for entry in data:
        knn = kdt.query(entry, k=kv, return_distance=False)[0]
        for nn in knn:
            indeg[nn] += 1
    sdeg = []
    for i in range(0, len(indeg)):
        sdeg.append([i, indeg[i]])
    sdeg = sorted(sdeg, key=itemgetter(1))
    knncand = []
    for i in range(0, noutliers):
        knncand.append(sdeg[i][0])
    print(str(kv) + ":" + str(sorted(knncand)))

knncand = set([33, 98, 233, 312, 428, 463, 561])
# knn =   [ 58, 160, 249, 355, 417, 538, 554]
#           or 561 instead of 417 for more global solutions
# km cands [561, 162, 440] clus 60 t 2
# kn cands [561, 599,
# km & box =    [160, 355, 554, 85, 249, 58, 538])

for T in [x*2.0/100 for x in range(1,200)]:
    kdt = KDTree(data, leaf_size=30, metric='euclidean')
    indeg = len(data)*[0]
    for entry in data:
        (dist, knn) = kdt.query(entry, k=200, return_distance=True)
        for i in range(0, len(dist[0])):
            if dist[0][i] < T:
                indeg[knn[0][i]] += 1
    sdeg = []
    for i in range(0, len(indeg)):
        sdeg.append([i, indeg[i]])
    sdeg = sorted(sdeg, key=itemgetter(1))
    knncand = []
    for i in range(0, noutliers):
        knncand.append(sdeg[i][0])
    print(str(T) + ":" + str(sorted(knncand)))

bb = set([33, 98, 233, 312, 428, 463, 561])
for g in [x*1.0/1000 for x in range(1,100)]:
    clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=g)
    _ = clf.fit(data)
    ys = clf.predict(data)
    svmcand = set()
    for i in range(0, len(ys)):
        if ys[i] == -1:
            svmcand.add(i)
    svmcand = sorted([it for it in svmcand])
    #print(svmcand)
    print(str(sorted([y for y in (bb & set(svmcand))])))