import numpy as np
import regression
from sklearn import preprocessing
path = "./Bank/Bank8FM/"
tr = np.genfromtxt(path + "bank8FM.data", delimiter=' ')
te = np.genfromtxt(path + "bank8FM.test", delimiter=' ')
min_max_scaler = preprocessing.MinMaxScaler()
standard_scaler = preprocessing.StandardScaler()

xtr = tr[:,0:len(tr[0])-1]
btr = standard_scaler.fit_transform(xtr)
atr = min_max_scaler.fit_transform(xtr)
ytr = tr[:,len(tr[0])-1]

xte = te[:,0:len(te[0])-1]
bte = standard_scaler.transform(xte)
ate = min_max_scaler.transform(xte)
yte = te[:,len(te[0])-1]

mses = []
ffnnerr = 1e10
for i in range(0,10):
   ffnnerr = min(ffnnerr,regression.ffnnr(atr, ytr, ate, yte))

print("MIN err " + str(ffnnerr))
mses.append(regression.mlr(xtr, ytr, xte, yte))
mses.append(regression.knn(xtr, ytr, xte, yte))
mses.append(regression.rfr(xtr, ytr, xte, yte))
mses.append(regression.svr(xtr, ytr, xte, yte))
mses.append(ffnnerr)
mses.append(regression.rbfnr(xtr, ytr, xte, yte))
import numpy as np
import regression
from sklearn import preprocessing
path = "./Bank/Bank8FM/"
tr = np.genfromtxt(path + "bank8FM.data", delimiter=' ')
te = np.genfromtxt(path + "bank8FM.test", delimiter=' ')
min_max_scaler = preprocessing.MinMaxScaler()
standard_scaler = preprocessing.StandardScaler()

xtr = tr[:,0:len(tr[0])-1]
btr = standard_scaler.fit_transform(xtr)
atr = min_max_scaler.fit_transform(xtr)
ytr = tr[:,len(tr[0])-1]

xte = te[:,0:len(te[0])-1]
bte = standard_scaler.transform(xte)
ate = min_max_scaler.transform(xte)
yte = te[:,len(te[0])-1]

mses = []
mses.append(regression.mlr(xtr, ytr, xte, yte))
mses.append(regression.knn(xtr, ytr, xte, yte))
mses.append(regression.rfr(xtr, ytr, xte, yte))
mses.append(regression.svr(xtr, ytr, xte, yte))
ffnnerr = 1e10
for i in range(0,10):
   ffnnerr = min(ffnnerr,regression.ffnnr(atr, ytr, ate, yte))
mses.append(ffnnerr)
mses.append(regression.rbfnr(xtr, ytr, xte, yte))


mses = []
mses.append(regression.mlr(atr, ytr, ate, yte))
mses.append(regression.knn(atr, ytr, ate, yte))
mses.append(regression.rfr(atr, ytr, ate, yte))
mses.append(regression.svr(atr, ytr, ate, yte))
mses.append(ffnnerr)
mses.append(regression.rbfnr(atr, ytr, ate, yte))

mses = []
mses.append(regression.mlr(btr, ytr, bte, yte))
mses.append(regression.knn(btr, ytr, bte, yte))
mses.append(regression.rfr(btr, ytr, bte, yte))
mses.append(regression.svr(btr, ytr, bte, yte))
mses.append(ffnnerr)
mses.append(regression.rbfnr(btr, ytr, bte, yte))

import numpy as np
import matplotlib.pyplot as plt


labels = ["MLR", "KNN", "RFR", "SVR", "FFNNR", "RBFNR"]
mses = [0.007729, 0.001189, 0.001145, 0.001550, 0.000937,  0.001936]
ind = np.arange(len(mses))  # the x locations for the groups
width = 0.6      # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind + width/2, mses, width, color='b')

# add some
ax.set_ylabel('MSE')
ax.set_xlabel("Regression algorithm")
ax.set_title('MSE by regression algorithm')
ax.set_xticks(ind+width)
ax.set_xticklabels(labels)

plt.show()

