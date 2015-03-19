__author__ = 'mac'

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def eval(clf, xtr, ytr, xte, yte):
    clf.fit(xtr, ytr)
    pred = clf.predict(xte)
    return np.mean(pred == yte)

def knn(xtr, ytr, xte, yte):
    best_acc = -1
    best_k = -1
    for k in [1,3,5,11,21,31]:
        acc = eval(KNeighborsClassifier(k), xtr, ytr,  xte, yte)
        if best_acc < acc:
            best_acc = acc
            best_k = k
    if __debug__:
        print str(best_acc) + " with _k_ " + str(best_k)
    return best_acc

def svm(xtr, ytr, xte, yte, bestHypers=True,  g = 1, c = 1):
    bestacc = -1
    bestg = -1
    bestc = -1
    bestClf = None
    if bestHypers:
        gamma = [10**(v-3) for v in range(0,9)]
        C = gamma
        for g in gamma:
            for c in C:
                clf = OneVsRestClassifier(SVC(C=c, gamma=g, kernel='rbf', probability=True))
                acc = eval(clf, xtr, ytr, xte, yte)
                if __debug__:
                    print("Acc with gamma {} and c {} -> {}".format(g, c, acc))
                if bestacc < acc:
                    bestacc = acc
                    bestg = g
                    bestc = c
                    bestClf = clf
        print("BEST Acc with gamma {} and c {} -> {}".format(bestg, bestc, bestacc))
    else:
        bestClf = OneVsRestClassifier(SVC(C=c, gamma=g, kernel='rbf', probability=True))
        bestClf.fit(xtr, ytr)
    return bestClf.predict(xte)

def rfc(xtr, ytr, xte, yte):
    best_acc = -1
    best_k = -1
    for k in [1,3,5,11,21,31]:
        rfc = RandomForestClassifier(max_features=k)
        acc = eval(rfc, xtr, ytr,  xte, yte)
        if best_acc < acc:
            best_acc = acc
            best_k = k
    if __debug__:
        print str(best_acc) + " with _mfeatures_ " + str(best_k)
    return best_acc