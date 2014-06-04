
from sklearn import neighbors, decomposition, preprocessing, cross_validation as cv, svm, ensemble
from sklearn.svm import NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
import neurolab as nl
import numpy as np
import pyradbas as pyrb

class Regressor:
    regressor = None
    def train(self, X, y, hypers):
        return None
    def test(self, X, real):
        if self.regressor == None:
            return 1e10
        predicted = self.regressor.predict(X)
        return mse(real, predicted)

class MLR(Regressor):
    def train(self, X, y, hypers):
        self.regressor = LogisticRegression(C=hypers).fit(X, y)

class RFR(Regressor):
    def train(self, X, y, hypers):
        self.regressor = RandomForestRegressor(max_features=hypers).fit(X, y)

class FFNNR (Regressor):
    def train(self, X, y, hypers):
        size = len(y)
        ny = y.reshape(size,1)
        self.regressor = nl.net.newff([[0, 1],[0,1],[0, 1],[0,1],[0, 1],[0,1],[0, 1],[0,1]],[hypers, 1])
        #Trained with Broyden-Fletcher-Coldfarb-Shanno algorithm.
        self.regressor.train(X, ny, epochs=40, show=10, goal=0.02)
    def test(self, X, real):
        if self.regressor == None:
            return 1e10
        predicted = np.array(self.regressor.sim(X))
        predicted = predicted.reshape(1,len(real))
        return mse(real, predicted[0])

class KNNR(Regressor):
    def train(self, X, y, hypers):
        self.regressor = KNeighborsRegressor(hypers).fit(X, y)

class RBFSVR(Regressor):
    def train(self, X, y, hypers):
        self.regressor = NuSVR(kernel='rbf', C=hypers[0], gamma=hypers[1]).fit(X, y)

class RBFNR(Regressor):
    def train(self, X, y, hypers):
        #self.regressor = pyrb.train_ols(X, y.reshape(len(y),1), 0.01, 0.3, verbose=True)
        self.regressor = pyrb.train_exact(X, y.reshape(len(y),1), 9.0/hypers)

    def test(self, X, real):
        if self.regressor == None:
            return 1e10
        predicted = np.array(self.regressor.sim(X))
        predicted = predicted.reshape(1,len(real))
        return mse(real, predicted[0])

def eval(regressor, TRX, TRy, TEX, TEy, hypers):
    minmse=1e100
    for hs in hypers:
        mse = 0.0
        print "hypers " + str(hs)
        # just one iteration
        for TR2,TE2 in cv.ShuffleSplit(len(TRX), n_iter=1, test_size=0.25, random_state=0):
            regressor.train(TRX[TR2], TRy[TR2], hs)
            mse = mse + regressor.test(TRX[TE2], TRy[TE2])
        if mse < minmse:
            minmse=mse
            minhyper=hs
    print "Best hyperparameters " + str(minhyper)
    regressor.train(TRX,TRy, minhyper)
    return regressor.test(TEX, TEy)

def mlr(xtr, ytr, xte, yte, cvalues = None):
    if(cvalues == None):
        cvalues = []
        for i in range(0,9):
            cvalues.append(pow(10,i-3))
    reg = MLR()
    return eval(reg, xtr, ytr, xte, yte, cvalues)

def knn(xtr, ytr, xte, yte, ks = [1,2,5,10,20]):
    reg = KNNR()
    return eval(reg, xtr, ytr, xte, yte, ks)

def svr(xtr, ytr, xte, yte, svrh = None):
    if(svrh == None):
        svrh = []
        for i in range(0,9):
            for j in range(0,9):
                svrh.append([])
                svrh[-1].append(pow(10,i-3))
                svrh[-1].append(pow(10,j-3))
    reg = RBFSVR()
    return eval(reg, xtr, ytr, xte, yte, svrh)

def rfr(xtr, ytr, xte, yte, ks = [2,3,4,5]):
    reg = RFR()
    return eval(reg, xtr, ytr, xte, yte, ks)

def ffnnr(xtr, ytr, xte, yte, ks = [1,2,5,10,20]):
    reg = FFNNR()
    return eval(reg, xtr, ytr, xte, yte, ks)

# I didn't found a straightforward way to set the number of neurons in the hidden layer with the pyradbas so I coded
# this part on R.
def rbfnr(xtr, ytr, xte, yte, ks = [2,5,10,20]):
    reg = RBFNR()
    return eval(reg, xtr, ytr, xte, yte, ks)

