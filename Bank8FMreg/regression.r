doit <- function() {
setwd("/Users/mac/PycharmProjects/mlcodes/regression/mo444ex5")
library(Rcpp)
library(Metrics)
library(RSNNS)
trpath = "./Bank/Bank8FM/bank8FM.data";
tepath = "./Bank/Bank8FM/bank8FM.test";
frame = read.csv(trpath, header=F, sep=' ')
tr = data.matrix(frame)
d = dim(tr)
xtr = tr[,1:d[2]<d[2]-1]
ytr =  as.matrix(tr[,1:d[2]==d[2]-1])
frame = read.csv(tepath, header=F, sep=' ')
te = data.matrix(frame)
d = dim(te)
xte = te[,1:d[2]<d[2]-1]
yte =  as.matrix(te[,1:d[2]==d[2]-1])

ks = c(2,5,10,20)
bestmse = 1e100;
bestk = 0;
idx = sample(1:d[1],floor(d[1]*0.75))
xtrtr = xtr[idx,]
xtrte = xtr[-idx,]
ytrtr = ytr[idx,]
ytrte = ytr[-idx,] 
for(k in  ks) {
  model <- rbf(xtrtr, ytrtr, size=k)
  predicted = predict(model, xtrte)
  curmse = mse(ytrte, predicted)
  print(curmse)
  if(curmse < bestmse) {
    bestmse = curmse
    bestk = k
  }
}
model <- rbf(xtr, ytr, size=bestk)
predicted = predict(model, xte)
gmse = mse(yte, predicted)
print(k)
print(gmse)
}

