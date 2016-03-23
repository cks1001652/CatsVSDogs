load("base_hsv/base_hsv_feature_test.RData")
load("base_hsv/base_hsv_feature_train.RData")


test.d = as.data.frame(data_test) ### test data
label.t = ifelse(da[,10]==1,1,0)
dat = as.data.frame(cbind(data_train,label.t))

### logistic
g = glm(trainlabel~.,data=dat,family=binomial)  
### has warning message: glm.fit:algorithm did not converge
### which means there should be perfect linear separating


### 
library(MASS)
new.dat = dat[,-c(seq(10,240,by=10),117,118,167,168,177,178,227,237,238,241:360)] ### remove collinear
lda.fit = lda(label.t~.,dat=new.dat)
pred = test(lda.fit,test.d)
table(pred$class,testlabel) ### 0.33, bad


### tree
library(tree)
tree.fit=tree(label.t~. , data=dat)
pred_test <- test(tree.fit, test.d)
table(pred_test, testlabel) ### 0.34 bad


### deeplearning
library(h2o)
h2o.init()
train_hex = as.h2o(new.dat)
dl = h2o.deeplearning(x=1:207,y=208,train_hex)
test_hex = as.h2o(test.d)
pred.prob = h2o.predict(dl,test_hex)
pred = rep(1,length(testlabel))
pred[as.vector(pred.prob)<0.5] = 0
table(pred,testlabel) ### 0.34, bad


### navie bayes
new.dat[,208] = as.factor(new.dat[,208])
train_hex = as.h2o(new.dat)
nb = h2o.naiveBayes(x=1:207,y=208,train_hex)
pred.prob = h2o.predict(nb,test_hex)
pred = as.vector(ifelse(pred.prob[,2]<pred.prob[,3],1,0))
table(pred,testlabel) ### 0.33, bad


### gbm
gbm = h2o.gbm(x=1:207,y=208,train_hex)
pred.prob = h2o.predict(gbm,test_hex)
pred = as.vector(ifelse(pred.prob[,2]<pred.prob[,3],1,0))
table(pred,testlabel) ### 0.3, bad




