#library
setwd("~/GitHub/cycle3cvd-team7/wyc/")
library(data.table)
library(caret)
library(randomForest)
library(e1071)
# Import the data
#data label
dir_images <- "./data/"
dir_names <- list.files(dir_images)
breed_name <- rep(NA, length(dir_names))
for(i in 1:length(dir_names)){
  tt <- unlist(strsplit(dir_names[i], "_"))
  tt <- tt[-length(tt)]
  breed_name[i] = paste(tt, collapse="_", sep="")
}
cat_breed <- c("Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau",
               "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx")

iscat <- breed_name %in% cat_breed
y_cat <- as.numeric(iscat)
iscat[iscat==TRUE] <- 1
iscat[iscat==FALSE] <-2 
label <- iscat
label <- as.factor(label)
#seperate the train and test data
set.seed(14249)
trainindex <- createDataPartition(label, p = .75,
                                  list = FALSE,
                                  times = 1)
# trainname <- dir_names[trainindex]
# testname <- dir_names[-trainindex]
trainlabel <- as.factor(label[trainindex])
testlabel <- as.factor(label[-trainindex])
#load the feature for rgb, sift500,sift800
# #no feature saving for rgb
# load(file="./output/featurepy_499.Rdata")
# load(file="./output/featurepy_800.Rdata")
#load train and test data
#rgb 
load(file="./output/feature_train.RData")
load(file="./output/feature_test.RData")
###rgb+sift499
load(file="./output/train_sift_499.Rdata")
load(file='./output/test_sift_499.Rdata')
###rgb+sift800
load(file="./output/train_sift_800.Rdata")
load(file='./output/test_sift_800.Rdata')


#Data training
#rgb_base model
source("./train.R")
# tm_train <- system.time(fit_train <- train(data_train,trainlabel))
#  user  system elapsed 
# 58.410   0.264  58.763  

# tm_trainsiftrgb_499 <- system.time(fit_trainsiftrgb_499 <- train(data_train_sift_rgb_499,trainlabel))
# user  system elapsed 
# 228.602   0.841 230.336 

# tm_trainsiftrgb_800 <- system.time(fit_trainsiftrgb_800 <- train(data_train_sift_rgb_800,trainlabel))
# user  system elapsed 
# 250.534   0.847 251.925 
# save(fit_train,file="./output/fit_train.Rdata")
# save(fit_trainsiftrgb_499,file='./output/fit_trainsiftrgb_499.Rdata')
# save(fit_trainsiftrgb_800,file='./output/fit_trainsiftrgb_800.Rdata')

load(file='./output/fit_train.Rdata')
load(file='./output/fit_trainsiftrgb_499.Rdata')
load(file='./output/fit_trainsiftrgb_800.Rdata')

#Test the model
source("./test.R")
# tm_test <- system.time(pred_test <- test(fit_train, data_test))
# user  system elapsed 
# 6.068   0.017   6.084 

# tm_testsiftrgb_499 <- system.time(pred_testsiftrgb_499 <- test(fit_trainsiftrgb_499,data_test_sift_rgb_499))
# user  system elapsed 
# 20.857   0.037  20.890 
# tm_testsiftrgb_800 <- system.time(pred_testsiftrgb_800 <- test(fit_trainsiftrgb_800,data_test_sift_rgb_800))
# user  system elapsed 
# 26.663   0.065  26.759 

# save(pred_test,file='./output/pred_test.Rdata')
# save(pred_testsiftrgb_499,file='./output/pred_testsiftrgb_499.Rdata')
# save(pred_testsiftrgb_800,file='./output/pred_testsiftrgb_800.Rdata')

load(file='./output/pred_test.Rdata')
load(file='./output/pred_testsiftrgb_499.Rdata')
load(file='./output/pred_testsiftrgb_800.Rdata')

#Test successful rate without cv
sum(pred_test==testlabel)/length(testlabel)
sum(pred_testsiftrgb_499==testlabel)/length(testlabel)
sum(pred_testsiftrgb_800==testlabel)/length(testlabel)
# cv
tune.out=tune(svm,data_train,trainlabel,kernel="linear",ranges=list(cost=c(10,100,1000)))
summary(tune.out)
# tune.out=tune(svm,data_train_sift_499,trainlabel,kernel="linear",ranges=list(cost=c(10,100,1000)))
# summary(tune.out)
# Parameter tuning of ‘svm’:
# - sampling method: 10-fold cross validation 
# - best parameters:
#   cost:100
# - best performance: 0.3157395 
# - Detailed performance results:
#   cost     error 
# 1   10 0.3274969 
# 2  100 0.3157395 
# 3  1000 0.3761188
tune.out=svm(data_train_sift_800,trainlabel,kernel="linear",cross = 5)
summary(tune.out)
#n=7377,basedmodel_p=1000
#n=7377,basedmodel_p=1000+
#n=7377,basedmodel_p=1000
# 
# feature_eval <- 
