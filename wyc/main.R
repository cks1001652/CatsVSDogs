setwd("~/GitHub/cycle3cvd-team7/wyc/")
library(data.table)
library(caret)
#Extract the class labels from the image name
dir_images <- "./data"
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
#Sample test and train data 
set.seed(14249)
trainindex <- createDataPartition(label, p = .75,
                                  list = FALSE,
                                  times = 1)
trainname <- dir_names[trainindex]
testname <- dir_names[-trainindex]
trainlabel <- as.factor(label[trainindex])
testlabel <- as.factor(label[-trainindex])
#data set sift
set.seed(14249)
trainindex <- createDataPartition(label, p = .75,
                                  list = FALSE,
                                  times = 1)

data_train_sift <- featurepy[trainindex,]
data_test_sift <- featurepy[-trainindex,]
trainlabel <- as.factor(label[trainindex])
testlabel <- as.factor(label[-trainindex])


#Image Analysis Toll
# source("https://bioconductor.org/biocLite.R")
# biocLite("EBImage")
library("EBImage")
source("feature.R")

#feature construction for train and test
img_test_dir <- img_train_dir <- "./data/"

tm_feature_train <- system.time(data_train <- feature(img_train_dir, trainname))
tm_feature_test <- system.time(data_test <- feature(img_test_dir, testname))

# tm_feature_train <- data_train
# tm_feature_train <- cbind(tm_feature_train,label)
save(data_train, file="./output/feature_train.RData")
save(data_test, file="./output/feature_test.RData")
load(file="./output/feature_train.RData")
load(file="./output/feature_test.RData")
data_train_sift_rgb <- cbind(data_train,data_train_sift)
data_test_sift_rgb <- cbind(data_test,data_test_sift)
# load(file="./output/feature_test.RData")
#lsvm training 
source("./train.R")
tm_train <- system.time(fit_train <- train(data_train,trainlabel))
#tm_trainsift <- system.time(fit_trainsift <- train(data_train_sift,trainlabel)) 
#tm_trainsiftrgb <- system.time(fit_trainsiftrgb <- train(data_train_sift_rgb,trainlabel))
save(fit_train,file="./output/fit_train.RData")
 
source("./test.R")
### Make prediction 
tm_test <- system.time(pred_test <- test(fit_train, data_test))
#tm_testsift <- system.time(pred_testsift <- test(fit_trainsift,data_test_sift))
#tm_testsiftrgb <- system.time(pred_testsiftrgb <- test(fit_trainsiftrgb,data_test_sift_rgb)) 
save(pred_test, file="./output/pred_test.RData")
load("./output/pred_test.RData")
sum(pred_test==testlabel)/length(testlabel)
# sum(pred_testsift==testlabel)/length(testlabel)
# sum(pred_testsiftrgb==testlabel)/length(testlabel)

library(rPython)
python.load("~/GitHub/cycle3cvd-team7/wyc/python/test.py")

featurepy <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/feature.txt", header=FALSE)
featurepy$V1 <- gsub('[[]','',featurepy$V1)
featurepy$V1 <- as.numeric(featurepy$V1)
featurepy$V5 <- gsub('[]]','',featurepy$V5)
featurepy$V5 <- as.numeric(featurepy$V5)
str(featurepy)
class(featurepy)
sum(duplicated(featurepy[,1]))
sum(duplicated(featurepy[,2]))
sum(duplicated(featurepy[,3]))
sum(duplicated(featurepy[,4]))
sum(duplicated(featurepy[,5]))


