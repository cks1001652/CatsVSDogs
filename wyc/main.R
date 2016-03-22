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

#Sample test and train data 
set.seed(14249)
trainindex <- createDataPartition(label, p = .75,
                                  list = FALSE,
                                  times = 1)
trainname <- dir_names[trainindex]
testname <- dir_names[-trainindex]
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

#lsvm training 
source("train.R")
tm_train <- system.time(fit_train <- train(data_train,trainlabel))
save(fit_train,file="./output/fit_train.RData")

source("test.R")
### Make prediction 
tm_test <- system.time(pred_test <- test(fit_train, data_test))
save(pred_test, file="./output/pred_test.RData")
sum(pred_test==testlabel)/length(testlabel)
