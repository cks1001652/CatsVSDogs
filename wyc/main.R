setwd("~/GitHub/cycle3cvd-team7/wyc/")
library(data.table)
library(caret)
#Extract the class labels from the image name
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
#Sample test and train data 
set.seed(14249)
trainindex <- createDataPartition(label, p = .75,
                                  list = FALSE,
                                  times = 1)
trainname <- dir_names[trainindex]
testname <- dir_names[-trainindex]
trainlabel <- as.factor(label[trainindex])
testlabel <- as.factor(label[-trainindex])
# #data set sift
# set.seed(14249)
# trainindex <- createDataPartition(label, p = .75,
#                                   list = FALSE,
#                                   times = 1)
# 
# trainlabel <- as.factor(label[trainindex])
# testlabel <- as.factor(label[-trainindex])

data_train_sift_499 <- featurepy_499[trainindex,]
data_test_sift_499 <- featurepy_499[-trainindex,]

data_train_sift_800 <- featurepy_800[trainindex,]
data_test_sift_800 <- featurepy_800[-trainindex,]


data_train_sift_rgb_499 <- cbind(data_train,data_train_sift_499)
data_test_sift_rgb_499 <- cbind(data_test,data_test_sift_499)

data_train_sift_rgb_800 <- cbind(data_train,data_train_sift_800)
data_test_sift_rgb_800 <- cbind(data_test,data_test_sift_800)

save(data_train_sift_rgb_499,file="./output/train_sift_499.Rdata")
save(data_test_sift_rgb_499,file='./output/test_sift_499.Rdata')

save(data_train_sift_rgb_800,file="./output/train_sift_800.Rdata")
save(data_test_sift_rgb_800,file='./output/test_sift_800.Rdata')


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

data_train_sift_rgb_2000 <- cbind(data_train,data_train_sift_2000)
data_test_sift_rgb_2000 <- cbind(data_test,data_test_sift_2000)

# load(file="./output/feature_test.RData")
#lsvm training 
source("./train.R")
tm_train <- system.time(fit_train <- train(data_train,trainlabel))
#tm_trainsift <- system.time(fit_trainsift <- train(data_train_sift,trainlabel)) 
#tm_trainsiftrgb <- system.time(fit_trainsiftrgb <- train(data_train_sift_rgb,trainlabel))
#tm_trainsiftrgb_2000 <- system.time(fit_trainsiftrgb_2000 <- train(data_train_sift_rgb_2000,trainlabel))

save(fit_train,file="./output/fit_train.RData")
#load("./output/fit_train.RData") 
source("./test.R")
### Make prediction 
tm_test <- system.time(pred_test <- test(fit_train, data_test))
#tm_testsift <- system.time(pred_testsift <- test(fit_trainsift,data_test_sift))
#tm_testsiftrgb <- system.time(pred_testsiftrgb <- test(fit_trainsiftrgb,data_test_sift_rgb))
#tm_testsiftrgb_2000 <- system.time(pred_testsiftrgb_2000 <- test(fit_trainsiftrgb_2000,data_test_sift_rgb_2000))

save(pred_test, file="./output/pred_test.RData")
load("./output/pred_test.RData")
sum(pred_test==testlabel)/length(testlabel)
# sum(pred_testsift==testlabel)/length(testlabel)
# sum(pred_testsiftrgb==testlabel)/length(testlabel)



save(data_train_sift_rgb,file="./output/train_sift.Rdata")
save(data_test_sift_rgb,file='./output/test_sift.Rdata')
save(fit_trainsiftrgb,file='./output/fit_train_sift.Rdata')
save(trainlabel,file="./output/trainlabel.Rdata")
save(testlabel,file="./output/testlabel.Rdata")


cv.function <- function(X.train, y.train, K){
  
  n <- length(y.train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- X.train[s != i,]
    train.label <- y.train[s != i]
    test.data <- X.train[s == i,]
    test.label <- y.train[s == i]
    
    fit <- train(train.data, train.label)
    pred <- test(fit, test.data)  
    cv.error[i] <- mean(pred != test.label)  
    
  }  		
  return(c(mean(cv.error),sd(cv.error)))
  
}
cv.function(data_train_sift_rgb,trainlabel,5)




# library(rPython)
# python.load("~/GitHub/cycle3cvd-team7/wyc/python/test.py")

featurepy_5377 <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/featuresift_5377.txt", header=FALSE)
featurepy_5377$V1 <- gsub('[[]','',featurepy_5377$V1)
featurepy_5377$V1 <- as.numeric(featurepy_5377$V1)
featurepy_5377$V499 <- gsub('[]]','',featurepy_5377$V499)
featurepy_5377$V499 <- as.numeric(featurepy_5377$V499)
str(featurepy_5377)
dim(featurepy_5377)
featurepy_2000 <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/featuresift_2000.txt", header=FALSE)
featurepy_2000$V1 <- gsub('[[]','',featurep_2000y$V1)
featurepy_2000$V1 <- as.numeric(featurepy_2000$V1)
featurepy_2000$V499 <- gsub('[]]','',featurepy_2000$V499)
featurepy_2000$V499 <- as.numeric(featurepy_2000$V499)
str(featurepy_2000)
dim(featurepy_2000)
featurepy_499 <- rbind(featurepy_2000,featurepy_5377)
dim(featurepy_499)
class(featurepy_499)
save(featurepy_499,file='./output/featurepy_499.Rdata')



#import feature-800
featurepy_2000_800 <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/featuresift_2000_800.txt", header=FALSE)
featurepy_2000_800$V1 <- gsub('[[]','',featurepy_2000_800$V1)
featurepy_2000_800$V1 <- as.numeric(featurepy_2000_800$V1)
featurepy_2000_800$V800 <- gsub('[]]','',featurepy_2000_800$V800)
featurepy_2000_800$V800 <- as.numeric(featurepy_2000_800$V800)
str(featurepy_2000_800)
dim(featurepy_2000_800)

featurepy_4000_800 <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/featuresift_4000_800.txt", header=FALSE)
featurepy_4000_800$V1 <- gsub('[[]','',featurepy_4000_800$V1)
featurepy_4000_800$V1 <- as.numeric(featurepy_4000_800$V1)
featurepy_4000_800$V800 <- gsub('[]]','',featurepy_4000_800$V800)
featurepy_4000_800$V800 <- as.numeric(featurepy_4000_800$V800)
str(featurepy_4000_800)
dim(featurepy_4000_800)

featurepy_6000_800 <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/featuresift_6000_800.txt", header=FALSE)
featurepy_6000_800$V1 <- gsub('[[]','',featurepy_6000_800$V1)
featurepy_6000_800$V1 <- as.numeric(featurepy_6000_800$V1)
featurepy_6000_800$V800 <- gsub('[]]','',featurepy_6000_800$V800)
featurepy_6000_800$V800 <- as.numeric(featurepy_6000_800$V800)
str(featurepy_6000_800)
dim(featurepy_6000_800)

featurepy_7377_800 <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/featuresift_7377_800.txt", header=FALSE)
featurepy_7377_800$V1 <- gsub('[[]','',featurepy_7377_800$V1)
featurepy_7377_800$V1 <- as.numeric(featurepy_7377_800$V1)
featurepy_7377_800$V800 <- gsub('[]]','',featurepy_7377_800$V800)
featurepy_7377_800$V800 <- as.numeric(featurepy_7377_800$V800)
str(featurepy_7377_800)
dim(featurepy_7377_800)

tmp1_800 <- rbind(featurepy_2000_800,featurepy_4000_800)
tmp2_800 <- rbind(tmp1_800,featurepy_6000_800)
featurepy_800 <- rbind(tmp2_800,featurepy_7377_800)
save(featurepy_800,file='./output/featurepy_800.Rdata')


# ##########voc=cluster:2000,using 2000 images############ 
# ffeaturepy_5377 <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/featuresift_5377_2000.txt", header=FALSE)
# ffeaturepy_5377$V1 <- gsub('[[]','',ffeaturepy_5377$V1)
# ffeaturepy_5377$V1 <- as.numeric(ffeaturepy_5377$V1)
# ffeaturepy_5377$V2000 <- gsub('[]]','',ffeaturepy_5377$V2000)
# ffeaturepy_5377$V2000 <- as.numeric(ffeaturepy_5377$V2000)
# str(ffeaturepy_5377)
# 
# dim(ffeaturepy_5377)
# 
# ffeaturepy_2000 <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/featuresift_2000_2000.txt", header=FALSE)
# ffeaturepy_2000$V1 <- gsub('[[]','',ffeaturepy_2000$V1)
# ffeaturepy_2000$V1 <- as.numeric(ffeaturepy_2000$V1)
# ffeaturepy_2000$V2000 <- gsub('[]]','',ffeaturepy_2000$V2000)
# ffeaturepy_2000$V2000 <- as.numeric(ffeaturepy_2000$V2000)
# str(ffeaturepy_2000)
# 
# dim(ffeaturepy_2000)
# 
# ffeaturepy <- rbind(ffeaturepy_2000,ffeaturepy_5377)
# dim(ffeaturepy)
# class(ffeaturepy)
