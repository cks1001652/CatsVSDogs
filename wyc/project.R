library()
newfeature_2000_800_new <- read.csv("~/GitHub/cycle3cvd-team7/wyc/python/featuresift_2000_800_new.txt", header=FALSE)
library(data.table)
library(caret)
library("EBImage")
newfeature_2000_800_new$V1 <- gsub('[[]','',newfeature_2000_800_new$V1)
newfeature_2000_800_new$V1 <- as.numeric(newfeature_2000_800_new$V1)
newfeature_2000_800_new$V800 <- gsub('[]]','',newfeature_2000_800_new$V800)
newfeature_2000_800_new$V800 <- as.numeric(newfeature_2000_800_new$V800)
str(newfeature_2000_800_new)
dim(newfeature_2000_800_new)
feature_eval <- newfeature_2000_800_new
save(featue_eval,file="./output/feature_eval.Rdata")
source("./test.R")
load(file='./output/fit_trainsiftrgb_800.Rdata')
tm_testsiftrgb_800 <- system.time(pred <- test(fit_trainsiftrgb_800,feature_eval))
source("./feature.R")
img_dir <- "./validate/"
img_name <- list.files(img_dir)
feature_eval_1 <- feature(img_dir,img_name)
f2 <- cbind(feature_eval_1,img_name)
img_name[2]


f2[,1001] <- gsub('[img_valid_]','',f2[,1001])
f2[,1001] <- gsub('[.jp]','',f2[,1001])
f2[,1001] <- as.numeric(f2[,1001])
str(f2[,1001])
#rbg order 
f2.1 <- f2[order(as.numeric(f2[,1001])),]
f2.2 <- f2.1[,1:1000]
f2.3 <- rbind(f2.2[1:1105,],rep(0,1000))
f2.4 <- rbind(f2.3,f2.2[1106:1999,])
dim(f2.4)
#sift
s2<- cbind(feature_eval,img_name)
s2[,801] <- gsub('[img_valid_]','',s2[,801])
s2[,801] <- gsub('[.jp]','',s2[,801])
s2[,801] <- as.numeric(s2[,801])
str(s2[,801])
s2.1[,801] <- s2[order(as.numeric(s2[,801])),]
feature_eval <- cbind(f2.4,s2.1)
feature_eval <- feature_eval[,-1]
dim(feature_eval)
newfeature_2000_800_new$V1 <- as.numeric(newfeature_2000_800_new$V1)