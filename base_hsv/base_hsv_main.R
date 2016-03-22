setwd("~/Desktop/studying/w4249 applied data models/cycle3cvd-team7")
# 
dir_images <- "./data/images/"
labels <- read.table("data/annotations/list.txt",stringsAsFactors = F)
dir_names <- paste0(labels[,1])

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
trainindex <- sample(1:length(label),length(label)*0.8)
trainname <- dir_names[trainindex]
testname <- dir_names[-trainindex]
trainlabel <- as.factor(label[trainindex])
testlabel <- as.factor(label[-trainindex])


source("base_hsv_feature.R")

#feature construction for train and test
tm_feature_train <- system.time(data_train <- features(dir_images, trainname))
tm_feature_test <- system.time(data_test <- features(dir_images, testname))

save(data_train, file=".base_hsv_feature_train.RData")
save(data_test, file=".base_hsv_feature_test.RData")


### Train a classification model with training images
source("./base_hsv_train.R")
source("./base_hsv_test.R")

# train the model with the entire training set
tm_train <- system.time(fit_train <- train(data_train, trainlabel))
save(fit_train, file="./base_hsv_fit_train.RData")

### Make prediction 
tm_test <- system.time(pred_test <- test(fit_train, data_test))
save(pred_test, file="./base_hsv_pred_test.RData")

### Summarize Running Time
cat("Time for constructing training features=", tm_feature_train[1], "s \n")
cat("Time for constructing testing features=", tm_feature_test[1], "s \n")
cat("Time for training model=", tm_train[1], "s \n")
cat("Time for making prediction=", tm_test[1], "s \n")

### calculate error rate
pred.table = table(pred_test,testlabel)
error.rate = (pred.table[1,2]+pred.table[2,1])/length(testlabel)
print(error.rate)

### 10-fold cross validation
source("./base_hsv_10-fold_cv.R")
tune.out=tune(svm,data_train, trainlabel,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out)
best = tune.out$best.model
tm_test <- system.time(pred.best <-  test(best, data_test))
pred.best.table = table(pred.best,testlabel)
best.error.rate = (pred.best.table[1,2]+pred.best.table[2,1])/length(testlabel)
print(best.error.rate)
save(best, file="./base_hsv_fit_best.RData")
save(pred.best, file="./base_hsv_pred_best.RData")
