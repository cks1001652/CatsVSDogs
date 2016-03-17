
library(e1071)
# model <- svm(x=tm_feature_train,y=label,kernel = "linear",scale=F)
# model

train <- function(dat_train, label_train){
  model <- svm(dat_train,label_train,kernal="linear",scale=F)
  return(model)
}
