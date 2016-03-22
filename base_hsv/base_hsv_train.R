
train <- function(dat_train, label_train){
  library("e1071")
  svm.fit <- svm(dat_train, label_train, kernel= "linear", scale =F)
  return(svm.fit)
}