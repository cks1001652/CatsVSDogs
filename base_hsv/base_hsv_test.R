test <- function(fit_train, dat_test){
  library("e1071")
  pred <- predict(fit_train, dat_test)
  return(pred)
}

