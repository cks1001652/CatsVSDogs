library(e1071)
test <- function(fit_train, data_test){
pred <- predict(fit_train, data_test)
return(pred)
}
