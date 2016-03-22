library(EBImage)
library(grDevices)
library(ggplot2)
library(lattice)
library(caret)


### Extract HSV for single image
extract.features <- function(img){
  mat <- imageData(img)
  mat_rgb <- mat
  dim(mat_rgb) <- c(nrow(mat)*ncol(mat), 3)
  mat_hsv <- rgb2hsv(t(mat_rgb))
  nH <- 10
  nS <- 6
  nV <- 6
  hBin <- seq(0, 1, length.out=nH)
  sBin <- seq(0, 1, length.out=nS)
  vBin <- seq(0, 0.005, length.out=nV) 
  freq_hsv <- as.data.frame(table(factor(findInterval(mat_hsv[1,], hBin), levels=1:nH), 
                                  factor(findInterval(mat_hsv[2,], sBin), levels=1:nS), 
                                  factor(findInterval(mat_hsv[3,], vBin), levels=1:nV)))
  hsv_feature <- as.numeric(freq_hsv$Freq)/(ncol(mat)*nrow(mat)) # normalization
  return(hsv_feature)
}


features <- function(img_dir,img_name){
  n = length(img_name)
  count = 0
  x <- matrix(rep(0,n*360),n,360)
  for (i in 1:n){
    tryCatch({
      img <- readImage(paste0(img_dir,img_name[i],".jpg"))
    }, error =function(err){count=count+1},
    finally = {x[i,] <- extract.features(img)})
  }
  return(x)
}
