# # library("EBImage")
# 
# feature <- function(img_dir, img_name, data_name=NULL){
  
#  
#   ### determine img dimensions
# #   img0 <-  readImage(paste0(img_dir, img_name))
# #   mat1 <- as.matrix(img0)
#   n_r <- as.integer(1)
#   n_c <- as.integer(1)
#   n_d <- as.integer(3)
#   
#   ### store vectorized pixel values of images
#   dat <- array(dim=c(n_files, n_r*n_c*n_d)) 
# 
#     
#     
# 
# for(i in 1:n_files){
# #   tryCatch({
#     img <- readImage(paste0(img_dir, img_name)[i])}
# #     }, warning=function(w){print(i)},error = function(e) {print(i)})
# #     img <- resize(img,n_r,n_c)
#      dat[i,] <- as.vector(img)}
# if(!is.null(data_name)){
#   save(dat, file=paste0( data_name, ".RData"))
# }
# return(dat)
# }
# 
#  
# 
# 
#   
#   
#  
#   
#   ### output constructed features
#  

feature <- function(img_dir,img_name,data_name=NULL){
#initial parameters   
  n_files <- length(img_name)
#   n_files <- length(trainname)
  nR <- 10
  nG <- 10
  nB <- 10
  rBin <- seq(0, 1, length.out=nR)
  gBin <- seq(0, 1, length.out=nG)
  bBin <- seq(0, 1, length.out=nB)
  rgb_feature <- matrix(NA,nrow=n_files,ncol = nR*nG*nB)
#Readin data and extract color feature   
  system.time(for(i in 1:n_files){
    if(trunc(i/5)*5==i){print(i)}
    mat <- readImage(paste0(img_dir, img_name)[i])
#     mat <- imageData(img)
    freq_rgb <- as.data.table(table(factor(findInterval(mat[,,1], rBin), levels=1:nR), 
                                    factor(findInterval(mat[,,2], gBin), levels=1:nG), 
                                    factor(findInterval(mat[,,3], bBin), levels=1:nB)))
    rgb_feature[i,] <- as.numeric(freq_rgb$N)/(ncol(mat)*nrow(mat)) # normalization
  })
if(!is.null(data_name)){
  save(rgb_feature, file=paste0("./output/feature_", data_name, ".RData"))
}
  return(rgb_feature)
  
}
  



















