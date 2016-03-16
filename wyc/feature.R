# library("EBImage")

feature <- function(img_dir, img_name, data_name=NULL){
  n_files <- length(list.files(img_dir))
  n_files <- as.numeric(n_files)
  ### determine img dimensions
#   img0 <-  readImage(paste0(img_dir, img_name))
#   mat1 <- as.matrix(img0)
  n_r <- 500
  n_c <- 500
  
  ### store vectorized pixel values of images
  dat <- array(dim=c(n_files, n_r*n_c)) 
  for(i in 1:n_files){
    img <- readImage(paste0(img_dir, img_name)[i])
    img <- resize(img,n_r,n_c)
    dat[i,] <- as.vector(img)
  }
  if(i%%10==0){print(i)}
  ### output constructed features
  if(!is.null(data_name)){
    save(dat, file=paste0( data_name, ".RData"))
  }
  return(dat)
}
