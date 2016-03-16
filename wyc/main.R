setwd("~/GitHub/cycle3cvd-team7/wyc/")
#Extract the class labels from the image name
dir_images <- "./data"
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

#Image Analysis Toll
source("https://bioconductor.org/biocLite.R")
biocLite("EBImage")
library("EBImage")
source("feature.R")
# 
img_train_dir <- "./data/"
tm_feature_train <- system.time(dat_train <- feature(img_train_dir, dir_names))

# img <- readImage(paste0(img_train_dir,dir_names)[2])
# readImage
# img_size <- resize(img, 500, 500)
# display(img_size)