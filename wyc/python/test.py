# a=1
# print 1

# import numpy #as np 
# import cv2
# import os
# 0 will give the grey color
# img = cv2.imread('jurassic_world.jpg',0)
# 
# cv2.imshow('image',img)
# cv2.waitKey(0)
# # 
# # # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# # # cv2.imshow('image',img)
# # # #MUST use it to display the image
# # # cv2.waitKey(0)
# # 
# cv2.destroyAllWindows()

# cv2.imwrite('jurassic_world2.jpg',img)
# #############Test run for saving and showing img##########################
# import numpy as np
# import cv2

# img = cv2.imread('jurassic_world.jpg')
# cv2.imshow('image',img) #to display in a window
# k = cv2.waitKey(0) 
# if k == 27: #wait for ESC to exit
# 	cv2.destroyAllWindows()
# elif k == ord('s'): #wait for 's' key to save and exit
# 	cv2.imwrite('testsave.jpg',img)
# 	cv2.destroyAllWindows()

#SIFT
# import numpy as np
# import cv2
# img = cv2.imread('Abyssinian_8.jpg',1)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# (kps, descs) = sift.detectAndCompute(gray, None)
# print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# img=cv2.drawKeypoints(gray,kps, img)
# cv2.imwrite('sift_Abyssinian_8.jpg',img)
# descs_list = descs.tolist()
# cv2.imshow('cat',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# ##Harris Corner Detector 
# import cv2
# import numpy as np
# filename = ''
# filename_list =[]
# import os
# import sys
# import re
# path = "./Github/cycle3cvd-team7/wyc/data/"
# dirlist = os.listdir(path)
# for file in dirlist:
#   a,b=os.path.splitext(file)
#   filename_list.append(a)
# 
# # filename_list
# # print filename_list[0:10]
# 
# #Get ready for the labels 
# label_list=[]
# labelpath = "./Github/cycle3cvd-team7/wyc/list.txt"
# label = open(labelpath)
# for line in label:
#   line = line.rstrip()
#   line = line.lstrip()
#   line = str.split(line)[0],str.split(line)[2]
#   label_list.append(line)
# 
# label_list[0:6]=[] #start from the name of the animals 
# # Test
# # print label_list[0:10]
# # 
# # print label_list[0][0]
# 
# # print label_list[:][1]
# 
# label_dict={}
# for j in label_list:
#   for i in filename_list:
#     if i in j[1]:
#       label_dict[i]=j[1]
#   
# 
# for x in label_dict:
#   print x
#   for y in label_dict[x]:
#     print label_dict[x][y]

import cv2
import numpy as np
import os

#############get image path###########
# train_names = os.listdir(path)
# train_names = train_names[1:len(train_names)]
detect = cv2.xfeatures2d.SIFT_create()

extract = cv2.xfeatures2d.SIFT_create()
def getimagedata(path):
  train_names = os.listdir(path)
  train_names = train_names[0:len(train_names)+1]
  image_paths=[]
  image_classes = []
  class_id =0
  for train_name in train_names:
    dir = os.path.join(path,train_name)
  #   class_path = [os.path.join(dir,f) for f in os.listdir(dir)]
    image_paths.append(dir)
  #   image_classes += [class_id]*len(class_path)
  #   class_id +=1
  return image_paths

# sift extraction
def siftprocessimage(image_paths):  
  descriptors=[]
  for image_path in image_paths:
    im = cv2.imread(image_path)
    gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    kpts =detect.detect(gray)
    kpts,des = extract.compute(im,kpts)
    descriptors.append(des)
  return descriptors

def getdata(feature_det,bow_extract,path):
    im = cv2.imread(path)
    featureset = bow_extract.compute(im, feature_det.detect(im))
    return featureset   

def train(descriptors,image_paths):
  flann_p=dict(algorithm = 1, tree = 5)
  matcher = cv2.FlannBasedMatcher(flann_p,{})
  bow_extract  =cv2.BOWImgDescriptorExtractor(extract,matcher)
  bow_train = cv2.BOWKMeansTrainer(5)
  for des in descriptors:
    bow_train.add(des)
    
  voc = bow_train.cluster()
  bow_extract.setVocabulary(voc)
  traindata = []
  for image_path in image_paths:
#     im = cv2.imread(image_path)
    featureset = getdata(detect,bow_extract,image_path)
    traindata.extend(featureset)
  return traindata
#read result 


path ='../data/'
r1=getimagedata(path)
r2=siftprocessimage(r1)
r3=train(r2,r1)

f=open('feature.txt','w')
for i in r3:
  a=str(i.tolist())
  f.write(a+'\n')
  
f.close()


# 
# 
# def train(descriptors,image_classes,image_paths):  
#     flann_params = dict(algorithm = 1, trees = 5)     
#     matcher = cv2.FlannBasedMatcher(flann_params, {})
#     bow_extract  =cv2.BOWImgDescriptorExtractor(sift,matcher)
#     bow_train = cv2.BOWKMeansTrainer(20)
#     for des in descriptors:
#         bow_train.add(des)
#         
#     voc = bow_train.cluster()
#     bow_extract.setVocabulary( voc )
#     traindata = []
# 
#     for imagepath in image_paths:
#         featureset = getImagedata(feature_det,bow_extract,imagepath)
#         traindata.extend(featureset)
#     clf = LinearSVC()
#     clf.fit(traindata, np.array(image_classes))
#     joblib.dump((voc,clf), "imagereco.pkl", compress=3)
# 
# def getdata(feature_det,bow_extract,path):
#     im = cv2.imread(path)
#     featureset = bow_extract.compute(im, feature_det.detect(im))
#     return featureset   
# # print descriptors[1]
# # import numpy as np
# import cv2
# img = cv2.imread('Abyssinian_8.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# (kps, descs) = sift.detectAndCompute(gray, None)
# print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# img=cv2.drawKeypoints(gray,kps, img)
# cv2.imwrite('sift_keypoints.jpg',img)
# descs_list = descs.tolist()

# feature_det = cv2.FeatureDetector_create("SIFT")
# descr_ext = cv2.DescriptorExtractor_create("SIFT")
# 
# def getImageMetadata(path):
#     train_names = os.listdir(path)
#     image_paths = []
#     image_classes = []
#     id = 0
#     for train_name in train_names:
#         dir = os.path.join(path, train_name)
#         class_path = [os.path.join(dir, f) for f in os.listdir(dir)]
#         image_paths+=class_path
#         image_classes+=[class_id]*len(class_path)
#         id+=1
#     return image_paths,image_classes
#    
# 
# import argparse as ap
# import cv2
# # import imutils 
# import numpy as np
# import os
# 
# train_path='../data2'
# training_names = os.listdir(train_path)
# 
# # Get all the path to the images and save them in a list
# # image_paths and the corresponding label in image_paths
#     
# 
# # Create feature extraction and keypoint detector objects
# fea_det = cv2.FeatureDetector_create("SIFT")
# des_ext = cv2.DescriptorExtractor_create("SIFT")
# 
# # List where all the descriptors are stored
# des_list = []
# 
# for image_path in image_paths:
#     im = cv2.imread(image_path)
#     kpts = fea_det.detect(im)
#     kpts, des = des_ext.compute(im, kpts)
#     des_list.append((image_path, des))   
#     
# # Stack all the descriptors vertically in a numpy array
# descriptors = des_list[0][1]
# for image_path, descriptor in des_list[1:]:
#     descriptors = np.vstack((descriptors, descriptor))  
# 
# # Perform k-means clustering
# k = 100
# voc, variance = kmeans(descriptors, k, 1) 
# 
# # Calculate the histogram of features
# im_features = np.zeros((len(image_paths), k), "float32")
# for i in xrange(len(image_paths)):
#     words, distance = vq(des_list[i][1],voc)
#     for w in words:
#         im_features[i][w] += 1
# 
# # Perform Tf-Idf vectorization
# nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
# idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
# 
# # Scaling the words
# stdSlr = StandardScaler().fit(im_features)
# im_features = stdSlr.transform(im_features)
# 
# # Train the Linear SVM
# clf = LinearSVC()
# clf.fit(im_features, np.array(image_classes))
# 
# # Save the SVM
# joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3) 
