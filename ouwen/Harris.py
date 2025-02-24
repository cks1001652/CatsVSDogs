# this is aborted due to the difficult in capturing face from the harris corner
#it was difficult to scan for the red dots and make a square
# this is also aborted because of the need to set for optimal value for each image
import cv2
import numpy as np
import os
os.chdir("C:\\Users\ouwen\Desktop\images")
filename = 'Abyssinian_13.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
