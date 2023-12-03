from typing import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from VO import *

img1 = cv.imread('data_KITTI/00/image_0/000001.png',cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('data_KITTI/00/image_0/000000.png',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


matches = VO.match_features(VO,des1,des2)
matches = sorted(matches,key=lambda x:x.distance)

# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:200],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:200],None,flags=2) # show first 200 matches
plt.imshow(img3),plt.show()