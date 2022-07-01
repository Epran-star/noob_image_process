import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def smooth_kernal(pic,i,j,k):
    #Weight of adjacent pixels:
    #1,2,1
    #2,4,2
    #1,2,1
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
    sub_img = pic[i - 1:i + 2, j - 1:j + 2]
    result = kernel * sub_img
    result = np.sum(result)/16 * k
    return round(result)

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)


img_smooth = np.zeros((len(gray),len(gray[0])),dtype='uint8')

for i in range(1,len(gray)-1):
    for ii in range(1,len(gray[0])-1):
        img_smooth[i][ii] = smooth_kernal(gray,i,ii,1)
        

cv2.imwrite('smooth.jpg',img_smooth)
