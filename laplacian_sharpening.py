import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def sharpening_kernal(pic,i,j,k):
    #Weight of adjacent pixels:
    #0, 1,0
    #1,-4,1
    #0, 1,0
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    sub_img = pic[i - 1:i + 2, j - 1:j + 2]
    result = kernel * sub_img
    result = abs(np.sum(result)) * k
    return round(result)
    

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)


img_laplacian_sharpening = np.zeros((len(gray),len(gray[0])),dtype='uint8')
for i in range(1,len(gray)-1):
    for ii in range(1,len(gray[0])-1):
        img_laplacian_sharpening[i][ii] = sharpening_kernal(gray,i,ii,0.3)
        
cv2.imwrite('laplacian_sharpening_border.jpg',img_laplacian_sharpening)

for i in range(len(gray)):
    for ii in range(len(gray[0])):
        pixel = int(img_laplacian_sharpening[i][ii]) + int(gray[i][ii])
        if pixel>255:
            img_laplacian_sharpening[i][ii] = 255
        else:
            img_laplacian_sharpening[i][ii] = pixel

cv2.imwrite('laplacian_sharpening.jpg',img_laplacian_sharpening)
