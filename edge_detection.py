import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def marr_hildreth55(pic,i,j):
    kernel = np.array([[0, 0, -1, 0, 0],
                       [0,-1, -2, 0, 0],
                       [-1,-2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0],])
    sub_img = pic[i - 2:i + 3, j - 2:j + 3]
    result = kernel * sub_img
    result = abs(np.sum(result))
    return round(result)

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)


marr_Hildreth = np.zeros((len(gray),len(gray[0])),dtype='uint8')
for i in range(2,len(gray)-2):
    for ii in range(2,len(gray[0])-2):
        marr_Hildreth[i][ii] = marr_hildreth55(gray,i,ii)
        
cv2.imwrite('marr_Hildreth.jpg',marr_Hildreth)
    
canny = cv2.Canny(gray, 30, 150)
cv2.imwrite('canny.jpg',canny)
        



