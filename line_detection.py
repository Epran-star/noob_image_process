import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def line_kernal(pic,i,j,k):
    kernel = []
    kernel.append(np.array([[-1, -1, -1],
                       [2, 2, 2],
                       [-1, -1, -1]]))
    
    kernel.append(np.array([[2, -1, -1],
                       [-1, 2, -1],
                       [-1, -1, -1]]))
    
    kernel.append(np.array([[-1, 2, -1],
                       [-1, 2, -1],
                       [-1, 2, -1]]))
    
    kernel.append(np.array([[-1, -1, 2],
                       [-1, 2, -1],
                       [2, -1, -1]]))
    sub_img = pic[i - 1:i + 2, j - 1:j + 2]

    Max = 0
    for i in range(4):
        result = kernel[i] * sub_img
        result = abs(np.sum(result)) * k
        if Max < result:
            Max = result
        
    return round(result)
    

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)


img_laplacian_sharpening = np.zeros((len(gray),len(gray[0])),dtype='uint8')
for i in range(1,len(gray)-1):
    for ii in range(1,len(gray[0])-1):
        img_laplacian_sharpening[i][ii] = line_kernal(gray,i,ii,0.3)
        
cv2.imwrite('line_detection.jpg',img_laplacian_sharpening)


