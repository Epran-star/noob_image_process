import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)

_, binary = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY)
cv2.imwrite('binary.jpg',binary)

####腐蚀
I = binary
# 创建矩形结构元
s = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# 腐蚀图像，迭代次数采用默认值1
r = cv2.erode(I, s)
# 显示原图和腐蚀后的结果
plt.subplot(131)
plt.axis('off')
plt.imshow(I, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.imshow(r, cmap='gray')


####膨胀
# 结构元半径
R = 1
# 创建结构元
s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*R+1, 2*R+1))
# 膨胀图像
d = cv2.dilate(r, s)
# 显示膨胀效果
plt.subplot(133)
plt.axis('off')
plt.imshow(d, cmap='gray')
plt.show()

####膨胀
# 结构元半径
R = 1
# 创建结构元
s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*R+1, 2*R+1))
# 膨胀图像
d = cv2.dilate(I, s)
# 显示膨胀效果
plt.subplot(131)
plt.axis('off')
plt.imshow(I, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.imshow(d, cmap='gray')

####腐蚀
I = binary
# 创建矩形结构元
s = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# 腐蚀图像，迭代次数采用默认值1
r = cv2.erode(d, s)
# 显示原图和腐蚀后的结果
plt.subplot(133)
plt.axis('off')
plt.imshow(r, cmap='gray')
plt.show()

print(I)
