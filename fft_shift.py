import cv2
import numpy as np
import matplotlib.pyplot as plt

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)

cv2.imwrite('gray.jpg',gray)

#傅里叶变换
fft = np.fft.fft2(gray)

#将频谱低频从左上角移动至中心位置
fft_shift = np.fft.fftshift(fft)

#频谱图像双通道复数转换为0-255区间
result = 20*np.log(np.abs(fft_shift))

#显示图像
plt.imshow(result, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.show()

#还原
'''
img_center = np.fft.ifftshift(fft_shift)                          # 逆傅里叶变换
img_lpf = np.abs(np.fft.ifft2(img_center))  # 还原图像
cv2.imwrite('fft_back.jpg',img_lpf)
'''
